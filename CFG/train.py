import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import argparse
import math
from tqdm import tqdm

from data import get_dataloader
from models import KARTFlowModel
from utils import EMA, generate_1step, get_vae, load_config
from evaluate import evaluate_model


def train():
    parser = argparse.ArgumentParser(description="Train KART-Flow model.")
    parser.add_argument("--config", type=str, required=True, help="Path to dataset-specific config")
    parser.add_argument("--global_config", type=str, default="configs/global.yaml", help="Path to global config")
    args = parser.parse_args()
    
    config = load_config(args.global_config, args.config)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Hyperparams from config
    batch_size = config['training']['batch_size']
    grad_accum_steps = config['training'].get('gradient_accumulation_steps', 1)
    epochs = config['training']['epochs']
    lr = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']
    save_every = config['training']['save_every']
    eval_every = config['training'].get('eval_every', 50)
    endpoint_weight = config['training'].get('endpoint_weight', 2.0)
    ema_decay = config['training'].get('ema_decay', 0.9999)
    out_dir = config['experiment']['output_dir']
    
    os.makedirs(out_dir, exist_ok=True)
    log_file = os.path.join(out_dir, "logs.txt")
    
    # Initialize log file
    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            f.write("KART-Flow Training Logs\n")
            f.write("="*30 + "\n")
    
    dataloader = get_dataloader(batch_size=batch_size, data_dir=out_dir)
    model = KARTFlowModel(config).to(device)
    
    if config.get('model', {}).get('print_model_stats', False):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("="*30)
        print("Model Statistics:")
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print("="*30)
        

    # Multi-GPU Support via DataParallel
    if config['training']['multi_gpu'] and torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
        
    # Set explicit 0.0 weight decay for physics parameters (frequency and friction)
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not (n.endswith('K.w') or n.endswith('K.gamma'))],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n.endswith('K.w') or n.endswith('K.gamma')],
            "weight_decay": 0.0,
        },
    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=lr)
    
    # Initialize EMA Hook (binds directly to the underlying model architecture)
    base_model = model.module if isinstance(model, nn.DataParallel) else model
    ema = EMA(base_model, beta=ema_decay)
    
    # Learning Rate Scheduler
    use_scheduler = config['training'].get('use_scheduler', False)
    if use_scheduler:
        warmup_epochs = config['training'].get('warmup_epochs', 5)
        min_lr = config['training'].get('min_lr', 1e-5)
        
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return float(epoch + 1) / float(warmup_epochs)
            progress = float(epoch - warmup_epochs) / float(max(1, epochs - warmup_epochs))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            lr_target = min_lr + (lr - min_lr) * cosine_decay
            return lr_target / lr
            
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Read CFG params once outside the loop (avoid repeated dict lookups)
    p_uncond = config.get('cfg', {}).get('p_uncond', 0.1)
    num_classes = config.get('cfg', {}).get('num_classes', 10)
    guidance_scale = config.get('cfg', {}).get('guidance_scale', 3.0)
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_loss_vel = 0.0
        epoch_loss_end = 0.0
        optimizer.zero_grad()
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for i, (x0, x1, labels) in enumerate(pbar):
            x0 = x0.to(device)
            x1 = x1.to(device)
            labels = labels.to(device)
            B = x0.size(0)
            
            # Target Velocity (Static Mapping)
            v_target = x1 - x0
            
            # Sample random times
            t = torch.rand((B, 1), device=device)
            
            # CFG Label Dropout (p_uncond of the time, replace label with null_class)
            if p_uncond > 0.0:
                drop_mask = torch.rand(B, device=device) < p_uncond
                labels = torch.where(drop_mask, torch.full_like(labels, num_classes), labels)
            
            # Predict Velocity and Endpoint Displacement concurrently
            v_pred, delta_x = model(x0, t, labels, return_delta_x=True)
            
            # Term 1: Instantaneous Velocity Loss
            loss_vel = F.mse_loss(v_pred, v_target)
            
            # Term 2: Analytical Endpoint Loss
            x_pred = x0 + delta_x
            loss_end = F.mse_loss(x_pred, x1)
            
            # Total Objective
            total_loss = loss_vel + (endpoint_weight * loss_end)
            
            loss = total_loss / grad_accum_steps  # Normalize loss for accumulation
            
            loss.backward()
            
            if ((i + 1) % grad_accum_steps == 0) or (i + 1 == len(dataloader)):
                optimizer.step()
                optimizer.zero_grad()
                
                # Silently update EMA weights in the background
                ema.update(base_model)
            
            total_loss_val = total_loss.item()
            epoch_loss += total_loss_val
            epoch_loss_vel += loss_vel.item()
            epoch_loss_end += loss_end.item()
            
            pbar.set_postfix({
                'L_vel': loss_vel.item(), 
                'L_end': loss_end.item(), 
                'Total': total_loss_val
            })
            
        avg_loss = epoch_loss / len(dataloader)
        avg_loss_vel = epoch_loss_vel / len(dataloader)
        avg_loss_end = epoch_loss_end / len(dataloader)
        
        if use_scheduler:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = lr
            
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")
        
        with open(log_file, "a") as f:
            f.write(f"Epoch {epoch+1:04d} | Total: {avg_loss:.4f} | L_vel: {avg_loss_vel:.4f} | L_end: {avg_loss_end:.4f}\n")
        
        # 1-Step Analytical Inference using EMA weights
        if (epoch + 1) % save_every == 0 or epoch == 0:
            ema.apply_shadow(base_model)
            base_model.eval()
            with torch.no_grad():
                filename = os.path.join(out_dir, f"epoch_{epoch+1}.png")
                num_samples = config['inference']['num_samples']
                eval_labels = torch.randint(0, num_classes, (num_samples,), device=device)
                generate_1step(base_model, device, num_samples=num_samples, filename=filename, 
                               labels=eval_labels, guidance_scale=guidance_scale, num_classes=num_classes)
                
                # Save EMA checkpoint
                ckpt_path = os.path.join(out_dir, f"ema_model_epoch_{epoch+1}.pt")
                torch.save(base_model.state_dict(), ckpt_path)
                
            base_model.train()
            ema.restore(base_model)
            
        # Periodic Rigorous Evaluation
        if (epoch + 1) % eval_every == 0:
            print(f"\n--- Starting Evaluation for Epoch {epoch+1} ---")
            ema.apply_shadow(base_model)
            vae = get_vae(device) if base_model.in_channels == 4 else None
            metrics = evaluate_model(base_model, vae, device, config, out_dir=out_dir)
            
            if metrics:
                log_str = f"Epoch {epoch+1:04d} EVAL | " + " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
                print(log_str)
                with open(log_file, "a") as f:
                    f.write(log_str + "\n")
                    
            base_model.train()
            ema.restore(base_model)

if __name__ == '__main__':
    train()
