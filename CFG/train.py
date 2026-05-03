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
from utils import EMA, generate_1step, get_vae, load_config, pair_samples
from evaluate import evaluate_model


def train():
    parser = argparse.ArgumentParser(description="Train KART-Flow model.")
    parser.add_argument("--config", type=str, required=True, help="Path to dataset-specific config")
    parser.add_argument("--global_config", type=str, default="configs/global.yaml", help="Path to global config")
    parser.add_argument("--resume", type=str, default=None, help="Path to a full training checkpoint (.pt) to resume from")
    parser.add_argument("--resume_epoch", type=int, default=None, help="Epoch number the checkpoint was saved at (training resumes from epoch+1)")
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
        
    # Set explicit 0.0 weight decay for physics parameters (frequency)
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not n.endswith('K.w')],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n.endswith('K.w')],
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
    
    # ---- Resume from checkpoint ----
    start_epoch = 0
    if args.resume is not None:
        if args.resume_epoch is None:
            parser.error("--resume_epoch is required when --resume is specified.")
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(f"Checkpoint not found: {args.resume}")
        
        print(f"Resuming from checkpoint: {args.resume} (epoch {args.resume_epoch})")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        
        # Load model weights
        base_model.load_state_dict(ckpt['model_state_dict'])
        
        # Load optimizer state (restores momentum buffers, etc.)
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        
        # Load scheduler state to get the exact LR curve position
        if use_scheduler and 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        
        # Load EMA shadow weights
        if 'ema_state_dict' in ckpt:
            ema.load_state_dict(ckpt['ema_state_dict'])
        
        start_epoch = args.resume_epoch
        print(f"Resuming training from epoch {start_epoch + 1} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        del ckpt  # free memory
    
    # Read CFG params once outside the loop (avoid repeated dict lookups)
    p_uncond = config.get('cfg', {}).get('p_uncond', 0.1)
    num_classes = config.get('cfg', {}).get('num_classes', 10)
    guidance_scale = config.get('cfg', {}).get('guidance_scale', 3.0)
    pairing_strategy = config.get('training', {}).get('pairing_strategy', 'sliced_sorting')
    
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        epoch_loss_vel = 0.0
        epoch_loss_end = 0.0
        optimizer.zero_grad()
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        # Buffers to construct the global batch
        accum_x0, accum_x1, accum_labels = [], [], []
        
        for i, (x0_local, x1_local, labels_local) in enumerate(pbar):
            # Accumulate on CPU to save VRAM
            accum_x0.append(x0_local)
            accum_x1.append(x1_local)
            accum_labels.append(labels_local)
            
            is_step = ((i + 1) % grad_accum_steps == 0) or (i + 1 == len(dataloader))
            if not is_step:
                continue
                
            # --- GLOBAL BATCH PAIRING ---
            global_x0 = torch.cat(accum_x0, dim=0).to(device)
            global_x1 = torch.cat(accum_x1, dim=0).to(device)
            global_labels = torch.cat(accum_labels, dim=0).to(device)
            
            global_x0, global_x1, perm = pair_samples(global_x0, global_x1, strategy=pairing_strategy)
            global_labels = global_labels[perm]
            
            # --- SPLIT BACK TO LOCAL BATCHES FOR GRADIENT ACCUMULATION ---
            local_batch_sizes = [x.size(0) for x in accum_x0]
            x0_chunks = torch.split(global_x0, local_batch_sizes)
            x1_chunks = torch.split(global_x1, local_batch_sizes)
            labels_chunks = torch.split(global_labels, local_batch_sizes)
            
            step_loss = 0.0
            step_loss_vel = 0.0
            step_loss_end = 0.0
            num_chunks = len(x0_chunks)
            
            for j in range(num_chunks):
                x0 = x0_chunks[j]
                x1 = x1_chunks[j]
                labels = labels_chunks[j]
                B = x0.size(0)
                
                # Target Velocity (Static Mapping)
                v_target = x1 - x0
                
                # Sample random times
                t = torch.rand((B, 1), device=device)
                
                # CFG Label Dropout
                if p_uncond > 0.0:
                    drop_mask = torch.rand(B, device=device) < p_uncond
                    labels = torch.where(drop_mask, torch.full_like(labels, num_classes), labels)
                
                # Predict Velocity and Endpoint Displacement concurrently
                v_pred, delta_x = model(x0, t, labels, return_delta_x=True)
                
                # Term 1: Instantaneous Velocity Loss
                loss_vel = F.mse_loss(v_pred, v_target)
                
                # Term 2: Analytical Endpoint Loss (integral from 0 to t)
                t_spatial = t.view(B, 1, 1, 1)
                x_target = (1 - t_spatial) * x0 + t_spatial * x1
                x_pred = x0 + delta_x
                loss_end = F.mse_loss(x_pred, x_target)
                
                # Total Objective
                total_loss = loss_vel + (endpoint_weight * loss_end)
                
                # Normalize loss for accumulation across chunks
                loss = total_loss / num_chunks
                loss.backward()
                
                step_loss += total_loss.item()
                step_loss_vel += loss_vel.item()
                step_loss_end += loss_end.item()
                
            # --- GRADIENT STEP ---
            optimizer.step()
            optimizer.zero_grad()
            ema.update(base_model)
            
            # Average chunk losses for logging
            avg_step_loss = step_loss / num_chunks
            avg_step_loss_vel = step_loss_vel / num_chunks
            avg_step_loss_end = step_loss_end / num_chunks
            
            # Add to epoch totals (scaling back up by num_chunks matches previous len(dataloader) average math)
            epoch_loss += step_loss
            epoch_loss_vel += step_loss_vel
            epoch_loss_end += step_loss_end
            
            pbar.set_postfix({
                'L_vel': avg_step_loss_vel, 
                'L_end': avg_step_loss_end, 
                'Total': avg_step_loss
            })
            
            # Reset accumulation buffers
            accum_x0, accum_x1, accum_labels = [], [], []
            
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
                
                # Save EMA-only checkpoint (for standalone inference)
                ema_ckpt_path = os.path.join(out_dir, f"ema_model_epoch_{epoch+1}.pt")
                torch.save(base_model.state_dict(), ema_ckpt_path)
                
            base_model.train()
            ema.restore(base_model)
            
            # Save full training checkpoint AFTER restoring training weights
            # (EMA shadow is preserved separately via ema.state_dict())
            full_ckpt = {
                'epoch': epoch + 1,
                'model_state_dict': base_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ema_state_dict': ema.state_dict(),
            }
            if use_scheduler:
                full_ckpt['scheduler_state_dict'] = scheduler.state_dict()
            full_ckpt_path = os.path.join(out_dir, f"training_checkpoint_epoch_{epoch+1}.ckpt")
            torch.save(full_ckpt, full_ckpt_path)
            
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
