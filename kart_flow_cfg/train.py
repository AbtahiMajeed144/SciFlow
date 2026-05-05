import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import yaml
import argparse
import numpy as np
from tqdm import tqdm

from dataset import get_cifar10_dataloader
from model import KARTFlowModel
from inference import generate_1step


# ──────────────────────────────────────────────────────────────────────────────
# Adaptive L2 Loss (from MeanFlow, adapted for KART Flow)
# ──────────────────────────────────────────────────────────────────────────────
def adaptive_l2_loss(error, gamma=1.0, c=1e-3):
    """Adaptive L2 loss: sg(w) * ||Δ||² where w = 1 / (||Δ||² + c)^(1-γ).
    
    Downweights easy-to-predict smooth regions so that high-frequency
    details (edges, textures) receive proportionally more gradient signal.
    
    Args:
        error: (B, C, H, W) tensor of prediction errors (v_pred - v_target)
        gamma: Power parameter. 0.5 → effective loss ∝ ||Δ||¹ (balances L1/L2)
        c: Small constant for numerical stability
    Returns:
        Scalar loss
    """
    delta_sq = torch.mean(error ** 2, dim=(1, 2, 3))  # (B,)
    p = 1.0 - gamma
    w = 1.0 / (delta_sq + c).pow(p)                    # (B,)
    return (w.detach() * delta_sq).mean()


# ──────────────────────────────────────────────────────────────────────────────
# Logit-Normal Time Sampling
# ──────────────────────────────────────────────────────────────────────────────
def sample_logit_normal_t(batch_size, mu=-0.4, sigma=1.0, device='cpu'):
    """Sample timesteps from a logit-normal distribution.
    
    Concentrates samples in the mid-range of [0,1] where the velocity field
    is hardest to learn, avoiding wasted compute on trivial t≈0 and t≈1 regions.
    
    Args:
        batch_size: number of samples
        mu: mean of the underlying normal (negative → slight bias toward t=0, data side)
        sigma: std of the underlying normal
        device: target device
    Returns:
        t: (B, 1) tensor of timesteps in (0, 1)
    """
    normal_samples = np.random.randn(batch_size).astype(np.float32) * sigma + mu
    t_np = 1.0 / (1.0 + np.exp(-normal_samples))  # Sigmoid → (0, 1)
    t = torch.tensor(t_np, device=device).unsqueeze(1)  # (B, 1)
    return t

def train(args):
    # Load Config
    with open('kart_flow_cfg/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Hyperparams from config
    batch_size = config['training']['batch_size']
    grad_accum_steps = config['training'].get('gradient_accumulation_steps', 1)
    epochs = config['training']['epochs']
    lr = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']
    save_every = config['training']['save_every']
    out_dir = config['experiment']['output_dir']
    
    os.makedirs(out_dir, exist_ok=True)
    
    dataloader = get_cifar10_dataloader(batch_size=batch_size, root=config['experiment']['data_dir'])
    model = KARTFlowModel(config).to(device)
    
    # Multi-GPU Support via DataParallel
    if config['training']['multi_gpu'] and torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
        
    # Set explicit 0.0 weight decay for KART layer w and B parameters
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not (n.endswith('K.w') or n.endswith('K.B'))],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n.endswith('K.w') or n.endswith('K.B')],
            "weight_decay": 0.0,
        },
    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=lr)
    
    # Learning Rate Scheduler
    use_scheduler = config['training'].get('use_scheduler', False)
    if use_scheduler:
        import math
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
    
    start_epoch = 0
    if args.ckpt_path and os.path.isfile(args.ckpt_path):
        print(f"Resuming training from checkpoint: {args.ckpt_path}")
        checkpoint = torch.load(args.ckpt_path, map_location=device)
        
        base_model = model.module if isinstance(model, nn.DataParallel) else model
        base_model.load_state_dict(checkpoint['model_state_dict'])
        
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if use_scheduler and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        start_epoch = checkpoint.get('epoch', 0)
        print(f"Resumed at epoch {start_epoch}")

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad() # Initialize gradients outside the loop
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for i, (x1, labels) in enumerate(pbar):
            x1 = x1.to(device)
            labels = labels.to(device)
            B = x1.size(0)
            
            # Dynamic noise: fresh Gaussian noise every iteration
            x0 = torch.randn_like(x1)
            
            # CFG Dropout
            p_uncond = config.get('cfg', {}).get('p_uncond', 0.1)
            num_classes = config.get('cfg', {}).get('num_classes', 10)
            drop_mask = torch.rand(B, device=device) < p_uncond
            labels[drop_mask] = num_classes
            
            # Target Velocity (Dynamic Pairing)
            v_target = x1 - x0
            
            # Logit-normal time sampling (concentrates on hard timesteps)
            t = sample_logit_normal_t(B, mu=-0.4, sigma=1.0, device=device)
            
            # Predict Velocity using initial noise x0 and labels
            v_pred = model(x0, t, labels)
            
            # Adaptive L2 Loss (downweights easy regions, emphasizes high-freq details)
            error = v_pred - v_target
            loss = adaptive_l2_loss(error)
            loss = loss / grad_accum_steps  # Normalize loss for accumulation
            
            loss.backward()
            
            if ((i + 1) % grad_accum_steps == 0) or (i + 1 == len(dataloader)):
                optimizer.step()
                optimizer.zero_grad()
            
            # Un-normalize loss for logging
            total_loss += loss.item() * grad_accum_steps
            pbar.set_postfix({'loss': loss.item() * grad_accum_steps})
            
        avg_loss = total_loss / len(dataloader)
        
        if use_scheduler:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = lr
            
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")
        
        # 1-Step Analytical Inference
        if (epoch + 1) % save_every == 0 or epoch == 0:
            base_model = model.module if isinstance(model, nn.DataParallel) else model
            base_model.eval()
            with torch.no_grad():
                filename = os.path.join(out_dir, f"epoch_{epoch+1}.png")
                num_samples = config['inference']['num_samples']
                num_classes = config.get('cfg', {}).get('num_classes', 10)
                guidance_scale = config.get('cfg', {}).get('guidance_scale', 3.0)
                eval_labels = torch.randint(0, num_classes, (num_samples,), device=device)
                generate_1step(base_model, device, num_samples=num_samples, filename=filename, 
                               labels=eval_labels, guidance_scale=guidance_scale, num_classes=num_classes)

            # Save checkpoint with metadata
            if (epoch + 1) % save_every == 0:
                checkpoint_path = os.path.join(out_dir, f"checkpoint_epoch_{epoch+1}.ckpt")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': base_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if use_scheduler else None,
                    'learning_rate': current_lr,
                    'config': config
                }, checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train KART Flow Model")
    parser.add_argument('--ckpt_path', type=str, default=None, help="Path to checkpoint to resume training from")
    args = parser.parse_args()
    train(args)
