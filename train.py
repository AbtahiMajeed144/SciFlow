import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import yaml
from tqdm import tqdm

from dataset import get_cifar10_dataloader, minibatch_ot_pairing
from model import KARTFlowModel
from inference import generate_euler

def train():
    # Load Config
    with open('config.yaml', 'r') as f:
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
        
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Learning Rate Scheduler
    use_scheduler = config['training'].get('use_scheduler', False)
    if use_scheduler:
        import math
        warmup_epochs = 5
        min_lr = config['training'].get('min_lr', 1e-5)
        
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return float(epoch + 1) / float(warmup_epochs)
            progress = float(epoch - warmup_epochs) / float(max(1, epochs - warmup_epochs))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            lr_target = min_lr + (lr - min_lr) * cosine_decay
            return lr_target / lr
            
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad() # Initialize gradients outside the loop
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for i, (x1, _) in enumerate(pbar):
            x1 = x1.to(device)
            B = x1.size(0)
            
            # 1. Minibatch OT Pairing
            x0, x1_paired = minibatch_ot_pairing(x1)
            
            # 2. Target Velocity
            v_target = x1_paired - x0
            
            # 3. Sample random times
            t = torch.rand((B, 1), device=device)
            
            # Calculate intermediate state x_t
            t_expand = t.view(B, 1, 1, 1)
            x_t = (1 - t_expand) * x0 + t_expand * x1_paired
            
            # 4. Predict Velocity using x_t
            v_pred = model(x_t, t)
            
            # 5. Loss
            loss = F.mse_loss(v_pred, v_target)
            loss = loss / grad_accum_steps # Normalize loss for accumulation
            
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
        
        # Euler ODE Solver Inference
        if (epoch + 1) % save_every == 0 or epoch == 0:
            base_model = model.module if isinstance(model, nn.DataParallel) else model
            base_model.eval()
            with torch.no_grad():
                filename = os.path.join(out_dir, f"epoch_{epoch+1}.png")
                num_samples = config['inference']['num_samples']
                generate_euler(base_model, device, num_samples=num_samples, filename=filename)

if __name__ == '__main__':
    train()
