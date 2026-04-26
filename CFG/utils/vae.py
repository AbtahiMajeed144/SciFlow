import os
import torch
import torchvision
from diffusers import AutoencoderKL


def save_image_grid(tensor, filename, nrow=8):
    """
    Saves a tensor of shape [B, C, H, W] to an image file.
    Assumes tensor is in [-1, 1] range.
    """
    tensor = (tensor + 1) / 2
    tensor = torch.clamp(tensor, 0, 1)
    
    dirpath = os.path.dirname(filename)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    torchvision.utils.save_image(tensor, filename, nrow=nrow)


# Singleton VAE cache
_vae = None

def get_vae(device):
    """Load and cache the frozen SD VAE decoder (singleton)."""
    global _vae
    if _vae is None:
        weight_dtype = torch.float16 if device.type == 'cuda' else torch.float32
        _vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=weight_dtype).to(device)
        _vae.eval()
        for param in _vae.parameters():
            param.requires_grad = False
    return _vae


def generate_1step(model, device, num_samples=64, filename="outputs/sample.png",
                   labels=None, guidance_scale=3.0, num_classes=10):
    """
    Generate sample images using 1-step analytical integration.
    Used during training for periodic visualization.
    """
    model.eval()
    in_channels = model.in_channels
    img_size = model.img_size
    
    with torch.no_grad():
        x_0 = torch.randn(num_samples, in_channels, img_size, img_size, device=device)
        if labels is None:
            labels = torch.randint(0, num_classes, (num_samples,), device=device)
            
        # CFG-guided blueprint extrapolation + analytical integral
        h_guided = model.generate_with_cfg(x_0, labels, guidance_scale, num_classes)
        
        # Final 1-Step Jump
        x_final = x_0 + h_guided
        
        # VAE Decoding (if in latent space mode)
        if in_channels == 4:
            vae = get_vae(device)
            latent_pred = x_final / 0.18215
            weight_dtype = torch.float16 if device.type == 'cuda' else torch.float32
            latent_pred = latent_pred.to(weight_dtype)
            image_tensor = vae.decode(latent_pred).sample
            x_final = image_tensor.to(torch.float32)
            
        save_image_grid(x_final, filename)
        print(f"Saved {filename}")
