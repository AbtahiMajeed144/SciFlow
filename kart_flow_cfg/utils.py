import os
import torch
import torchvision

def save_image_grid(tensor, filename, nrow=8):
    """
    Saves a tensor of shape [B, C, H, W] to an image file.
    Assumes tensor is in [-1, 1] range.
    """
    # Unnormalize to [0, 1]
    tensor = (tensor + 1) / 2
    tensor = torch.clamp(tensor, 0, 1)
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torchvision.utils.save_image(tensor, filename, nrow=nrow)
