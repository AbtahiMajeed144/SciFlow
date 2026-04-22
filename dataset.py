import torch
import torchvision
import torchvision.transforms as transforms
from scipy.optimize import linear_sum_assignment

def get_cifar10_dataloader(batch_size=128, root='./data'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Scale to [-1, 1]
    ])
    
    # Use ImageFolder for reading raw .png/.jpg images organized in class subfolders
    dataset = torchvision.datasets.ImageFolder(root=root, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return dataloader

def minibatch_ot_pairing(x1):
    """
    Optimized Minibatch OT using GPU for distance calculation.
    """
    B = x1.size(0)
    x0 = torch.randn_like(x1)
    
    # Flatten keeping them on the GPU
    x0_flat = x0.view(B, -1)
    x1_flat = x1.view(B, -1)
    
    # 1. Compute squared Euclidean distance matrix natively on GPU
    # cdist computes ||x0 - x1||_2, so we square it
    cost_matrix_tensor = torch.cdist(x0_flat, x1_flat, p=2).pow(2)
    
    # 2. Move ONLY the 128x128 distance matrix to CPU for SciPy
    cost_matrix = cost_matrix_tensor.cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    x1_paired = x1[col_ind]
    return x0, x1_paired
