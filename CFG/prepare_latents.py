import os
import argparse
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from utils import load_config


def main():
    parser = argparse.ArgumentParser(description="Prepare static optimal transport pairings.")
    parser.add_argument("--config", type=str, required=True, help="Path to dataset-specific config (e.g., configs/cifar10.yaml)")
    parser.add_argument("--global_config", type=str, default="configs/global.yaml", help="Path to global config")
    args = parser.parse_args()

    config = load_config(args.global_config, args.config)
    
    dataset_type = config['experiment'].get('dataset_type', 'cifar10')
    data_dir = config['experiment']['data_dir']
    out_dir = config['experiment']['output_dir']
    noise_multiplier = config.get('data_prep', {}).get('noise_multiplier', 4)
    
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Loading {dataset_type} dataset from {data_dir}...")
    
    if dataset_type == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=transform)
        
        # Load all into memory
        loader = torch.utils.data.DataLoader(dataset, batch_size=500, shuffle=False)
        all_data, all_labels = [], []
        for x, y in tqdm(loader, desc="Loading CIFAR-10 into RAM"):
            all_data.append(x)
            all_labels.append(y)
            
        Z_data = torch.cat(all_data, dim=0)
        labels = torch.cat(all_labels, dim=0)
        
    elif dataset_type == 'imagenet_latents':
        class_folders = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        class_to_idx = {cls_name: i for i, cls_name in enumerate(class_folders)}
        
        all_data, all_labels = [], []
        for cls_name in tqdm(class_folders, desc="Loading ImageNet latents"):
            cls_dir = os.path.join(data_dir, cls_name)
            for file_name in os.listdir(cls_dir):
                if file_name.endswith('.npy'):
                    file_path = os.path.join(cls_dir, file_name)
                    arr = np.load(file_path)
                    
                    if arr.shape == (32, 32, 4):
                        arr = arr.transpose(2, 0, 1) # Convert to [4, 32, 32]
                    elif arr.shape == (1, 4, 32, 32):
                        arr = arr.squeeze(0) # Convert to [4, 32, 32]
                        
                    tensor = torch.from_numpy(arr).float()
                    all_data.append(tensor)
                    all_labels.append(class_to_idx[cls_name])
                    
        Z_data = torch.stack(all_data, dim=0)
        labels = torch.tensor(all_labels, dtype=torch.long)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    N, C, H, W = Z_data.shape
    M = N * noise_multiplier
    print(f"Loaded dataset: N={N}, shape=({C}, {H}, {W}). M={M} (multiplier={noise_multiplier})")
    
    print("Generating Z_noise...")
    Z_noise = torch.randn(M, C, H, W)
    
    print("Performing Vectorized Asymmetric 1D Sliced OT...")
    flat_data = Z_data.view(N, -1)
    flat_noise = Z_noise.view(M, -1)
    D = flat_data.shape[1]
    
    # Generate single random projection vector
    v = torch.randn(D, 1)
    v = v / torch.norm(v)
    
    # Project to 1D
    P_data = flat_data @ v  # [N, 1]
    P_noise = flat_noise @ v  # [M, 1]
    
    # Repeat data projections and indices
    data_idx = torch.arange(N)
    P_data_expanded = P_data.repeat_interleave(noise_multiplier)  # [M, 1]
    data_idx_expanded = data_idx.repeat_interleave(noise_multiplier)  # [M]
    
    # Sort
    print("Sorting projections...")
    sort_indices_noise = torch.argsort(P_noise.squeeze())
    sort_indices_data = torch.argsort(P_data_expanded.squeeze())
    
    # Map
    print("Building pairing map...")
    pairing_map = torch.zeros(M, dtype=torch.long)
    pairing_map[sort_indices_noise] = data_idx_expanded[sort_indices_data]
    
    # Save outputs
    print(f"Saving to {out_dir}...")
    torch.save(Z_data, os.path.join(out_dir, "latents.pt"))
    torch.save(Z_noise, os.path.join(out_dir, "noise.pt"))
    torch.save(pairing_map, os.path.join(out_dir, "pairing_map.pt"))
    torch.save(labels, os.path.join(out_dir, "labels.pt"))
    
    print("Done! Data preparation successful.")

if __name__ == '__main__':
    main()
