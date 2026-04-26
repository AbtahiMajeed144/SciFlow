import os
import torch
from torch.utils.data import Dataset, DataLoader


class PairedOTDataset(Dataset):
    """
    Dataset for pre-computed Asymmetric 1D Sliced OT pairings.
    
    Loads three .pt files (latents, noise, pairing_map) plus labels into RAM.
    __len__ returns M (noise size), __getitem__ returns the paired (noise, data, label) tuple.
    """
    def __init__(self, data_dir):
        print(f"Loading paired dataset from {data_dir}...")
        self.latents = torch.load(os.path.join(data_dir, "latents.pt"), map_location='cpu')
        self.noise = torch.load(os.path.join(data_dir, "noise.pt"), map_location='cpu')
        self.pairing_map = torch.load(os.path.join(data_dir, "pairing_map.pt"), map_location='cpu')
        self.labels = torch.load(os.path.join(data_dir, "labels.pt"), map_location='cpu')
        
    def __len__(self):
        return len(self.noise)
        
    def __getitem__(self, idx):
        z_0 = self.noise[idx]
        target_idx = self.pairing_map[idx]
        x_1 = self.latents[target_idx]
        label = self.labels[target_idx]
        return z_0, x_1, label


def get_dataloader(data_dir, batch_size=128):
    dataset = PairedOTDataset(data_dir=data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return dataloader
