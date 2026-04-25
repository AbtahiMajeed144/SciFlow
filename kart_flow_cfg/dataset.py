import torch
import torchvision
import torchvision.transforms as transforms

class StaticOTCIFAR10(torch.utils.data.Dataset):
    def __init__(self, root='./data', seed=42):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Scale to [-1, 1]
        ])
        
        # Use ImageFolder for reading raw .png/.jpg images
        self.cifar10 = torchvision.datasets.ImageFolder(root=root, transform=transform)
        
        # Generate static Gaussian noise matching the dataset length
        torch.manual_seed(seed)
        self.fixed_noise = torch.randn(len(self.cifar10), 3, 32, 32)
        
    def __len__(self):
        return len(self.cifar10)
        
    def __getitem__(self, index):
        image, label = self.cifar10[index]
        noise = self.fixed_noise[index]
        return noise, image, label

def get_cifar10_dataloader(batch_size=128, root='./data'):
    dataset = StaticOTCIFAR10(root=root)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return dataloader
