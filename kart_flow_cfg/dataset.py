import torch
import torchvision
import torchvision.transforms as transforms

class CIFAR10Dataset(torch.utils.data.Dataset):
    """CIFAR-10 dataset with augmentation. Noise is generated dynamically in the training loop."""
    def __init__(self, root='./data'):
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Scale to [-1, 1]
        ])
        
        # Use ImageFolder for reading raw .png/.jpg images
        self.cifar10 = torchvision.datasets.ImageFolder(root=root, transform=transform)
        
    def __len__(self):
        return len(self.cifar10)
        
    def __getitem__(self, index):
        image, label = self.cifar10[index]
        return image, label

def get_cifar10_dataloader(batch_size=128, root='./data'):
    dataset = CIFAR10Dataset(root=root)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4
    )
    return dataloader
