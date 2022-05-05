from typing import Tuple
import torch
from torchvision import transforms, datasets

from dataset.dataset import Dataset


class Cifar10(Dataset):
    def __call__(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        train_dataset = datasets.CIFAR10(self.download_path, download=True, train=True, transform=train_transform, target_transform=torch.tensor)
        test_dataset = datasets.CIFAR10(self.download_path, download=True, train=False, transform=test_transform, target_transform=torch.tensor)
        return train_dataset, test_dataset


class Cifar100(Dataset):
    def __call__(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        train_dataset = datasets.CIFAR100(self.download_path, download=True, train=True, transform=train_transform, target_transform=torch.tensor)
        test_dataset = datasets.CIFAR100(self.download_path, download=True, train=False, transform=test_transform, target_transform=torch.tensor)
        return train_dataset, test_dataset