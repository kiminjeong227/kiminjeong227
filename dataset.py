import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms






def load_dataset(dataset='mnist', download=True, data_dir='./dataset'):
    
    assert dataset in {'mnist', 'fashion_mnist', 'svhn', 'cifar10', 'cifar100'}
    
    if dataset == 'mnist':
        train_transform = transforms.Compose([transforms.ToTensor()])
        valid_transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.MNIST(download=download, root=data_dir, train=True, transform=train_transform)
        valid_dataset = datasets.MNIST(download=download, root=data_dir, train=False, transform=valid_transform)
        class_to_idx = train_dataset.class_to_idx

    elif dataset == 'fashion_mnist':
        train_transform = transforms.Compose([transforms.ToTensor()])
        valid_transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.FashionMNIST(download=download, root=data_dir, train=True, transform=train_transform)
        valid_dataset = datasets.FashionMNIST(download=download, root=data_dir, train=False, transform=valid_transform)
        class_to_idx = train_dataset.class_to_idx

    elif dataset == 'svhn':
        train_transform = transforms.Compose([transforms.ToTensor()])
        valid_transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.SVHN(root=data_dir, split='train', download=download, transform=train_transform)
        valid_dataset = datasets.SVHN(root=data_dir, split='test', download=download, transform=train_transform)
        class_to_idx = {str(number): number for number in range(10)}

    elif dataset == 'cifar10':
        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
        valid_transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=download, transform=train_transform)
        valid_dataset = datasets.CIFAR10(root=data_dir, train=False, download=download, transform=train_transform)
        class_to_idx = train_dataset.class_to_idx

    elif dataset == 'cifar100':
        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
        valid_transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=download, transform=train_transform)
        valid_dataset = datasets.CIFAR10(root=data_dir, train=False, download=download, transform=train_transform)
        class_to_idx = train_dataset.class_to_idx

    else:
        raise NotImplementedError

    return train_dataset, valid_dataset, class_to_idx

