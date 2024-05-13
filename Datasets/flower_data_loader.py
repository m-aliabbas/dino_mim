import torchvision
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader,ConcatDataset

def load_flowers_data(transform=None,split='all'):

    if split in ['train','val','test']:
        dataset = datasets.Flowers102(root='data/', split=split, download=True, transform=transform)
    elif split == 'all':
        train_dataset = datasets.Flowers102(root='data/', split='train', download=True, transform=transform)
        test_dataset = datasets.Flowers102(root='data/', split='test', download=True, transform=transform)
        val_dataset = datasets.Flowers102(root='data/', split='val', download=True, transform=transform)
        # Combine the datasets
        dataset = ConcatDataset([train_dataset, test_dataset, val_dataset])
    else:
        raise('You can only pass train / val / test / all (for single combine dataset).')
        dataset = None
    return dataset