import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import datasets, transforms

import numpy as np
import os

class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def set_random_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def get_correct_num(Y: torch.Tensor, Y_hat: torch.Tensor):
    with torch.no_grad():
        if Y_hat.dim() > 1 and Y_hat.shape[1] > 1:
            Y_hat = F.softmax(Y_hat, dim = 1)
            Y_hat = Y_hat.argmax(dim = 1)
        cmp = Y_hat.type(Y.dtype) == Y
        return float(cmp.type(Y.dtype).sum())

def data_iter(dataset: str, batch_size: int = 256, seed: int = 0):
    set_random_seed(seed)
    data_pth = f'./data/{dataset}'
    if not os.path.exists(data_pth):
        os.makedirs(data_pth)

    if 'CIFAR' in dataset:
        transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.AutoAugment(),
                transforms.ToTensor(),  # numpy -> Tensor
            ]
        )
        train_set = eval(dataset)(
            root=data_pth,
            train=True,
            download=True,
            transform=transform
        )
        test_set = eval(dataset)(
            root=data_pth,
            train=False,
            download=True,
            transform=transform
        )

    elif dataset == 'imagenette':
        transform = transforms.Compose(
            [
                transforms.CenterCrop(160),
                transforms.Resize(224),
                transforms.AutoAugment(),
                transforms.ToTensor(),  # numpy -> Tensor
            ]
        )
        train_set = datasets.ImageFolder(
            './data/imagenette2-160/train/',
            transform=transform
        )
        test_set = datasets.ImageFolder(
            './data/imagenette2-160/val/',
            transform=transform
        )

    elif dataset == 'fashion-mnist':
        transform = transforms.Compose(
            [
                transforms.RandomCrop(28, padding=2),#数据增强
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        train_set = datasets.FashionMNIST(
            root=data_pth,
            train=True,
            download=True,
            transform=transform
        )
        test_set = datasets.FashionMNIST(
            root=data_pth,
            train=False,
            download=True,
            transform=transform
        )
    else:
        raise ValueError(f'找不到数据集:{dataset}')

    train_iter = data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True
    )
    test_iter = data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False
    )
    return train_iter, test_iter
