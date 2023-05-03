import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import datasets, transforms
import timm

import model as mymodel

import numpy as np
import matplotlib.pyplot as plt
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
        dataset_name = 'datasets.' + dataset
        transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.AutoAugment(),
                transforms.ToTensor(),  # numpy -> Tensor
                transforms.Normalize(
                    mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2023, 0.1994, 0.2010)
                )
            ]
        )
        train_set = eval(dataset_name)(
            root=data_pth,
            train=True,
            download=True,
            transform=transform
        )
        test_set = eval(dataset_name)(
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

def creat_model(model_name: str, in_ch: int = 1, out_ch: int = 10, pretrained: bool = True) -> mymodel.BaseModel:
    model: mymodel.BaseModel
    if not pretrained:
        if 'alexnet' in model_name.lower():
            model = mymodel.AlexNet(in_ch, out_ch)
    else:
        if 'mobilenet' in model_name.lower():            
            model = timm.create_model('mobilenetv3_small_050', pretrained = True, num_classes = out_ch)
            model.conv_stem = nn.Conv2d(in_ch, 16, kernel_size = 3, stride = 1, padding = 2, bias = False)
        else:
            pass
    model.name = model_name
    return model

def test_accuracy(model: nn.Module, data_iter) -> float:
    '''
    在指定数据集上测试模型准确率

    Args:
    - `model`: 待测试模型
    - `data_iter`: 一组数据集，可使用`data_iter()`生成

    Return:
    - 模型准确率(float)
    '''
    accu = Accumulator(2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    with torch.no_grad():
        for X, Y in data_iter:
            X, Y = X.to(device), Y.to(device)
            Y_hat = model(X)
            accu.add(get_correct_num(Y, Y_hat), len(Y))
    model.cpu()
    return accu[0] / accu[1] * 100

def imshow(img: torch.Tensor):
    '''
    显示图片

    Args:
    - `img`: 待显示图片
    '''
    # img = img / 2 + 0.5 # reverse normalization
    np_img = img.numpy()  # tensor --> numpy
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.show()