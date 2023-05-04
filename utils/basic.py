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
from tqdm import tqdm

class Accumulator:
    def __init__(self, n: int):
        '''        
        累加器类，方便累加一些数据，如正确数、损失等
        
        Args:
        - `n`: 累加器维度
        '''
        self.data = [0.0] * n


    def add(self, *args):
        '''
        累加函数

        Args:
        - `*args`: 待累加数据组，与累加器维度对应
        '''
        self.data = [a + float(b) for a, b in zip(self.data, args)]


    def reset(self):
        '''
        重置函数，将保存数据重置为0
        '''
        self.data = [0.0] * len(self.data)


    def __getitem__(self, idx):
        return self.data[idx]



def set_random_seed(seed: int):
    '''
    设置随机种子，用于复现结果

    Args:
    - `seed`: 随机种子
    '''
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)



def get_correct_num(Y: torch.Tensor, Y_hat: torch.Tensor):
    '''
    获取批量中预测正确个数

    Args:
    - `Y`: 正确标签
    - `Y_hat`: 预测标签

    Retern:
    - `correct_num`: 预测正确个数(float)
    '''
    with torch.no_grad():
        if Y_hat.dim() > 1 and Y_hat.shape[1] > 1:
            Y_hat = F.softmax(Y_hat, dim = 1)
            Y_hat = Y_hat.argmax(dim = 1)
        cmp = Y_hat.type(Y.dtype) == Y
        return float(cmp.type(Y.dtype).sum())
    


def data_iter(dataset: str, batch_size: int = 256, seed: int = 0):
    '''
    获取数据集迭代器
    
    Args:
    - `dataset`: 数据集名称，可选["CIFAR*" "imagenette" "fashion-mnist"]中的一种
    - `batch_size`: 批量大小，影响显存占用率
    - `seed`: 随机种子，用于复现结果，影响洗牌顺序

    Returns:
    - `train_iter`: 训练集(DataLoader)
    - `test_iter`: 测试集(DataLoader)
    '''
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



def creat_model(model_name: str, model_path: str = '', in_ch: int = 1, out_ch: int = 10, pretrained: bool = True) -> mymodel.BaseModel:
    '''
    创建新模型或从路径加载模型

    Args:
    - `model_name`: 模型名称，字符串须包含["alexnet" "moilenet"]中的一种
    - `model_path`: 模型路径，为空则创建新模型，模型类型须与名称对应
    - `in_ch`: 输入通道数，须与数据集图片通道数对应
    - `out_ch`: 输出通道数，须与数据集图片种类对应
    - `pretrained`: 是否为预训练模型

    Return:
    - `model`: 输出模型(BaseModel)
    '''
    model: mymodel.BaseModel
    
    if not pretrained:
        if 'alexnet' in model_name.lower():
            model = mymodel.AlexNet(in_ch, out_ch)
    else:
        if 'mobilenet' in model_name.lower():            
            model = timm.create_model('mobilenetv3_small_050', pretrained = True, num_classes = out_ch)
            model.conv_stem = nn.Conv2d(in_ch, 16, kernel_size = 3, stride = 1, padding = 2, bias = False)
        elif 'resnet' in model_name.lower():
            model = timm.create_model('resnet18', pretrained=True, num_classes = out_ch)
            pass

    if model_path != '':
        if os.path.exists(model_path):            
            model.load_state_dict(torch.load(model_path, map_location = 'cpu'))
            model.is_load = True
        else:
            raise ValueError(f'模型路径{model_path}不存在！')

    model.name = model_name

    return model



@torch.no_grad()
def test_accuracy(model: mymodel.BaseModel, data_iter) -> float:
    '''
    在指定数据集上测试模型准确率

    Args:
    - `model`: 待测试模型
    - `data_iter`: 一组数据，可使用`data_iter()`生成

    Return:
    - 模型准确率(float)
    '''
    print(f'test on {model.name}')
    
    if not model.is_load:
        raise(f'模型{model.name}未训练')
    accu = Accumulator(2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    with torch.no_grad():
        with tqdm(enumerate(data_iter), total = len(data_iter), leave = True) as t:
            for idx, (X, Y) in t:
                X, Y = X.to(device), Y.to(device)
                Y_hat: torch.Tensor = model(X)
                accu.add(get_correct_num(Y, Y_hat), len(Y))
    model.cpu()
    return accu[0] / accu[1]



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