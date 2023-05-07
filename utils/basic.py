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
import shutil
from tqdm import tqdm

class_names = {
    'Imagenette': (
        'Tench', 'English Springer', 'Cassette Player', 'Chain Saw', 'Church', 
        'French Horn', 'Garbage Truck', 'Gas Pump', 'Golf Ball', 'Parachute'
    ),
    'CIFAR10': (
        'Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse',
        'Ship', 'Trunk'
    ),
    'FashionMNIST': (
        'T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt',
        'Sneaker', 'Bag', 'Ankle boot'
    )
}

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
    


def data_iter(dataset: str,
              mode: str = 'train',
              batch_size: int = 256,
              seed: int = 0
    ):
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
        if mode == 'test':
            transform = transforms.Compose(
                [                    
                    transforms.Resize(224),
                    transforms.ToTensor()
                ]
            )
        elif mode == 'train':            
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
    
    if 'alexnet' in model_name.lower():
        model = mymodel.AlexNet(in_ch, out_ch)
    elif 'mobilenet' in model_name.lower():            
        model = timm.create_model('mobilenetv3_small_050', pretrained = pretrained, num_classes = out_ch)
        model.conv_stem = nn.Conv2d(in_ch, 16, kernel_size = 3, stride = 1, padding = 2, bias = False)
    elif 'resnet' in model_name.lower():
        model = timm.create_model('resnet18', pretrained = pretrained, num_classes = out_ch)
    else:
        raise ValueError(f'没有这种模型')
    
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
    在指定数据集上测试模型准确率和平均损失

    Args:
    - `model`: 待测试模型
    - `data_iter`: 一组数据，可使用`data_iter()`生成

    Return:
    - 平均损失(float)
    - 平均准确率(float)
    '''
    print(f'test on {model.name}')
    
    if not model.is_load:
        raise(f'模型{model.name}未训练')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    accu = Accumulator(3)
    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    
    with torch.no_grad():
        with tqdm(enumerate(data_iter), total = len(data_iter), leave = True) as t:
            for idx, (X, Y) in t:
                X, Y = X.to(device), Y.to(device)
                Y_hat: torch.Tensor = model(X)
                loss: torch.Tensor = loss_fn(Y_hat, Y)
                correct_num = get_correct_num(Y, Y_hat)
                accu.add(loss.item() * len(Y), correct_num, len(Y))
                
    model.cpu()
    return accu[0] / accu[-1], accu[1] / accu[-1]



def imshow(img: torch.Tensor):
    '''
    显示图片

    Args:
    - `img`: 待显示图片
    '''
    # img = img / 2 + 0.5 # reverse normalization
    if img.dim() == 4:
        np_img = img[0].numpy()
    else:
        np_img = img.numpy()  # tensor --> numpy
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.show()


@torch.no_grad()
def predict(
    model: mymodel.BaseModel,
    imgs_in: torch.Tensor,
    normalize_mode: str,
    get_Y_hat: bool = True,
    use_cuda: bool = True
):
    # print(f'predict {normalize_mode} on {model.name}')

    if normalize_mode is not None:
        mean = {
            'cifar10': (0.4914, 0.4822, 0.4465),
            'cifar100': (0.5071, 0.4867, 0.4408),
            'imagenette': (0.485, 0.456, 0.406),
        }
        std = {
            'cifar10': (0.2023, 0.1994, 0.2010),
            'cifar100': (0.2675, 0.2565, 0.2761),
            'imagenette': (0.229, 0.224, 0.225),
        }
        transform = transforms.Normalize(
            mean[normalize_mode.lower()], std[normalize_mode.lower()]
        )
        imgs = transform(imgs_in.clone())
    else:
        imgs = imgs_in.clone()

    device = torch.device(f'cuda' if use_cuda and torch.cuda.is_available() else 'cpu')

    model.to(device)

    Y_hat = model(imgs.to(device))

    probs = F.softmax(Y_hat, dim=1)

    values, indices = probs.max(dim=1)

    if get_Y_hat:
        return indices.cpu(), values.cpu(), Y_hat.cpu()
    else:
        return indices.cpu(), values.cpu()
    
def check_path(path: str):
    '''
    检查对应目录下是否存在数据

    如不存在日志文件以外的数据，则继续运行

    如存在日志文件以外的数据，则向用户询问，若确认继续训练，则覆盖原有数据。

    Args:
    - `path`: 文件夹路径
    '''
    if not os.path.isdir(path):
        print(f'不存在文件夹{path}')
        return
    
    exist_file = False
    for dirpath, __, filenames in os.walk(path):
        if len(filenames) != 0 and 'log' not in dirpath:
            print(f'文件夹{dirpath.replace(path, "")}下存在文件{filenames}')
            exist_file = True

    if exist_file:
        print(f'文件夹{path}下已存在数据，继续运行将覆盖原有数据，是否继续运行？y/[n]')
        confirm = input().lower().strip()
        if (confirm not in 'yes') or (len(confirm) == 0):
            exit()
        else:
            if os.path.exists(path + '/tb_out/'):
                shutil.rmtree(path + '/tb_out/')
