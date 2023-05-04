import torch
from torch import nn
from torch.nn import functional as F

import model as mymodel

import numpy as np

class BaseAttack:
    def __init__(
            self,
            model: mymodel.BaseModel,
            use_cuda: bool = True
    ) -> None:
        self.model = model
        if use_cuda and torch.cuda.is_available():            
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.model.to(self.device)
        print(f'attack {model.name} on {self.device}')

    def __call__(
            self,
            imgs_in: torch.Tensor, 
            labels: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            if len(imgs_in) != len(labels):
                raise ValueError(f'输入图像数量({len(imgs_in)})与标签数量({len(labels)})不一致！')
