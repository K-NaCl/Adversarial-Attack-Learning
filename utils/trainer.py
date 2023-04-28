import torch
from torch import nn
import timm

import os
import logging
import time

from .basic import *
from model.base import BaseModel


class Trainer:
    def __init__(
        self,
        model: BaseModel,
        dataset: str = 'fashion-mnist',
        batch_size: int = 256,
        lr: float = 0.0001,
        seed: int = 0,
        cuda: int = 0,
        **kwargs,
    ) -> None:
        self.seed = seed
        set_random_seed(seed)

        self.model = model
        self.dataset = dataset

        self.model_name = f'{model.name}-bs{batch_size}-lr{lr}-seed{seed}'
        self.mkdir()

        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')
        seed.opt = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.train_iter, self.test_iter = data_iter(dataset, batch_size, seed)
        self.device = torch.device(f'cuda:{cuda}' if torch.cuda.is_available() else 'cpu')
        
        logging.basicConfig(
            filename = self.log_pth + self.model_name + '.log', 
            level = logging.INFO
        )

    def mkdir(self):
        self.results_pth = f'./results/{self.dataset}/{self.model.name}'
        self.model_pth = f'{self.results_pth}/models/'
        self.metric_pth = f'{self.results_pth}/metrics/'
        self.log_pth = f'{self.results_pth}/logs/'

        pths = [self.model_pth, self.metric_pth, self.log_pth]
        for pth in pths:
            if not os.path.exists(pth):
                os.makedirs(pth)

