import torch
from torch import nn
import timm

import os
import logging
import time
import tqdm

from .basic import *
import model as mymodel
    
model_dict = {
    'alexnet': mymodel.AlexNet,
    'resnet': None
}

class Trainer:    
    def __init__(
        self,
        model_name: str = 'alexnet',
        dataset: str = 'fashion-mnist',
        in_ch: int = 1,
        out_ch: int = 10,
        batch_size: int = 256,
        lr: float = 0.0001,
        seed: int = 0,
        cuda: int = 0,
        **kwargs,
    ) -> None:
        self.seed = seed
        set_random_seed(seed)


        self.model = model_dict[model_name](in_ch, out_ch)

        self.dataset = dataset

        self.model_name = f'{model_name}-bs{batch_size}-lr{lr}-seed{seed}'
        self.mkdir()

        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.train_iter, self.test_iter = data_iter(dataset, batch_size, seed)
        self.device = torch.device(f'cuda:{cuda}' if torch.cuda.is_available() else 'cpu')
        
        logging.basicConfig(
            filename = self.log_path + self.model_name + '.log', 
            level = logging.INFO
        )

    def train(self, epochs: int = 100):
        set_random_seed(self.seed)
        mertics = {
            'train_loss': [],
            'test_loss': [],
            'train_acc': [],
            'test_acc': []
        }
        max_acc, best_epoch = 0.0, 0

        logging.info(f'train on {self.device}')
        self.model.to(self.device)

        for epoch in range(epochs):
            time_start = time.time()
            train_loss, train_acc = self.train_epoch(epoch, epochs)
            time_end = time.time()

    def train_epoch(self, epoch, epochs):
        self.model.train()
        accu = Accumulator(3)
        with tqdm(enumerate(self.train_iter), total = len(self.train_iter), leave = True) as t:
            for idx, (X, Y) in t:
                self.opt.zero_grad()
                X, Y = X.to(self.device), Y.to(self.device)
                Y_hat = torch.tensor(self.model(X))
                loss = torch.tensor(self.loss_fn(Y_hat,Y))
                loss.backward()
                self.opt.step()

                with torch.no_grad():
                    correct_num = get_correct_num(Y, Y_hat)

                accu.add(loss.item() * len(Y), correct_num, len(Y))

                t.set_description(f'Epoch: [{epoch}/{epochs}]')
                t.set_postfix({
                    'batch': f'{idx} / {len(self.train_iter)}',
                    'training loss': f'{accu[0] / accu[-1]:.2e}',
                    'training acc': f'{(accu[1] / accu[-1]) * 100:4.2f}%'
                })
        return accu[0] / accu[-1], accu[1] / accu[-1]

    def test(self, epochs: int = 100):
        pass



    def mkdir(self):
        self.results_path = f'./results/{self.dataset}/{self.model_name}'
        self.model_path = f'{self.results_path}/models/'
        self.metric_path = f'{self.results_path}/metrics/'
        self.log_path = f'{self.results_path}/logs/'

        paths = [self.model_path, self.metric_path, self.log_path]
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)

