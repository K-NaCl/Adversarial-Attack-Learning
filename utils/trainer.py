import torch
from torch import nn
import timm

import os
import logging
import shutil
import time
import pandas as pd
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from .basic import *
import model as mymodel
    

class Trainer:    
    def __init__(
        self,
        model: mymodel.BaseModel,
        dataset: str = 'fashion-mnist',
        batch_size: int = 256,
        lr: float = 0.1,
        seed: int = 0,
        use_cuda: bool = True,
        use_lr_sche: bool = True,
        use_tb: bool = False,
        **kwargs,
    ) -> None:
        self.seed = seed
        set_random_seed(seed)

        self.model = model

        self.dataset = dataset

        self.model_name_param = f'{model.name}-bs{batch_size}-lr{lr}-seed{seed}'
        self.mkdir()

        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        if use_lr_sche:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.opt, 'max', factor = 0.5, patience = 2
            )

        self.train_iter, self.test_iter = data_iter(dataset, batch_size, seed)

        self.device = torch.device(f'cuda:0' if use_cuda and torch.cuda.is_available() else 'cpu')
        
        self.use_tb = use_tb

    def train(self, epochs: int = 100):
        self.check_path()
        set_random_seed(self.seed)

        if self.use_tb:
            self.tb_writer = SummaryWriter(self.tb_path)            

        logging.basicConfig(
            filename = self.log_path + self.model_name_param + '.log', 
            format='%(asctime)s: %(message)s',
            datefmt='%m-%d %H:%M:%S',
            level = logging.INFO
        )

        metrics = {
            'train_loss': [],
            'test_loss': [],
            'train_acc': [],
            'test_acc': []
        }        
        
        max_acc, best_epoch = 0.0, 0

        print(f'train on {self.device}')
        logging.info(f'train on {self.device}')
        self.model.to(self.device)

        for epoch in range(epochs):
            time_start = time.time()

            train_loss, train_acc = self.train_epoch(epoch, epochs)
            test_loss, test_acc = self.test_epoch()

            if hasattr(self, 'scheduler'):
                self.scheduler.step(test_acc)

            time_end = time.time()

            if test_acc > max_acc:
                max_acc = test_acc
                best_epoch = epoch
                print(f'best epoch: {best_epoch}, max acc={max_acc * 100:4.2f}%')
                torch.save(
                    self.model.state_dict(),
                    self.model_path + self.model_name_param
                )

            for metric in list(metrics.keys()):
                metrics[metric].append(eval(metric))

            if self.use_tb:
                self.tb_writer.add_scalars(
                    'epoch',
                    {
                        'train_loss': train_loss,
                        'test_loss': test_loss,
                        'train_acc': train_acc,
                        'test_acc': test_acc
                    },
                    epoch
                )

            if (epoch + 1) % 10 == 0:
                pd.DataFrame(metrics).to_csv(
                    self.metric_path + self.model_name_param + '.csv'
                )

            train_info = f'train loss: {train_loss:.2e},  train acc: {train_acc * 100:4.2f}%'
            test_info = f'test loss: {test_loss:.2e},  test acc: {test_acc * 100:4.2f}%'
            other_info = f'time: {time_end-time_start:.2f},  best epoch: {best_epoch + 1}'
            info = f'epoch: {epoch + 1:3},  {train_info},  {test_info},  {other_info}'
            logging.info(info)            

    def train_epoch(self, epoch, epochs):
        self.model.train()
        accu = Accumulator(3)
        with tqdm(enumerate(self.train_iter), total = len(self.train_iter), leave = True) as t:
            for idx, (X, Y) in t:
                self.opt.zero_grad()
                X, Y = X.to(self.device), Y.to(self.device)
                Y_hat: torch.Tensor = self.model(X)
                loss: torch.Tensor = self.loss_fn(Y_hat,Y)

                loss.backward()
                self.opt.step()

                with torch.no_grad():
                    correct_num = get_correct_num(Y, Y_hat)

                accu.add(loss.item() * len(Y), correct_num, len(Y))

                t.set_description(f'Epoch: [{epoch + 1}/{epochs}]')
                t.set_postfix({
                    'batch': f'{idx + 1} / {len(self.train_iter)}',
                    'training loss': f'{accu[0] / accu[-1]:.2e}',
                    'training acc': f'{(accu[1] / accu[-1]) * 100:4.2f}%'
                })
        return accu[0] / accu[-1], accu[1] / accu[-1]

    def test_epoch(self):
        self.model.eval()
        accu = Accumulator(3)
        with torch.no_grad():
            for X, Y in self.test_iter:
                X, Y = X.to(self.device), Y.to(self.device)
                Y_hat: torch.Tensor = self.model(X)
                loss: torch.Tensor = self.loss_fn(Y_hat, Y)
                correct_num = get_correct_num(Y, Y_hat)
                accu.add(loss.item() * len(Y), correct_num, len(Y))
        return accu[0] / accu[-1], accu[1] / accu[-1]

    def mkdir(self):
        self.results_path = f'./results/{self.dataset}/{self.model_name_param}'
        self.model_path = f'{self.results_path}/models/'
        self.metric_path = f'{self.results_path}/metrics/'
        self.log_path = f'{self.results_path}/logs/'
        self.tb_path = f'{self.results_path}/tb_out/'
        
        paths = [self.model_path, self.metric_path, self.log_path, self.tb_path]
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)

    def load_model(self):        
        load_path = self.model_path + self.model_name_param
        if not os.path.exists(load_path):
            raise ValueError(f'模型路径{load_path}不存在！')
        
        self.model.load_state_dict(torch.load(load_path, map_location = 'cpu'))
        self.model.eval()

    def check_path(self):
        exist_file = False
        for dirpath, __, filenames in os.walk(self.results_path):
            if len(filenames) != 0:
                print(f'文件夹{dirpath.replace(self.results_path, "")}下存在文件{filenames}')
                exist_file = True

        if exist_file:
            print(f'模型{self.model_name_param}文件夹下已存在数据，继续训练将覆盖原有数据，是否继续训练？y/[n]')
            confirm = input().lower().strip()
            if (confirm not in 'yes') or (len(confirm) == 0):
                exit()
            else:
                if os.path.exists(self.tb_path):
                    shutil.rmtree(self.tb_path)

