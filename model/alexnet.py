import torch
from torch import nn

class AlexNet(nn.Module):    
    def __init__(
        self,
        in_ch: int,
        out_ch: int
    ) -> None:                
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_ch, 32, kernel_size=11, stride=4, padding=1
        )
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size = 2, padding = 1)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 3, padding = 1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size = 2, padding = 1)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size = 3, padding = 1)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(128, 128, kernel_size = 3, padding = 1)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(128, 128, kernel_size = 3, padding = 1)
        self.relu5 = nn.ReLU()

        self.maxpool3 = nn.MaxPool2d(kernel_size = 2, padding = 1)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128*4, out_ch)

        self.register_params()
        
    def register_params(self):
        def initialize(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        self.apply(initialize)
    
    def forward(self, X):
        for module in self.children():
            X = module(X)
        return X
    
    def name() -> str:
        return 'AlexNet'