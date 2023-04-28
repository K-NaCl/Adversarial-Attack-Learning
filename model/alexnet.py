import torch
from torch import nn
from .base import BaseModel

class AlexNet(BaseModel):
    def __init__(self):
        super().__init__()    
        self.name = 'AlexNet'
        