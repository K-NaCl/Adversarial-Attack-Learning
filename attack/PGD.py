import torch
from torch import nn

from model import BaseModel
from .base import BaseAttack

from tqdm import tqdm

class PGD(BaseAttack):
    def __init__(self, model: BaseModel, use_cuda: bool = True) -> None:
        super().__init__(model, use_cuda)

    def __call__(
        self, 
        imgs_in: torch.Tensor, 
        labels: torch.Tensor,
        eps: float = 0.1,
        max_iter: int = 5
    ) -> torch.Tensor:
        super().__call__(imgs_in, labels)
        
        loss_fn = nn.CrossEntropyLoss()
        imgs = imgs_in.clone().detach().to(self.device)
        imgs_ori = imgs_in.clone().detach().to(self.device)

        imgs_out = torch.zeros_like(imgs).to(self.device)
        deltas = torch.zeros_like(imgs)

        for n in range(max_iter):
            if n != 0:
                imgs = imgs_out.clone().detach()
            imgs.requires_grad_(True)

            self.model.zero_grad()
            Y = labels.clone().to(self.device)
            Y_hat: torch.Tensor = self.model(imgs)
            loss: torch.Tensor = loss_fn(Y_hat, Y)
            loss.backward()
            
            grads = imgs.grad

            with torch.no_grad():
                deltas = deltas + eps / max_iter * grads.sign()
                imgs_out = torch.clamp(imgs_ori.data + deltas, 0, 1)

        return imgs_out.cpu().detach()