import torch
from torch import nn

from model import BaseModel
from .base import BaseAttack

from tqdm import tqdm

class NIFGSM(BaseAttack):
    def __init__(self, model: BaseModel, use_cuda: bool = True) -> None:
        super().__init__(model, use_cuda)
    
    def __call__(
        self, 
        imgs_in: torch.Tensor, 
        labels: torch.Tensor,
        eps: float = 0.1,
        mu: float = 0.1,
        max_iter: int = 5
    ) -> torch.Tensor:
        super().__call__(imgs_in, labels)
        
        loss_fn = nn.CrossEntropyLoss()        
        imgs = imgs_in.clone().detach().to(self.device)
        
        imgs_out = torch.zeros_like(imgs)
        g = torch.zeros_like(imgs).to(self.device)

        for n in range(max_iter):
            if n != 0:
                imgs = imgs_out.clone().detach()
            
            imgs_nes = imgs + eps / (n + 1) * mu * g
            imgs_nes.requires_grad_(True)
            
            self.model.zero_grad()
            Y = labels.clone().to(self.device)
            Y_hat: torch.Tensor = self.model(imgs_nes)
            loss: torch.Tensor = loss_fn(Y_hat, Y)
            loss.backward()
            
            grads = imgs_nes.grad
            for idx, grad in enumerate(grads):
                g[idx] = mu * g[idx] + grad / torch.linalg.norm(grad)

            delta = eps / max_iter * g.sign()
            imgs_out = torch.clamp(imgs + delta, 0, 1)
        
        return imgs_out.cpu().detach()