import torch
from torch import nn

from model import BaseModel
from .base import BaseAttack

from tqdm import tqdm

class FGSM(BaseAttack):
    def __init__(self, model: BaseModel, use_cuda: bool = True) -> None:
        super().__init__(model, use_cuda)

    def __call__(
            self, 
            imgs_in: torch.Tensor, 
            labels: torch.Tensor,
            eps: float = 0.1
    ) -> torch.Tensor:
        super().__call__(imgs_in, labels)
        
        self.model.zero_grad()
        loss_fn = nn.CrossEntropyLoss()
        imgs = imgs_in.clone().detach().to(self.device)

        imgs.requires_grad_(True)

        Y = labels.clone().to(self.device)
        Y_hat: torch.Tensor = self.model(imgs)
        loss: torch.Tensor = loss_fn(Y_hat, Y)
        loss.backward()
        
        grads = imgs.grad.cpu()

        imgs_out = torch.zeros_like(imgs)
        with tqdm(enumerate(grads), total = len(grads), leave = True) as t:
            for idx, grad in t:
                delta = eps * grad.sign().cpu()
                imgs_out[idx] = torch.clamp(imgs.data[idx].cpu() + delta, 0, 1)
                t.set_description(f'正在攻击图片{idx + 1}')
        return imgs_out.cpu().detach()