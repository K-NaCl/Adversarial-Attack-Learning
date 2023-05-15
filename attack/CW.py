import torch
from torch import nn

from model import BaseModel
from .base import BaseAttack

from tqdm import tqdm


class CW(BaseAttack):
    def __init__(self, model: BaseModel, use_cuda: bool = True) -> None:
        super().__init__(model, use_cuda)
        self.loss_fn = nn.CrossEntropyLoss()
    
    def __call__(
        self,
        imgs_in: torch.Tensor,
        labels: torch.Tensor,
        max_iter: int = 200,
        num_classes: int = None,
    ) -> torch.Tensor:        
        super().__call__(imgs_in, labels)

        target_clses = (labels + 1) % 10
        att_imgs = torch.zeros_like(imgs_in)
        for i in range(len(imgs_in)):
            att_imgs[i] = self.attack_one(
                imgs_in[i], target_clses[i], max_iter
            )

        return att_imgs
    
    def f1(self, img_in, target_cls):
        logits = self.model(img_in.unsqueeze(0).to(self.device))
        target_cls = torch.tensor([target_cls], dtype = torch.long).to(self.device)
        # return (1 - self.loss_fn(logits, target_cls))
        return self.loss_fn(logits, target_cls)
        
    def opt_func(
        self, 
        img: torch.Tensor, 
        delta: torch.Tensor, 
        target_cls: int,
        c: float
    ):
        img_in = torch.clamp(img + delta, 0, 1)
        loss = delta.norm() + c * self.f1(img_in, target_cls)
        return loss

    def attack_one(
        self, 
        img: torch.Tensor,
        target_cls: int, 
        max_iter: int = 200,
        c: float = 0.05,
        lr: float = 0.001,
    ) -> torch.Tensor:
        self.model.to(self.device)
        img.requires_grad_(True)

        delta = torch.zeros_like(img, requires_grad = True)
        opt = torch.optim.Adam([delta], lr = lr)
        
        for idx in range(max_iter):
            opt.zero_grad()
            loss = self.opt_func(img, delta, target_cls, c)
            loss.backward()
            print(loss.item())
            opt.step()
        
        perturb_img = torch.clamp(img + delta.detach(), 0, 1)
        return perturb_img