import torch
from torch import nn
from torch.nn import functional as F

from model import BaseModel
from .base import BaseAttack

from tqdm import trange

class DeepFool(BaseAttack):
    def __init__(self, model: BaseModel, use_cuda: bool = True) -> None:
        super().__init__(model, use_cuda)

    def __call__(
        self, 
        imgs_in: torch.Tensor,
        labels: torch.Tensor,
        overshoot: float = 0.02,
        num_classes: int = 10,
        max_iter: int = 100
    ) -> torch.Tensor:
        super().__call__(imgs_in, labels)        
        
        imgs_out = torch.zeros_like(imgs_in)
        with trange(len(imgs_in)) as t:
            for i in t:
                imgs_out[i] = self.attack_one(imgs_in[i], overshoot, num_classes, max_iter)

        return imgs_out
    
    def attack_one(
        self, 
        img: torch.Tensor,
        overshoot: float = 0.02,
        num_classes: int = 10,
        max_iter: int = 100,
    ) -> torch.Tensor:
        img_clone, i = img.clone(), 0
        img_clone = img_clone.to(self.device)
        if img_clone.dim() == 3:
            img_clone.unsqueeze_(0)
            
        with torch.no_grad():
            raw_logits = self.model(img_clone)
            label = torch.argmax(F.softmax(raw_logits, dim = 1)).item()
        perturbed_label = label
        
        while i < max_iter:
            grads, fs = [], []
            self.model.zero_grad()
            img_clone.requires_grad_(True)
            if img_clone.grad is not None:
                img_clone.grad.zero_()
            logits: torch.Tensor = self.model(img_clone)
            with torch.no_grad():
                perturbed_label = F.softmax(logits, dim = 1).argmax().item()
                # print(f'perturbed_label: {perturbed_label}, label: {label}, i: {i}')
                if perturbed_label != label:
                    break
            logits[0, label].backward(retain_graph = True)
            grad_raw = img_clone.grad.clone()
            
            for k in range(num_classes):
                if k == label: continue
                self.model.zero_grad()
                if img_clone.grad is not None:
                    img_clone.grad.zero_()
                logits[0, k].backward(retain_graph = True)
                grad = img_clone.grad.clone()
                
                grads.append(grad)
                fs.append(logits[0, k].item())
            
            tmp1 = (torch.as_tensor(fs) - logits[0, label].item()).to(self.device)
            tmp2 = (torch.cat(grads, dim = 0) - grad_raw).reshape(9, -1).to(self.device)
            tmp3: torch.Tensor = torch.abs(tmp1) / torch.linalg.norm(tmp2, dim = 1)
            l = tmp3.argmin().item()
            
            r = tmp3[l] * grads[l]
            with torch.no_grad():
                img_clone += r * (1 + overshoot)
                img_clone = torch.clamp(img_clone, 0, 1)
            i += 1
            
        return img_clone.detach().cpu()
            
                
