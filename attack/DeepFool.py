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
        image: torch.Tensor,
        overshoot: float = 0.02,
        num_classes: int = 10,
        max_iter: int = 100,
    ) -> torch.Tensor:
        # 将图像张量转为可求导的状态，并将其复制为一个新的 tensor，用于存储扰动向量
        img_tensor = image.clone().detach().to(self.device)
        img_tensor.unsqueeze_(0)
        img_tensor.requires_grad_(True)
        perturbation = torch.zeros_like(img_tensor)

        # 对抗样本分类预测标签和置信度
        f_image = self.model(img_tensor).data.cpu().numpy().flatten()
        I = f_image.argsort()[::-1]

        # 最大类别及其分数
        label0 = I[0]
        score0 = f_image[label0]

        # 用语句计数器初始化迭代次数和扰动的步数
        count = 0
        k = label0

        while k == label0 and count < max_iter:
            # 计算关于当前样本的梯度
            input_var = img_tensor.requires_grad_(True)
            output = self.model(input_var)
            pred_label = output.data.max(1, keepdim=True)[1][0].item()
            if pred_label != label0:
                break
            grad_f = torch.zeros_like(output)
            grad_f[0][k] = 1
            output.backward(gradient=grad_f, retain_graph=True)

            # 计算当前样本到决策边界的距离（即最小扰动的模）
            w_k = (self.model.fc.weight.data[label0, :] - self.model.fc.weight.data[k, :]).clone().detach().to(self.device)
            f_k = self.model.fc.bias.data[label0] - self.model.fc.bias.data[k]
            distance = abs(f_k) / torch.norm(w_k)

            # 将各个标签之间决策边界的线性分隔超平面的法向量进行线性组合
            pert = torch.zeros_like(self.model.fc.weight.data[0, :]).to(self.device)
            for i in range(self.model.fc.weight.shape[0]):
                if i != k:
                    w_i = torch.tensor(self.model.fc.weight.data[label0, :] - self.model.fc.weight.data[i, :]).to(self.device)
                    f_i = self.model.fc.bias.data[label0] - self.model.fc.bias.data[i]
                    pert_i = abs(f_i) / torch.norm(w_i) * w_i
                    pert = pert + pert_i

            # 计算对抗扰动，保证其尽量小
            w_norm = torch.norm(pert, p=2)

            r_i = (pert + 1e-4) * torch.ones_like(pert).to(self.device) * pert / w_norm
            print(r_i.shape)
            perturbation += r_i


            # 对抗攻击计算当前样本的类别
            img_tensor = img_tensor + (1 + overshoot) * r_i
            output = self.model(img_tensor)
            k = output.data.max(1, keepdim=True)[1][0].item()

            count += 1

        # 将图像还原为原始状态，并将扰动裁剪到[-epsilon, epsilon] 区间内
        img_adv = torch.clamp(img_tensor - image, min=-0.2, max=0.2) + image

        return img_adv.detach().cpu()
            
                
