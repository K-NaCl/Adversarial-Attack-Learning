#----------------------------------------#
# 导入需要的包
#----------------------------------------#
import torch
from torch.utils import data
from torch import nn
from torchvision import datasets

import utils
import attack

import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt

import tqdm
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


#----------------------------------------#
# 设置基本参数
#----------------------------------------#

seed = 0
utils.set_random_seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_name = 'pre-resnet'
model_path = r'./results/CIFAR10/pre-resnet-bs128-lr0.01-seed0/models/pre-resnet-bs128-lr0.01-seed0'

dataset_name = 'CIFAR10'
data_channel = 3
batch_size = 128

num_examples = 5
metrics = {}
examples = {}

loss_fn = nn.CrossEntropyLoss(reduction='mean')


#----------------------------------------#
# 创建待评估模型与数据集
#----------------------------------------#

model = utils.creat_model(model_name, model_path, in_ch=data_channel, pretrained=True)
test_data = utils.data_iter(dataset_name, 'test', batch_size, 0)[1]


# None/FGSM/IFGSM
att_modes = ['FGSM']
# attack_modes = 'None'

#----------------------------------------#
# 无对抗攻击
#----------------------------------------#
if att_modes == 'None':

    loss, top1_acc = utils.test_accuracy(model, test_data)
    print(f'loss: {loss:.2e}\tTop 1 acc: {top1_acc * 100:4.2f}%')


#----------------------------------------#
# FGSM
#----------------------------------------#
if 'FGSM' in att_modes:
    metrics_temp = {}
    examples_temp = {}

    att_method = attack.FGSM(model)
    att_name = att_method.__class__.__name__

    epsilon = np.arange(0, 0.5, 0.05)
    with tqdm(epsilon, total = len(epsilon), leave = True) as t_eps:
        for eps in t_eps:

            att_accu = utils.Accumulator(5)            
            
            with tqdm(test_data, total = len(test_data), leave = True) as t_data:
                for imgs_in, labels in t_data:
                    
                    #----------------------------------------#
                    # 准备数据
                    #----------------------------------------#

                    raw_preds, raw_probs, Y_hat = utils.predict(model, imgs_in, dataset_name)

                    loss = loss_fn(Y_hat, labels)

                    suc_indices = raw_preds == labels
                    suc_imgs, suc_labels, suc_probs = imgs_in[suc_indices], labels[suc_indices], raw_probs[suc_indices]


                    #----------------------------------------#
                    # 开始攻击
                    #----------------------------------------#

                    attacks = torch.zeros_like(suc_imgs)

                    time_start = time.time()
                    attacks = att_method(suc_imgs, suc_labels, eps)
                    # BUG
                    # attacks = suc_imgs
                    time_end = time.time()

                    deltas = attacks - suc_imgs
                    att_norm = torch.mean(
                    torch.linalg.norm(deltas.reshape(len(deltas), -1), dim=1) 
                        / torch.linalg.norm(suc_imgs.reshape(len(deltas), -1), dim=1)
                    )


                    att_preds, att_probs, att_Y_hat = utils.predict(model, attacks, dataset_name)
                    att_loss = loss_fn(att_Y_hat, suc_labels)
                    att_suc_rate = (att_preds != suc_labels).sum() / len(att_preds)
                    att_acc = 1 - att_suc_rate
                    att_time = (time_end - time_start)/len(suc_imgs)

                    att_accu.add(att_acc, att_suc_rate, att_loss, att_norm, att_time)                    
            
            if eps == 0:
                att_suc_indices = att_preds == suc_labels
            else:
                att_suc_indices = att_preds != suc_labels

            examples_temp[eps] = {
                'suc_labels': suc_labels[att_suc_indices][0:num_examples].detach().cpu(),
                'suc_probs':suc_probs[att_suc_indices][0:num_examples].detach().cpu(),
                'att_preds':att_preds[att_suc_indices][0:num_examples].detach().cpu(),
                'att_probs':att_probs[att_suc_indices][0:num_examples].detach().cpu(),
                'imgs':attacks[att_suc_indices][0:num_examples].detach().cpu(),
                'deltas':deltas[att_suc_indices][0:num_examples].detach().cpu()
            }
            examples[att_name] = examples_temp

            num_batch = len(test_data)
            metrics_temp[eps] = {
                '模型精度'  : att_accu[0]/num_batch,
                '攻击成功率': att_accu[1]/num_batch,
                '损失大小'  : att_accu[2]/num_batch,
                '扰动大小'  : att_accu[3]/num_batch,
                '攻击时间'  : att_accu[4]/num_batch
            }
            metrics[att_name] = metrics_temp
            # print(f'{att_name}-{eps}:\n{metrics_temp}')

            torch.cuda.empty_cache()

if 'IFGSM' in att_modes:
    pass

if 'DeepFool' in att_modes:
    pass

print(f'imgs: {examples["FGSM"][0]["imgs"].shape}')

att_results_path = f'./results_attack/{dataset_name}/{model.name}'
att_metrics_path = f'{att_results_path}/metrics/'
att_tb_path = f'{att_results_path}/tb_out/'
paths = [att_metrics_path, att_tb_path]
for path in paths:
    if not os.path.exists(path):
        os.makedirs(path)
utils.check_path(att_results_path)


metrics_df= pd.DataFrame.from_dict({(i, j): metrics[i][j] for i in metrics.keys() for j in metrics[i].keys()})
print(metrics_df)
metrics_df.to_csv(att_metrics_path + 'att_metrics.csv', encoding='utf_8')


tb_writer = SummaryWriter(att_tb_path)

for att_mode in att_modes:
    plt.figure(att_mode, figsize=(30, 30))
    cnt = 0
    step = 0
    for eps in epsilon:
        ex = examples[att_mode][eps]
        for idx in range(num_examples):
            cnt += 1
            plt.subplot(len(epsilon), num_examples, cnt)
            plt.xticks([], [])
            plt.yticks([], [])
            if idx == 0:
                plt.ylabel("Eps: {}".format(eps), fontsize=14)
            suc_label, suc_prob = ex['suc_labels'][idx], ex['suc_probs'][idx]
            att_pred, att_prob = ex['att_preds'][idx], ex['att_probs'][idx]
            
            suc_tag = utils.class_names[dataset_name][suc_label]
            att_tag = utils.class_names[dataset_name][att_pred]

            img = ex['imgs'][idx]
            img_np = img.permute(1, 2, 0).numpy()

            plt.title(f'{suc_tag}({suc_prob*100:4.2f}%) -> {att_tag}({att_prob*100:4.2f}%)')
            plt.imshow(img_np)

        tb_writer.add_images(att_mode, ex['imgs'], step)
        step += 1
    

    plt.tight_layout()
    plt.show()