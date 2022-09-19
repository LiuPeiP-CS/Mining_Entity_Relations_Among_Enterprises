#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/10/14 下午4:28
# @Author  : PeiP Liu
# @FileName: PGD_adv.py
# @Software: PyCharm

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import sys
sys.path.append('..')


# the first method
# rf https://fyubang.com/2019/10/15/adversarial-train/
class ModelPGD(nn.Module):  # 在模型层次进行扰动
    """
    this is the adversarial training learning
    """

    def __init__(self, model, epsilon=1, alpha=0.3):
        super(ModelPGD).__init__()
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.emb_backup = {}
        self.grad_backup = {}

    def forward(self, emb_name='emb.', is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()  # identify data and name of params to emb_name
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, self.epsilon)

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]  # 新数据与旧数据的相差,可以认为是梯度
        if torch.norm(r) > epsilon:  # 限制数据扰动的范围
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def restore(self, emb_name='emb.'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = self.grad_backup[name]


"""
# ths sample of application
pgd = ModelPGD(model=model)
K = 3
for batch_input, batch_label in gen_data(data):
    loss = model(batch_input, batch_label)
    loss.backward()
    pgd.backup_grad()
    for t in range(K):
        pgd(is_first_attack=(t == 0))
        if t == K-1: # the last time
            model.zero_grad()
        else:
            pgd.restore_grad()
        loss_adv = model(batch_input, batch_label)
        loss_adv.backward()
    pgd.restore()
    optimizer.step()
    model.zero_grad()
"""


# the second method
# https://github.com/karandwivedi42/adversarial/blob/master/main.py
class DataPGD(nn.Module):  # 在数据层次进行扰动
    def __init__(self, model, K=10, epsilon=1, alpha=0.3, random_start=True):
        super(DataPGD).__init__()
        self.model = model
        self.adv_times = K  # 扰动次数
        self.epsilon = epsilon
        self.alpha = alpha
        self.rand = random_start

    def forward(self, batch_inputdata, batch_label):  # 输入是torch.tensor类型，在输入数据层次进行扰动
        x = batch_inputdata.detach()
        if self.rand:
            x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)  # 加入扰动因子
        for i in range(self.adv_times):
            x.requires_grad_()  # 需要求梯度
            with torch.enable_grad():  # 允许局部的梯度计算
                scores = self.model(x)
                loss = self.model.loss(scores, batch_label)
            grad = torch.autograd.grad(loss, [x])[0]  # 获取数据梯度
            x = x.detach() + self.alpha * torch.sign(grad.detach())
            x = torch.min(torch.max(x, batch_inputdata - self.epsilon), batch_inputdata + self.epsilon)
            x = torch.clamp(x, 0, 1)  # 归一化处理
        return self.model(x), x


"""
# the sample of application
model = DataPGD(model)  # 输入模型是原始的模型，输出是添加扰动后的模型
if device == 'CUDA':
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

optimizer = optim.SGD(model.parameters(), lr=lr, ...) # 后续模型使用和正常一样
"""

# the third method
# https://github.com/wanglouis49/pytorch-adversarial_box/blob/master/mnist_adv_train.py
# 数据扰动，但是没看懂...
