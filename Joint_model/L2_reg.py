#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/11/1 下午5:32
# @Author  : PeiP Liu
# @FileName: L2_reg.py
# @Software: PyCharm
import torch
import torch.nn as nn


def should_decay_parameter(name: str, param: torch.tensor) -> bool:
    if not param.requires_grad:
        return False
    elif 'batchnorm' in name.lower() or 'bn' in name.lower() or 'bias' in name.lower():
        return False
    elif param.ndim == 1:
        return False
    else:
        return True


def get_keys_to_decay(model: nn.Module) -> list:
    to_decay = []
    for name, param in model.named_parameters():
        if should_decay_parameter(name, param):
            to_decay.append(name)
    return to_decay


class L2(nn.Module):
    """
    this is the L2 regularization
    """

    def __init__(self, model: nn.Module, alpha: float):
        super().__init__()

        self.alpha = alpha
        self.keys = get_keys_to_decay(model)

    def forward(self, model):
        """
        rf
        :param model: your model
        :return:
        """
        l2_loss = 0
        for key, param in model.named_parameters():
            if key in self.keys:
                l2_loss = l2_loss + param.pow(2).sum() * 0.5
        return l2_loss * self.alpha
