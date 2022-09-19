#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/11/2 下午5:08
# @Author  : PeiP Liu
# @FileName: Ent_FFNN.py
# @Software: PyCharm
import torch
import numpy as np
import torch.nn as nn


def orthonormal_initializer(output_size, input_size):
    """
    rf the code of paper 'Named Entity Recognition as Dependency Parsing'
    :param output_size:
    :param input_size:
    :return:
    """
    I = np.eye(output_size)
    lr = 0.1
    eps = 0.05/(output_size + input_size)
    success = False
    tries = 0
    while not success and tries < 10:
        Q = np.random.rand(input_size, output_size) / np.sqrt(output_size)
        for i in range(100):
            QTQmI = Q.T.dot(Q) - I
            loss = np.sum(QTQmI**2/2)
            Q2 = Q**2
            Q = Q - lr * Q.dot(QTQmI)/(
                    np.abs(Q2 + Q2.sum(axis=0, keepdims=True) + Q2.sum(axis=1, keepdims=True) - 1) + eps)
            if np.max(Q) > 1e6 or loss > 1e6 or not np.isfinite(loss):
                tries = tries + 1
                lr = lr / 2
                break
        success = True
    if success:
        print('Orthonormal pretrainer loss: %.2e' % loss)
    else:
        print("Orthonormal pretrainer failed, we will use the non-orthonormal random matrix")
        Q = np.random.randn(input_size, output_size)/np.sqrt(output_size)
    return np.transpose(Q.astype(np.float32))


class NonLinear(nn.Module):
    def __init__(self, input_size, hidden_size, activation = None):
        super(NonLinear, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear = nn.Linear(in_features=input_size, out_features=hidden_size)
        if activation is None:
            self._activate = lambda x: x
        else:
            if not callable(activation):
                raise ValueError("The activation must be callable: type={}".format(type(activation)))
            self._activate = activation
        # self.reset_parameters()

    def forward(self, x):
        y = self.linear(x)
        return self._activate(y)

    # def reset_parameters(self):  # initialize the parameters by ourselves but not the system
    #     # W = orthonormal_initializer(self.hidden_size, self.input_size)
    #     # self.linear.weight.data.copy_(W)
    #     nn.init.normal_(self.linear.weight, mean=0, std=1)
    #     b = np.zeros(self.hidden_size, dtype=np.float32)
    #     self.linear.bias.data.copy_(torch.from_numpy(b))


class EntFFNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EntFFNN, self).__init__()
        self.ffnn1 = NonLinear(input_size=input_size,  # 此处的尺寸需要按照实际的应用进行修正
                               hidden_size=hidden_size,  # 此处的尺寸需要按照实际的应用进行修正
                               # activation=nn.LeakyReLU(0.1))
                               activation=nn.Tanh())
        self.ffnn2 = NonLinear(input_size=input_size,  # 此处的尺寸需要按照实际的应用进行修正
                               hidden_size=hidden_size,  # 此处的尺寸需要按照实际的应用进行修正
                               # activation=nn.LeakyReLU(0.1))
                               activation=nn.Tanh())

    def forward(self, ent1, ent2):
        head = self.ffnn1(ent1)
        tail = self.ffnn2(ent2)
        return head, tail  # (rel_num, config.ent_unified_feature_dim)

