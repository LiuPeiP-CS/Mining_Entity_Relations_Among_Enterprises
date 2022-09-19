#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/11/2 上午9:22
# @Author  : PeiP Liu
# @FileName: Biaffine.py
# @Software: PyCharm
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


class Biaffine(nn.Module):
    def __init__(self, ent1_feature, ent2_feature, out_feature, bias=(True, True)):
        super(Biaffine, self).__init__()
        self.ent1_feature = ent1_feature
        self.ent2_feature = ent2_feature
        self.out_feature = out_feature
        self.bias = bias
        self.linear_input_size = ent1_feature + int(bias[0])
        self.linear_output_size = out_feature*(ent2_feature + int(bias[1]))
        self.linear = nn.Linear(in_features=self.linear_input_size,
                                out_features=self.linear_output_size,
                                bias=False)

        # self.reset_parameters()

    # def reset_parameters(self):
    #     W = np.zeros((self.linear_output_size, self.linear_input_size), dtype=np.float32)
    #     self.linear.weight.data.copy_(torch.from_numpy(W))
    # def reset_parameters(self):
    #     # nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain('relu'))
    #     nn.init.normal_(self.linear.weight, mean=0, std=1)

    def forward(self, ent1, ent2):
        # print(ent1)
        # print(ent2)
        batch_size, len1, dim1 = ent1.size()  # 经过FFNN处理后的实体向量
        batch_size, len2, dim2 = ent2.size()  # 同理于ent1
        if self.bias[0]:
            ones = ent1.data.new(batch_size, len1, 1).zero_().fill_(1)
            ent1 = torch.cat([ent1, Variable(ones)], dim=2)
            dim1 = dim1 + 1
        if self.bias[1]:
            ones = ent2.data.new(batch_size, len2, 1).zero_().fill_(1)
            ent2 = torch.cat([ent2, Variable(ones)], dim=2)
            dim2 = dim2 + 1

        affine = self.linear(ent1)  # (batch_size, len1, self.out_feature*dim2)
        affine = affine.view(batch_size, len1*self.out_feature, dim2)
        ent2 = torch.transpose(ent2, 1, 2)  # (batch_size, dim2, len2)

        # (batch_size, len1*self.out_feature, len2) ->  (batch_size, len2, len1*self.out_feature)
        biaffine = torch.transpose(torch.bmm(affine, ent2), 1, 2)
        # biaffine = biaffine.contiguous().view(batch_size, len2, len1, self.out_feature)
        # return biaffine  # (batch_size, len2, len1, self.out_feature)
        return biaffine.squeeze(1)  # (batch_size, len1*self.out_feature)