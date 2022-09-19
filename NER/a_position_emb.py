#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/11/3 下午6:15
# @Author  : PeiP Liu
# @FileName: position_emb.py
# @Software: PyCharm
import math
import torch
import torch.nn as nn
import numpy as np
# from torch.autograd import Variable

class APosEmb(nn.Module):
    # rf https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py
    def __init__(self, model_dim, dropout_rate, maxseq_len=512):
        super(APosEmb, self).__init__()
        self.register_buffer('position_embedding', self.get_sinusoid_encoding_table(maxseq_len, model_dim))  # 相当于属性赋值
        self.dropout = nn.Dropout(p=dropout_rate)

    def get_sinusoid_encoding_table(self, maxseq_len, model_dim):
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2*(model_dim_j//2)/model_dim) for model_dim_j in range(model_dim)]

        sinusoid_table = np.array([get_position_angle_vec(i_position) for i_position in range(maxseq_len)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim = 2i, : for all tokens of sequence but not the batch
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim = 2i+1
        return torch.tensor(sinusoid_table, dtype=torch.float32)
        # return torch.tensor(sinusoid_table, dtype=torch.float32).unsqueeze(0)  # shape = (batch, sent_len, model_dim)

    def forward(self, x, seq_len):
        # addition for position embedding and input_feature
        return self.dropout(x + self.position_embedding[:seq_len].clone().detach())
        # return self.dropout(x + self.position_embedding[:, :x.size(1)].clone().detach())

"""
class APosEmb(nn.Module):
    # rf nlp.seas.hard
    def __init__(self, seq_len, apos_dim, dropout_rate, device): # apos_dim = emb_dim
        super(APosEmb).__init__()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.ape = torch.zeros(seq_len, apos_dim)
        position = torch.arange(0, seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, apos_dim, 2) * -(math.log(10000.0) / apos_dim))
        self.ape[:, 0::2] = torch.sin(position * div_term)
        self.ape[:, 1::2] = torch.cos(position * div_term)
        self.ape = self.ape.unsqueeze(0).to(device)
        self.register_buffer('ape', self.ape)

    def forward(self, input_seq):  # 输入的形式是(batch_size, seq_len, emb_dim)
        # we also can fuse the position embedding by multiply or concatenation. However, we use the addition here.
        input_seq = input_seq + Variable(self.ape[:, :input_seq.size(1)], requires_grad=False)
        return self.dropout(input_seq)
"""
