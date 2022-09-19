#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/12/3 下午4:58
# @Author  : PeiP Liu
# @FileName: re_attention.py
# @Software: PyCharm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

Attention_Module = {}


def build_attention(attention_type, *args, **kwargs):
    return Attention_Module[attention_type](*args, **kwargs)  # 相当于返回了类cls


def create_attention_name(name):

    def create_attention_class(cls):
        if name in Attention_Module:
            raise ValueError("The attention model has been created")
        if not issubclass(cls, BaseAttention):
            raise ValueError("Attention ({}:{}) must extend BaseAttention".format(name, cls.__name__))
        Attention_Module[name] = cls
        return cls

    return create_attention_class


class BaseAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError


@create_attention_name('global_context')
class GeneralAttention(BaseAttention):
    def __init__(self, q_dim, k_dim, dropout_rate=0.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(q_dim, k_dim), requires_grad=True)
        self.dropout = nn.Dropout(dropout_rate)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, q, k, v, attention_mask=None):
        # the input is 3-dim, where q.size() = k.size() = v.size() = (batch_size, seq_len, dim)
        # we also can use transpose instead of permute
        attention_matrix = q.matmul(self.weight).bmm(k.transpose(1, 2).contiguous())

        if attention_mask is not None:
            attention_matrix.masked_fill_(attention_mask, -np.inf)
            # masked_fill_() change the value self but masked_fill copy
            # attention_matrix = attention_matrix.masked_fill(attention_mask, -np.inf)

        soft_attention = F.softmax(attention_matrix, dim=-1)
        soft_attention = self.dropout(soft_attention)

        output = soft_attention.bmm(v)
        return output, soft_attention


@create_attention_name('span_fusion')
class DotProductAttention(BaseAttention):
    def __init__(self, dropout_rate=0.0, **kwargs):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, q, k, v, attetion_mask=None):
        # the input is 3-dim, where q.size() = k.size() = v.size() = (batch_size, seq_len, dim)
        attention_matrix = torch.bmm(q, k.permute(0, 2, 1).contiguous())  # (batch_size, sent_len, sent_len)

        if attetion_mask is not None:
            attention_matrix.masked_fill_(attetion_mask, -np.inf)  # (batch_size, q_sent_len, k_sent_len)

        # (num_head*batch_size, q_sent_len, k_sent_len). after softmax, the place of -np.inf will be converted to 0
        soft_attention = F.softmax(attention_matrix, dim=-1)
        soft_attention = self.dropout(soft_attention)

        output = torch.bmm(soft_attention, v)  # (batch_size, q_sent_len, v_dim)
        return output, soft_attention