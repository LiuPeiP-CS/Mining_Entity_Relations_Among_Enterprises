#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/11/25 下午4:41
# @Author  : PeiP Liu
# @FileName: utils.py
# @Software: PyCharm

import torch
import torch.nn as nn


def extend_tensor(tensor, extended_shape, fill_value=0):
    tensor_shape = tensor.shape

    extended_tensor = torch.zeros(extended_shape, dtype=tensor.dtype).to(tensor.device)
    extended_tensor = extended_tensor.fill_(fill_value)

    if len(tensor_shape) == 1:
        extended_tensor[:tensor_shape[0]] = tensor
    elif len(tensor_shape) == 2:
        extended_tensor[:tensor_shape[0], :tensor_shape[1]] = tensor
    elif len(tensor_shape) == 3:
        extended_tensor[:tensor_shape[0], :tensor_shape[1], :tensor_shape[2]] = tensor
    elif len(tensor_shape) == 4:
        extended_tensor[:tensor_shape[0], :tensor_shape[1], :tensor_shape[2], :tensor_shape[3]] = tensor


def padded_stack(tensors, padding_value=0):
    dim_count = len(tensors[0].shape)

    max_shape = [max([t.shape[d] for t in tensors]) for d in range(dim_count)]  # 对每个tensor进行遍历，在每个维度上取所有tensor的最大值
    padded_tensors = []

    for t in tensors:
        e = extend_tensor(t, max_shape, fill=padding_value)
        padded_tensors.append(e)

    stacked = torch.stack(padded_tensors)
    return stacked

def batch_index(tensor, index, pad=False):
    if tensor.shape[0] != index.shape[0]:
        raise Exception

    if pad:
        return padded_stack([tensor[i][index[i]] for i in range(index.shape[0])])
    else:
        torch.stack([tensor[i][index[i]] for i in range(index.shape[0])])