#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/11/9 下午5:32
# @Author  : PeiP Liu
# @FileName: module.py
# @Software: PyCharm

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
# from transformers.modeling_bert import BertModel

class SModule(nn.Module):
    def __init__(self, data, device):
        super(SModule, self).__init__()
        self.word_emb_dim = data.word_emb_dim
        self.gaz_alphabet_emb_dim = data.gaz_alphabet_emb_dim

        self.gaz_embedding = nn.Embedding(data.gaz_alphabet.size(), self.gaz_alphabet_emb_dim)
        self.gaz_embedding.weight.data.copy_(torch.from_numpy(data.gaz_alphabet_embedding))

        self.word_embedding = nn.Embedding(data.word_alphabet.size(), self.word_emb_dim)
        self.word_embedding.weight.data.copy_(torch.from_numpy(data.word_embedding))

        self.feature_dim = self.word_emb_dim + 4*self.gaz_alphabet_emb_dim  # 串接原始词向量和匹配增强向量

        # self.bert_encoder = BertModel.from_pretrained(data.bert_path)
        self.device = device

    def forward(self, instance_input):
        words = instance_input[1]
        # labels = instance_input[3]
        layer_gazs = instance_input[4]  # 匹配出的词在gaz中的id，填充信息用0表示
        gaz_count = instance_input[5]  # 对应layer_gazs中匹配出来的词的id，在gaz_alphabet中出现的频次,填充信息用0表示
        gaz_mask = instance_input[6]  # 对每个B/E/M/S进行填充内容的mask,有效位置有0,无效位置为1;为了后面的masked_fill_()填充
        # bert_ids = instance_input[4]

        word_seq_len = len(words)
        word_seq_tensor = torch.zeros(word_seq_len, dtype=torch.long).to(self.device)
        # label_seq_tensor = autograd.Variable(torch.zeros(word_seq_len)).long()
        # mask = autograd.Variable(torch.zeros(word_seq_len)).byte()

        # bert_seq_tensor = autograd.Variable(torch.zeros(word_seq_len + 2)).long()
        # bert_mask = autograd.Variable(torch.zeros(word_seq_len + 2)).long()

        gaz_num = len(layer_gazs[0][0])  # 取第一个token的B上的数量
        layer_gaz_tensor = torch.zeros(word_seq_len, 4, gaz_num).long().to(self.device)
        gaz_count_tensor = torch.zeros(word_seq_len, 4, gaz_num).float().to(self.device)
        gaz_mask_tensor = torch.ones(word_seq_len, 4, gaz_num).byte().to(self.device)

        word_seq_tensor[:word_seq_len] = torch.LongTensor(words).to(self.device)
        # label_seq_tensor[:word_seq_len] = torch.LongTensor(labels)
        layer_gaz_tensor[:word_seq_len, :, :gaz_num] =torch.LongTensor(layer_gazs).to(self.device)
        # mask[:word_seq_len] = torch.Tensor([1]*int(word_seq_len))
        # bert_mask[:word_seq_len+2] = torch.LongTensor([1]*int(word_seq_len+2))
        gaz_mask_tensor[:word_seq_len, :, :gaz_num] = torch.ByteTensor(gaz_mask).to(self.device)
        gaz_count_tensor[:word_seq_len, :, :gaz_num] = torch.FloatTensor(gaz_count).to(self.device)
        gaz_count_tensor[word_seq_len:] = 1
        # bert_seq_tensor[:word_seq_len+2] = torch.LongTensor(bert_ids)

        word_emb = self.word_embedding(word_seq_tensor)  # (seq_len, emb_dim)

        gaz_embeds = self.gaz_embedding(layer_gaz_tensor)  # 填充后的embedding, (seq_len, 4, gaz_num, gaz_alphabet_emb_dim)
        gaz_mask_tensor = gaz_mask_tensor.unsqueeze(-1).repeat(1,1,1,self.gaz_alphabet_emb_dim)
        # 在gaz_mask_tensor为True的位置上全部填充0,则需要gaz_mask_tensor表示有效位置时为０.因此gaz_embeds在1(无效)填充０
        gaz_embeds = gaz_embeds.data.masked_fill_(gaz_mask_tensor.data, 0)

        # 以下操作，相当于进行了匹配数量上的norm
        count_sum = torch.sum(gaz_count_tensor, dim=2, keepdim=True) # (seq_len, 4, 1),每个B/M/E/S的集合中数字相加
        count_sum = torch.sum(count_sum, dim=1, keepdim=True) # (seq_len, 1, 1), 每个word的B/M/E/S的总数相加
        weights = gaz_count_tensor.div(count_sum)
        weights = weights*4
        weights = weights.unsqueeze(-1)

        structure_aug_emb = weights*gaz_embeds
        structure_aug_emb = torch.sum(structure_aug_emb, dim=2)  # 对B/M/E/S内部的向量集合进行相加

        structure_aug_emb = structure_aug_emb.view(word_seq_len, -1)  # (seq_len, emb_dim)

        word_input_emb = torch.cat([word_emb, structure_aug_emb], dim=-1) # (seq_len, self.feature_dim)

        return word_input_emb
