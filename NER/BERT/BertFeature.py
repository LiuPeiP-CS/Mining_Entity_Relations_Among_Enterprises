#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/11/12 下午7:17
# @Author  : PeiP Liu
# @FileName: BertFeature.py
# @Software: PyCharm
import os
import torch
import torch.nn
import time
import datetime
import numpy as np
from tqdm import trange
import sys
sys.path.append('../..')
from data_utils import *
from NER.BERT.BertModel import Bert_CRF
from transformers import BertTokenizer, BertModel, BertConfig

class BertFeature(nn.Module):
    def __init__(self, BertPath, Bert_saved_path, label_alphabet, device):
        super(BertFeature, self).__init__()

        self.device = device

        self.bert_model = Bert_CRF(BertPath, label_alphabet, self.device)

        checkpoint = torch.load(Bert_saved_path + 'bert_crf_ner.checkpoint.pt')
        pretrained_model_dict = checkpoint['model_state']
        model_state_dict = self.bert_model.state_dict()
        # get the params interacting between model_state_dict and pretrained_model_dict
        selected_model_state = {k: v for k, v in pretrained_model_dict.items() if k in model_state_dict}
        model_state_dict.update(selected_model_state)
        # load the params into model
        self.bert_model.load_state_dict(model_state_dict)
        self.bert_model.to(self.device)  # gpu
        self.bert_model.eval()

    def forward(self, sentence):
        with torch.no_grad():
            # sentence = sentence.to(self.device)
            bert_feature = self.bert_model.get_feature(sentence) # (1, seq_len, dim)
            return bert_feature.squeeze(0)
            # return bert_feature[0] 与前一条命令相等