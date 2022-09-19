#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/11/12 上午9:57
# @Author  : PeiP Liu
# @FileName: BertModel.py
# @Software: PyCharm

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertModel, BertConfig
from torchcrf import CRF
import torch.nn.functional as F
# rf https://pytorch-crf.readthedocs.io/en/stable/
# also, we think the numtags contain the start and end, so the numtags=5 if you have valid 3 tags


class Bert_CRF(nn.Module):
    def __init__(self, bert_path, label_alphabet, device):
        super(Bert_CRF, self).__init__()

        self.tokenizer = BertTokenizer.from_pretrained(bert_path, do_lower_case=True)
        config = BertConfig.from_pretrained(bert_path, output_hidden_states=True)
        self.bert_encoder = BertModel.from_pretrained(bert_path, config=config)

        self.label_alphabet = label_alphabet
        self.emission_size = self.label_alphabet.size()

        self.device = device

        self.hid2label = nn.Linear(768, self.emission_size)  # 768 is the emb_dim from BERT
        nn.init.xavier_uniform_(self.hid2label.weight)
        nn.init.constant_(self.hid2label.bias, 0.0)

        self.dropout = nn.Dropout(0.5)
        self.bert_sigmoid = nn.Sigmoid()

        self.crf = CRF(self.emission_size, batch_first=True)

    def data_processor(self, sentence):
        """
        :param sentence: the one sentence with a list of words
        :param sentence_label: the one sentence label with a list of labels
        :return:
        """
        bert_text = ['[CLS]'] + sentence + ['[SEP]']
        bert_text_ids = self.tokenizer.convert_tokens_to_ids(bert_text)  # len(original)+2

        bert_seq_tensor = torch.tensor(bert_text_ids, dtype=torch.long).to(self.device)
        bert_mask = torch.tensor([1]*int(len(sentence)+2)).to(self.device)
        seg_id = torch.zeros(bert_mask.size()).long().to(self.device)

        return bert_seq_tensor.unsqueeze(0), bert_mask.unsqueeze(0), seg_id.unsqueeze(0) # (batch_size, seq_len)

    def get_feature(self, sentence):
        """
        :param bert_seq_tensor: from data_processor()
        :param bert_mask: from data_processor()
        :param seg_id: from data_processor()
        :return: the bert feature for crf
        """
        bert_seq_tensor, bert_mask, seg_id = self.data_processor(sentence)
        bert_output = self.bert_encoder(input_ids=bert_seq_tensor, attention_mask=bert_mask, token_type_ids=seg_id)
        bert_emb = bert_output.last_hidden_state  # (1, seq_len, bert_dim)
        # bert_emb = bert_emb.squeeze(0)
        bert_emb = bert_emb[:, 1:-1, :]  # (seq_len, bert_dim)

        dropout_feature = self.dropout(bert_emb)

        return dropout_feature # (1, seq_len, dim)

    def neg_log_likehood(self, sentence, sentence_label):
        bert_feature = self.get_feature(sentence)
        emission_feature = self.hid2label(bert_feature)
        emission_feature = self.bert_sigmoid(emission_feature)

        true_label_ids = [[self.label_alphabet.get_index(label) for label in sentence_label]]  # len(original)
        true_label_tensor = torch.tensor(true_label_ids, dtype=torch.long).to(self.device)

        object_score = - self.crf(emission_feature, true_label_tensor)
        return object_score

    def decoder(self, sentence):
        bert_feature = self.get_feature(sentence)
        emission_feature = self.hid2label(bert_feature)
        emission_feature = self.bert_sigmoid(emission_feature)

        return self.crf.decode(emission_feature)[0] # [list]
