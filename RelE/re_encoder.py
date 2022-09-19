#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/11/28 下午1:59
# @Author  : PeiP Liu
# @FileName: re_encoder.py
# @Software: PyCharm


import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('..')
from RelE.re_attention import build_attention
from RelE.Ent_FFNN import EntFFNN

class Re_encoder(nn.Module):
    def __init__(self, config, label_alphabet, dis_emb):
        super(Re_encoder, self).__init__()
        self.ent_size_emb_table = nn.Embedding(config.max_ent_size, config.ent_size_emb_dim)  # 构建实体标签的emb_table
        nn.init.xavier_normal_(self.ent_size_emb_table.weight)

        self.Label_emb = nn.Embedding(label_alphabet.size(), config.label_emb_dim)  # 构建实体标签的emb_table
        nn.init.xavier_normal_(self.Label_emb.weight)  

        self.distance_emb = dis_emb

        self.input_uniform = nn.Linear(config.re_cat_input_dim, config.re_unified_input_dim)
        self.dropout = nn.Dropout(config.dropout_rate)

        """
        **********************the above for relation encoder input**********************
        """

        self.device = config.device
        self.bilstm_hid_dim = config.re_hid_dim
        self.bilstm_layers = config.re_bilstm_layers
        self.dropout_rate = config.dropout_rate
        self.sigmoid = nn.Sigmoid()

        self.bilstm = nn.LSTM(config.re_unified_input_dim, self.bilstm_hid_dim//2, num_layers=config.re_bilstm_layers, batch_first=True, bidirectional=True)
        self.global_attention = build_attention('global_context', self.bilstm_hid_dim, self.bilstm_hid_dim, self.dropout_rate)

        """
        **********************the above for relation encoding**********************
        """
        self.cat_linear = nn.Linear(3*self.bilstm_hid_dim+config.ent_size_emb_dim, config.ent_unified_feature_dim)
        self.ents_ffnn = EntFFNN(config.ent_unified_feature_dim, config.ent_unified_feature_dim)
        self.ent_feature_norm = nn.BatchNorm1d(config.ent_unified_feature_dim)
        """
        **********************the above for ent encoding, and the following for mlp encoding**********************
        """
        self.mlp_linear1 = nn.Linear(config.mlp_input_dim, config.mlp_hid_dim)
        self.mlp_linear2 = nn.Linear(config.mlp_hid_dim, 1)
        self.mlp_relu = nn.ReLU()


    def init_lstm_hidden(self, rel_num, num_directs):
        return (torch.randn(self.bilstm_layers*num_directs, rel_num, self.bilstm_hid_dim//2).to(self.device),
                torch.randn(self.bilstm_layers*num_directs, rel_num, self.bilstm_hid_dim//2).to(self.device))


    def encoder_input(self, pair_tags, pair_pos, hidden_emb):
        """
        :param pair_tags: (rel_num, seq_len)，可以把rel_num当做batch_size来看待
        :param pair_pos: (rel_num, 2, seq_len)
        :param hidden_emb: (seq_len, config.unified_encoder_output_dim)
        :return:
        """
        # print(pair_pos.device)
        # print(self.distance_emb.device)
        # print('The pair_tags is {}'.format(pair_tags))
        # print('The pair_pos is {}'.format(pair_pos))
        # print('The hidden_emb is {}'.format(hidden_emb))

        rel_num, seq_len = pair_tags.shape
        _pair_pos = torch.transpose(pair_pos, 1, 2).contiguous()  # (rel_num, seq_len, 2)

        tag_feature = self.Label_emb(pair_tags)  # (rel_num, seq_len, label_emb_dim)
        dis_feature = self.distance_emb[_pair_pos].view(rel_num, seq_len, -1)  # (rel_num, seq_len, 2*relative_dis_dim)
        transformer_feature = hidden_emb.unsqueeze(0).repeat(rel_num, 1, 1)  # (rel_num, seq_len, unified_encoder_output_dim)

        # print(tag_feature.device)
        # print(dis_feature.device)
        # print(transformer_feature.device)
        cat_input_feature = torch.cat([tag_feature, dis_feature, transformer_feature], dim=-1)
        unified_input_feature = self.input_uniform(cat_input_feature)

        return self.dropout(unified_input_feature), self.dropout(transformer_feature)


    def get_entspan_emb(self, ent_mask, hidden_emb):
        # 该部分主要是求取实体span的表示
        expand_ent_mask = ent_mask.unsqueeze(-1) # (rel_num, seq_len, 1)
        padded_ent_span_emb = expand_ent_mask*hidden_emb  # (rel_num, seq_len, hidden_dim)
        tokens_score = self.mlp_score(padded_ent_span_emb)  # (rel_num, seq_len, 1)
        tokens_score = tokens_score + torch.log(expand_ent_mask.float()+1e-8) # (rel_num, seq_len, 1), for cleaning the noise tokens # +1e-8防止log(0)出现Nan
        ent_tokens_attention = F.softmax(tokens_score, dim=1)  # (rel_num, seq_len, 1)
        entspan_emb = torch.sum(ent_tokens_attention*padded_ent_span_emb, dim=1)  # (rel_num, hidden_dim)

        return entspan_emb


    def get_entbe_emb(self, ent_be_index, hidden_emb):
        """
        :param ent_be_index: (rel_num, 2)
        :param hidden_emb: (rel_num, seq_len, hidden_dim)
        :return:
        """
        rel_num, seq_len, hidden_dim = hidden_emb.shape
        ent_be_gather_index = ent_be_index.unsqueeze(-1).repeat(1, 1, hidden_dim)  # (rel_num, 2, hidden_dim)
        ent_be_emb = torch.gather(hidden_emb, 1, ent_be_gather_index) # (rel_num, 2, hidden_dim)
        ent_be_emb = ent_be_emb.view(rel_num, -1).contiguous()  # (rel_num, 2*hidden_dim)

        # 或者我们可以直接采用索引的方式，比较实现，但较难理解
        # 让每一个关系中的实体be去索引序列中的所有token
        # ent_be_emb = torch.stack([hidden_emb[i][ent_be_index[i]]for i in range(ent_be_index.shape[0])])

        return ent_be_emb


    def get_entsize_emb(self, allent_size_embs, ent_size_index):
        """
        :param allent_size_embs: (ent_num, ent_size_emb_dim)
        :param ent_size_index: (rel_num,)
        :return: (rel_num, ent_size_emb_dim)
        """
        return allent_size_embs[ent_size_index]


    def mlp_score(self, hidden_emb):
        """
        :param hidden_emb: (rel_num, seq_len, hidden_dim)
        :param output_size: 1
        :return: (rel_num, seq_len, 1)
        """
        mlp_emb1 = self.mlp_linear1(hidden_emb)
        mlp_relu_emb1 = self.mlp_relu(mlp_emb1)
        mlp_relu_emb1 = self.dropout(mlp_relu_emb1)
        mlp_emb2 = self.mlp_linear2(mlp_relu_emb1)
        attention_scores = self.mlp_relu(mlp_emb2)

        return attention_scores


    def encoder(self, ent_sizes:torch.tensor, ent_index_pair:torch.tensor, ent_pair_span:torch.tensor,
                ent_pair_mask:torch.tensor, ent_pair_tags:torch.tensor, ent_pair_pos:torch.tensor, rel_classifier:str, hidden_emb:torch.tensor):
        """
        :param ent_sizes: (ent_num, )
        :param ent_index_pair: (rel_num, 2)
        :param ent_pair_span: (rel_num, 2, 2)
        :param ent_pair_mask: (rel_num, seq_len)
        :param ent_pair_tags: (rel_num, seq_len)
        :param ent_pair_pos: (rel_num, 2, seq_len)
        :param hidden_emb: (seq_len, unified_encoder_output_dim) == (seq_len, re_hid_dim)
        :return:
        """
        rel_num, seq_len = ent_pair_tags.shape
        # (rel_num, seq_len, re_unified_input_dim), (rel_num, seq_len, unified_encoder_output_dim)
        fused_input_feature, extended_transformer_feature = self.encoder_input(ent_pair_tags, ent_pair_pos, hidden_emb)
        # print('The fused_input_feature is {}'.format(fused_input_feature))
        # print('The extended_transformer_feature is {}'.format(extended_transformer_feature))
        lstm_output, _ = self.bilstm(fused_input_feature, self.init_lstm_hidden(rel_num, 2))  # (rel_num, seq_len, re_hid_dim)
        # print('The lstm_output1 is {}'.format(lstm_output))
        lstm_output = self.sigmoid(lstm_output)
        # print('The lstm_output2 is {}'.format(lstm_output))
        attention_output, _ = self.global_attention(lstm_output, lstm_output, lstm_output)  # (rel_num, seq_len, re_hid_dim)
        span_input_feature = attention_output + extended_transformer_feature # (rel_num, seq_len, re_hid_dim)
        """
        ###########################for the first ent###########################
        """
        ent1_mask = ent_pair_mask == 1  # (rel_num, seq_len)
        ent1_span_emb = self.get_entspan_emb(ent1_mask, span_input_feature)
        # print('The ent1_span_emb in relation training is {}'.format(ent1_span_emb))

        ent1_be_index = ent_pair_span[:, 0, :]  # (rel_num, 2)
        ent1_be_emb = self.get_entbe_emb(ent1_be_index, span_input_feature)
        # print('The ent1_be_emb in relation training is {}'.format(ent1_be_emb))

        allent_size_embs = self.ent_size_emb_table(ent_sizes)  # (ent_num, ent_size_emb_dim)

        ent1_size_index = ent_index_pair[:, 0]  # (rel_num,)
        ent1_size_emb = self.get_entsize_emb(allent_size_embs, ent1_size_index)
        # print('The ent1_size_emb in relation training is {}'.format(ent1_size_emb))

        ent1_cat_feature = torch.cat([ent1_span_emb, ent1_be_emb, ent1_size_emb], dim=-1)
        ent1_feature = self.dropout(self.cat_linear(ent1_cat_feature))
        # print('The ent1_feature before ents_ffnn is {}'.format(ent1_feature))

        """
        ###########################for the second ent###########################
        """
        ent2_mask = ent_pair_mask == 2  # (rel_num, seq_len)
        ent2_span_emb = self.get_entspan_emb(ent2_mask, span_input_feature)

        ent2_be_index = ent_pair_span[:, 1, :]  # (rel_num, 2)
        ent2_be_emb = self.get_entbe_emb(ent2_be_index, span_input_feature)

        ent2_size_index = ent_index_pair[:, 1]  # (rel_num,)
        ent2_size_emb = self.get_entsize_emb(allent_size_embs, ent2_size_index)

        ent2_cat_feature = torch.cat([ent2_span_emb, ent2_be_emb, ent2_size_emb], dim=-1)
        ent2_feature = self.dropout(self.cat_linear(ent2_cat_feature))

        if rel_classifier == 'R':
            print("The embs of two ents before ents_ffnn are :\n{}\n{}".format(ent1_feature, ent2_feature))
            ent1_ffnn_rep, ent2_ffnn_rep = self.ents_ffnn(ent1_feature, ent2_feature) # (rel_num, config.ent_unified_feature_dim)
            print("The embs of two ents after ents_ffnn are :\n{}\n{}".format(ent1_ffnn_rep, ent2_ffnn_rep))

            ent1_biaffine_input = ent1_ffnn_rep.unsqueeze(1)
            ent2_biaffine_input = ent2_ffnn_rep.unsqueeze(1)

            normed_ent1_biaffine_input = self.ent_feature_norm(ent1_biaffine_input.transpose(1,2)).transpose(1,2)
            normed_ent2_biaffine_input = self.ent_feature_norm(ent2_biaffine_input.transpose(1, 2)).transpose(1, 2) # 数据归一化

            return normed_ent1_biaffine_input, normed_ent2_biaffine_input  # (rel_num, 1, config.ent_unified_feature_dim)

        elif rel_classifier == 'L':

            return ent1_feature, ent2_feature

