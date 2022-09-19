#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/11/12 下午7:25
# @Author  : PeiP Liu
# @FileName: config.py
# @Software: PyCharm
import os
import torch
import numpy as np
import sys
sys.path.append('..')
from data_utils import *
import random

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
root_dir = '/data/liupeipei/paper/Enterprise_Relations'

def processing_origdata(data_file):

    dataset = json.loads(open(data_file).read())
    max_seq_length = dataset['max_len']  # the max length of sentence

    str_relation_dict = dataset['relations_dict']  # 关系字典，关系：关系id
    relation_dict = {}
    for rel_name, rel_sym in str_relation_dict.items():
        if rel_sym == "True":
            relation_dict[rel_name] = True
        elif rel_sym == "False":
            relation_dict[rel_name] = False
    print('The original relation_dict is {}'.format(relation_dict))

    data = dataset['data']  # data是一个列表，列表里面的每个元素是一个字典，代表了一条数据
    ent_cnt = []
    for each_data in data:
        ent_cnt.append(each_data['entities_num'])
    max_ent_cnt = max(ent_cnt)

    return max_seq_length, relation_dict, max_ent_cnt, data


class BasicArgs:
    BertPath = os.path.join(root_dir, 'NER/BERT/BERT_Chinese')
    GazVec = os.path.join(root_dir, 'pretrained_emb/Tencent_AILab_ChineseEmbedding.txt')
    WordVec = os.path.join(root_dir, 'pretrained_emb/Tencent_AILab_ChineseEmbedding.txt')
    Original_Dataset = os.path.join(root_dir, 'Data/final_data.json')
    Bert_saved_path = os.path.join(root_dir, 'NER/BERT/FinBERT') # the path where we save the trained model
    JointModel_saved_path = os.path.join(root_dir, 'Result/models/full')
    model_jpg = os.path.join(root_dir, 'Result/images/full')
    sem_aug_addr = os.path.join(root_dir, 'Result/token_aug.npy')
    result_record_addr = os.path.join(root_dir, 'Result/result_record.txt')

    max_seq_length, relation_dict, max_ent_cnt, data = processing_origdata(Original_Dataset)  # 记得划分train/dev/test
    data_piece = len(data) // 10
    random.shuffle(data) # 每次测评，使用不同的数据进行训练和测试
    train_data = data[: data_piece * 8]
    dev_data = data[data_piece * 8: data_piece * 9]
    test_data = data[data_piece * 9:]

    batch_size = 1
    max_seq_len = 512
    learning_rate = 1e-4 # 原始设置1e-4，该学习率是为了bert设置；联合模型的学习率在下面lr
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    total_epoch = 5 # 1用于测试整个程序的正常运行，2为了测试参数lr和re_loss，最终设置为40

    weight_decay_finetune = 1e-5
    lr_crf_fc = 1e-5
    weight_decay_crf_fc = 1e-5
    warmup_proportion = 0.002

    word_emb_dim = 200
    bert_emb_dim = 768
    hid_dim = 256

    transformer_num_layer = 4
    transformer_num_heads = 8
    transformer_mod_dim = 512
    transformer_ff_dim = 512

    unified_encoder_output_dim = 512

    semantic_num_sim = 3
    gaz_alphabet_emb_dim = 200
    dropout_rate = 0.2

    lr = 5e-4 # 5e-5与weight_decay＝0.01、re_bilstm_layers=2一起被证明是可行的
    weight_decay = 0.005 # 后面测试的是0.0005，前面测试的是0.01
    min_lr = 1e-9
    lr_decay_factor = 0.5
    nan_eps = 1e-4

    RE4NER = 1 # 这里表示在最终的损失中，关系抽取的损失与实体识别损失的比例，即re/ner

    neg_rel_prop = 0.5 # 用于采集每条数据中关系负样本的比例，该比例是与正样本(确定关系数量)的比较。负样本数量为pos_num*neg_rel_prop。如果是0,则维持原来的全部负采样方式
    rel_classifier = 'R' # 'L'或者是'R' # 分别代表线性分类器和双仿射分类器
    rel_classifier_linear_without_none = True #　rel_classifier = 'L'下使用; 如果是True,那么分类时去除None，只考虑19类；如果False, 则将None视为正常的一类
    rel_predict_threshold = 0.4 # rel_classifier = 'L'下使用;

    gradient_accumulation_steps = 40

    """
    *************************************上述主要面向NER，下面面向RE。*************************************
    """
    max_ent_size = 50  # 实体的尺寸大小
    max_distance = max_seq_length  # 其他token相对实体token的最远距离

    ent_size_emb_dim = 50  # 实体的尺寸向量维度
    label_emb_dim = 25  # 实体标签的向量表示
    rel_emb_dim = 100  # 关系标签的向量表示
    relative_dis_dim = 50  # 相对位置的向量表示
    re_cat_input_dim = 637  # relative_dis_dim*2+label_emb_dim+unified_encoder_output_dim
    re_unified_input_dim = 256
    re_hid_dim = 512  # 一定要保证和unified_encoder_output_dim一致
    re_bilstm_layers = 2 # 2被证明是可行的

    ent_unified_feature_dim = 512

    """
    *************************************下述主要面向re_mlp*************************************
    """
    mlp_input_dim = 512  # 要与re_hid_dim一致
    mlp_hid_dim = 256

