#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/11/25 下午7:26
# @Author  : PeiP Liu
# @FileName: sampling.py
# @Software: PyCharm
import random
import torch
import sys
sys.path.append('..')
from RelE.utils import *

def create_train_sample(seq_len, reader_dataset, position, OBI, neg_rel_prop, device):
    """
    :param seq_len: 该数据是对原始输入token序列的长度计算
    :param reader_dataset: 该数据来自于对每条instance_data读取后的read()返回数据dataset
    :param rel_type_dict: 该数据是InputReader初始化后得到的rel_type_dict
    :param position: the class object from RPositionEmb which the seq_len is the most length of all
    :param OBI: 用于对每条关系中的实体进行标签编码
    :param neg_rel_prop: 用于采集负样本的比例。如果是0,则维持原来的全部负采样方式
    :return:
    """
    # rel_type_count = len(rel_type_dict)
    ents_list = reader_dataset.all_ents  # 数据里的所有实体对象
    rels_list = reader_dataset.all_rels  # 数据里的所有关系对象
    # 此处初始化一个相对位置类对象
    ent_sizes = []
    for each_ent in ents_list:
        # 实体列表中每个实体的跨度
        ent_sizes.append(each_ent.get_ent_size)

    rels, rel_spans, rel_types, rel_ent_masks, rel_tag_seqs, rel_pos_pair = [], [], [], [], [], []

    if neg_rel_prop != 0: #采样部分负样本
        # 先找出正样本和负样本
        filtered_pos_rels = []
        filtered_neg_rels = []
        for each_filter_rel in rels_list:
            if each_filter_rel.relation.get_name == 'None':
                filtered_neg_rels.append(each_filter_rel)
            else:
                filtered_pos_rels.append(each_filter_rel)
        filtered_neg_rels_num = round(len(filtered_pos_rels)*neg_rel_prop + 1e-9)

        # print('The number of filtered_pos_rels is {}'.format(len(filtered_pos_rels)))
        # print('The filtered_neg_rels_num is {}'.format(filtered_neg_rels_num))

        rels_list = filtered_pos_rels + random.sample(filtered_neg_rels, min(filtered_neg_rels_num, len(filtered_neg_rels)))

    for each_rel in rels_list:  # 遍历所有的关系对象
        head_ent, tail_ent = each_rel.rel_head, each_rel.rel_tail  # 获取头尾实体(类)对象
        s1, s2 = head_ent.get_ent_range, tail_ent.get_ent_range  # 获取头尾实体的范围(元组)
        rels.append((ents_list.index(head_ent), ents_list.index(tail_ent)))  # 确定是哪两个实体构成的关系
        rel_spans.append(([s1[0], s1[1]-1],[s2[0], s2[1]-1]))  # 这两个实体的范围
        rel_types.append(each_rel.relation.get_id)  # 该关系在关系字典中的id
        rel_ent_masks.append(create_seq_mask(s1,s2, seq_len))  # 第一个实体位置标１，第二个实体位置标２
        rel_tag_seqs.append(create_seq_tag(s1,s2, seq_len, OBI))  # 产生标签序列
        # 每个元素都是一个列表，每个表包含两个元素，第一个元素是针对第一个实体的相对位置，第二个元素是针对第二个实体的位置编码。
        rel_pos_pair.append(position.distance_index(seq_len, s1, s2))

    assert len(rels) == len(rel_ent_masks) == len(rel_types)

    """
    neg_entity_spans, neg_entity_sizes = [], []
    for size in range(1, max_span_size):
        for i in range(0,(token_count-size) + 1)
            span = tokens[i:i+size].span
            if span not in pos_entity_spans:
                neg_entity_spans.append(span)
                neg_entity_sizes.append(size)
                
    neg_rel_spans = []
    for i1, s1 in enumerate(pos_entity_spans):
        for i2, s2 in enumerate(pos_entity_spans):
            rev = (s2, s1)
            rev_symmetric = rev in pos_rel_spans and pos_rel_types[pos_rel_spans.index(rev)].symmetric
            if s1 != s2 and (s1, s2) not in pos_rel_spans and not rev_symmetric:
                neg_rel_spans.append((s1, s2))
    """

    if rels:
        ent_sizes = torch.tensor(ent_sizes, dtype=torch.long)
        rels = torch.tensor(rels, dtype=torch.long)
        rel_spans = torch.tensor(rel_spans, dtype=torch.long)
        rel_types = torch.tensor(rel_types,dtype=torch.long)
        rel_ent_masks = torch.stack(rel_ent_masks)
        rel_tag_seqs = torch.tensor(rel_tag_seqs, dtype=torch.long)
        rel_pos_pair = torch.tensor(rel_pos_pair, dtype=torch.long)
    else:
        print("There are no relations !")

    # rel_types_onehot = torch.zeros([rel_types.shape[0], rel_type_count], dtype=torch.float32)
    # rel_types_onehot.scatter_(1, rel_types.unsqueeze(1), 1)  # 构造关系groundtruth标签

    return dict(ent_sizes=ent_sizes.to(device), ent_index_pair=rels.to(device), ent_pair_span=rel_spans.to(device),
                rel_truth=rel_types.to(device),ent_pair_mask=rel_ent_masks.to(device),
                ent_pair_tags=rel_tag_seqs.to(device), ent_pair_pos=rel_pos_pair.to(device))

def create_test_sample(seq_len, pred_ents, position, OBI, device):
    # 参考_filter_spans()的实现
    """
    注：在测试时，我们同样需要groundtruth信息，至于完全取自reader_dataset，还是来自于create_train_sample()再议。
    :param: seq_len 是原始输入序列的长度
    :param pre_ents: 来自于ner_model里面的extract_ents().
    :return: 预测出的实体，构造出的所有关系可能
    """
    ent_sizes = []
    for each_ent in pred_ents:
        ent_sizes.append(each_ent[1]-each_ent[0])

    rels, rels_span, rel_ent_masks, rel_tag_seqs, rel_pos_pair = [], [], [], [], []
    for ent1_id, ent1_span in enumerate(pred_ents):
        for ent2_id, ent2_span in enumerate(pred_ents):
            if ent1_id != ent2_id:
                rels.append((ent1_id, ent2_id))
                rels_span.append(([ent1_span[0], ent1_span[1]-1], [ent2_span[0], ent2_span[1]-1]))  # 两个实体的跨度
                rel_ent_masks.append(create_seq_mask(ent1_span, ent2_span, seq_len))
                rel_tag_seqs.append(create_seq_tag(ent1_span, ent2_span, seq_len, OBI))  # 产生标签序列
                rel_pos_pair.append(position.distance_index(seq_len, ent1_span, ent2_span))

    assert len(rels) == len(rel_ent_masks) == len(rel_tag_seqs) == len(rel_pos_pair)

    if rels:
        ent_sizes = torch.tensor(ent_sizes, dtype=torch.long)
        rels = torch.tensor(rels, dtype=torch.long)
        rels_span = torch.tensor(rels_span, dtype=torch.long)
        rel_ent_masks = torch.stack(rel_ent_masks)
        rel_tag_seqs = torch.tensor(rel_tag_seqs, dtype=torch.long)
        rel_pos_pair = torch.tensor(rel_pos_pair, dtype=torch.long)
    else:
        print("No entities can be used for relations !")

    return dict(ent_sizes=ent_sizes.to(device), test_ent_index_pair=rels.to(device),
                test_ent_pair_span=rels_span.to(device), test_ent_pair_mask=rel_ent_masks.to(device),
                test_ent_pair_tags=rel_tag_seqs.to(device), test_ent_pair_pos=rel_pos_pair.to(device))

def create_seq_mask(s1, s2, seq_len):  # 在re_encoder后取特征时使用，<s1=1, s2=2> != <s2=1,s1=2>
    mask = torch.zeros(seq_len, dtype=torch.long)
    mask[s1[0]:s1[1]] = 1
    mask[s2[0]:s2[1]] = 2
    return mask

def create_seq_tag(s1, s2, seq_len, OBI):  # 只要是同一对s1和s2，结果是一样的

    seq_tag =  [OBI[0]] * seq_len # 元素列表*10 != 元素*10
    s1_b, s1_e = s1[0], s1[1]
    s2_b, s2_e = s2[0], s2[1]

    seq_tag[s1_b] = OBI[1]
    seq_tag[s1_b+1:s1_e] = (s1_e-s1_b-1)*[OBI[2]]

    seq_tag[s2_b] = OBI[1]
    seq_tag[s2_b+1:s2_e] = (s2_e-s2_b-1)*[OBI[2]]

    return seq_tag

def collate_fn_padding(batch):
    padded_batch = dict()
    keys = batch[0].keys()

    for key in keys:
        samples = [s[key] for s in batch]

        if not batch[0][key].shape:
            padded_batch[key] = torch.stack(samples)
        else:
            padded_batch[key] = padded_stack([s[key] for s in batch])

    return padded_batch
