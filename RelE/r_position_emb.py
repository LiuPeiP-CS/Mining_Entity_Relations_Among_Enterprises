#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/10/18 下午2:22
# @Author  : PeiP Liu
# @FileName: position_emb.py
# @Software: PyCharm
import torch
import math
import numpy as np
import torch.nn as nn

all_zeros = "ALL_ZERO"
unknown = "UNKNOWN"

def init_random(elements, embedding_dim, add_all_zeros=False, add_unknown=False):
    """
    :param elements: collection of elements to construct the embedding matrix
    :param embedding_dim: size of the embedding
    :param add_all_zeros: add an all zero embedding at the index 0
    :param add_unknown: add unknown embedding at the last index
    :return: an embedding matrix and a dictionary mapping elements to rows in the matrix
    """
    scale = np.sqrt(3.0 / embedding_dim)

    elements = sorted(elements)
    elements2index = {all_zeros: 0} if add_all_zeros else {}
    elements2index.update({ele : index for index, ele in enumerate(elements, start=len(elements2index))})
    if add_unknown:
        elements2index[unknown] = len(elements2index)
    # embeddings = np.random.random(size=(len(elements2index), embedding_dim)).astype('float32')
    embeddings = np.random.uniform(-scale, scale, [len(elements2index), embedding_dim]).astype('float32')
    if add_all_zeros:
        embeddings[0] = np.zeros([embedding_dim])

    return embeddings, elements2index


class RPositionEmb(nn.Module):
    def __init__(self, config):
        super(RPositionEmb, self).__init__()
        self.position_emb, self.position2index = init_random(np.arange(-config.max_seq_len, config.max_seq_len), config.relative_dis_dim)

    def distance_token2entity(self, entity_tokens_positions, token_position): # the distance of any other token-to-entity
        """
        :param entity_tokens_positions: the position of entity, such as [4,5,6]
        :param token_position: the position of any other token, such as [0] or 9
        :return: the nearest distance between entity and other token, such as [0-4]=[-4] or [9-6]=[3]
        """
        if len(entity_tokens_positions) < 1:
            entity_tokens_positions = np.array([-1])
        return (token_position-np.array(entity_tokens_positions))[np.abs(token_position-np.array(entity_tokens_positions)).argmin()]

    def distance_index(self, seq_len, ent1, ent2): # the sentence_tokens have be cropped, but not padding
        """
        :param seq_len: the sentence length of our input
        :param ent1: the index of the left entity, a list such as [4,5,6]
        :param ent2: the index of the right entity, the same as left
        :param position2index: each dis value will be mapped into a vector, so they should have a dictionary like word_dic and word2index
        where _, position2index = init_random(np.arange(-max_len, max_len), 1, add_all_zeros=True) is in the main.py
        :return:
        """
        """
        # 不区分ent1和ent2哪个在前面，哪个在后
        if ent2[0] < ent1[0]:
            left = [i for i in range(ent2[0], ent2[1])]
            right = [i for i in range(ent1[0], ent1[1])]
        else:
            left = [i for i in range(ent1[0], ent1[1])]
            right = [i for i in range(ent2[0], ent2[1])]
        """
        # 区分ent1和ent2哪个在前，哪个在后
        left = [i for i in range(ent1[0], ent1[1])]
        right = [i for i in range(ent2[0], ent2[1])]

        left_dis_to_index = []
        right_dis_to_index = []
        for i in range(seq_len):
            left_dis_to_index.append(self.position2index[self.distance_token2entity(left, i)])
            right_dis_to_index.append(self.position2index[self.distance_token2entity(right, i)])
        return [left_dis_to_index, right_dis_to_index]
