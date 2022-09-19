#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/11/25 上午10:47
# @Author  : PeiP Liu
# @FileName: data.py
# @Software: PyCharm

import torch
import torch.nn as nn
import sys
sys.path.append('..')
from NER.structure_augmentation.ner_data import NERData

class JointData(nn.Module):
    def __init__(self, conf):
        super(JointData, self).__init__()

        self.conf = conf
        self.ner_data = NERData()

        self.ner_data.build_alphabet(self.conf.data)

        self.ner_data.build_gaz_file(self.conf.GazVec)  # build the trie for matching

        self.ner_data.build_gaz_alphabet(self.conf.data)

        self.ner_data.get_matched_with_gaz(self.conf.train_data, 'train')  # match the train_data with gaz, and return train_ids.
        self.ner_data.get_matched_with_gaz(self.conf.dev_data, 'dev')  # the same as train
        self.ner_data.get_matched_with_gaz(self.conf.test_data, 'test')  # the same as train

        self.ner_data.build_word_embedding(self.conf.WordVec)  # get the word embedding table
        self.ner_data.build_gaz_alphabet_embedding(self.conf.GazVec)  # get the structure augmentation embedding table

    @property
    def get_nerdata_object(self):
        return self.ner_data