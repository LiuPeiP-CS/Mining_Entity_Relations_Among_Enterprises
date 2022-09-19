#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/11/12 下午10:49
# @Author  : PeiP Liu
# @FileName: BertData.py
# @Software: PyCharm
import torch
import torch.nn
import sys
sys.path.append('..')
from structure_augmentation.alphabet import Alphabet


class BertData:
    def __init__(self):
        self.label_alphabet = Alphabet('label', True)

    def build_alphabet(self,data):
        for each_data in data:
            for label in each_data['label']:  # label_alphabet添加了<BOS>和<EOS>
                self.label_alphabet.add(label)

