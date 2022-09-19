#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/11/9 上午9:07
# @Author  : PeiP Liu
# @FileName: data.py
# @Software: PyCharm
import sys
import numpy as np
sys.path.append('../..')
from NER.structure_augmentation.alphabet import Alphabet
from NER.structure_augmentation.functions import *
from NER.structure_augmentation.gazetteer import Gazetteer


class NERData:
    def __init__(self):

        # self.bert_path = bert_path
        self.max_seqlen = 512

        self.word_alphabet = Alphabet('word') # 对输入的数据集（所有词或字符）进行了字典的构建
        self.label_alphabet = Alphabet('label', True) # 对输入数据集的标签进行了字典的构建
        self.gaz_lower = False  # main for english, but not the chinese
        self.gaz = Gazetteer(self.gaz_lower) # 由外部支撑预料构建的字典树trie及相应的dict
        self.gaz_alphabet = Alphabet('gaz')  # (输入数据集)所有匹配出的词构建了字典
        self.gaz_count = {}  # 用于统计匹配出的词出现的频率(三个文件一起的结果), 词在gaz_alphabet中的id:词出现的频率

        self.seq_lens = []  # 用于记录每条数据的长度

        self.train_ids = []
        self.dev_ids= []
        self.test_ids = []

        self.word_embedding = None
        self.word_emb_dim = 0  # 初始化为0

        self.alphabet_embedding = None
        self.alphabet_emb_dim = 0  # 初始化为0

    def build_alphabet(self, data): # the input is all the data from train or dev or test
        # 这里的输入数据是每个文件(train/dev/test)中的数据内容，来建立字典
        for each_data in data:
            for word in each_data['token']:  # word_alphabet添加了<UNK>
                self.word_alphabet.add(word)  # 此处只需要读出每个字符就行
            for label in each_data['label']:  # label_alphabet添加了<BOS>和<EOS>
                self.label_alphabet.add(label)
            self.seq_lens.append(len(each_data['token']))

        self.word_alphabet_size = self.word_alphabet.size()
        self.label_alphabet_size = self.label_alphabet.size()

    def build_gaz_file(self, gaz_file): # we build the gazetteer based on trie, and it can help us with
        # 构建gaz字典树,便于查询匹配词以及其在字典中的索引
        if gaz_file:
            fins = open(gaz_file, 'r', encoding='utf-8').readlines()
            for fin in fins:
                fin = fin.strip().split()[0]  # we only get the word here
                if fin:
                    self.gaz.insert(fin, 'one_source')
            print("Load gaz file: ", gaz_file, " total size: ", self.gaz.size())
        else:
            print("Gaz file is None, load nothing")

    def build_gaz_alphabet(self, data):  # the input is the same as build_alphabet()
        # 数据输入的内容和格式与build_alphabet()相同
        # 该函数的目标在于，对每个word实现匹配，找出匹配的词、词在gaz中的索引、以及匹配词出现的频率
        seq_list = [each_data['token'] for each_data in data]
        for each_seq in seq_list:
            seq_len = len(each_seq)
            # entities = []
            for word_idx in range(seq_len):
                # 找出该字能匹配出的gaz中的所有的词
                matched_word_list = self.gaz.enumerateMatchList(each_seq[word_idx:]) # 匹配所有以word_idx开头的词汇，并返回所有的匹配字符串
                # entities = entities + matched_word_list
                for entity in matched_word_list:
                    self.gaz_alphabet.add(entity)
                    index = self.gaz_alphabet.get_index(entity)
                    self.gaz_count[index] = self.gaz_count.get(index, 0)  # 记录匹配出的词，匹配成功的频率；实际上进行的是数据初始化过程
                # 针对每一次匹配，而非整个句子的匹配
                matched_word_list.sort(key=lambda x: -len(x))  # sort by the reversed length
                while matched_word_list:
                    longest_entity = matched_word_list[0]
                    longest_entity_idx = self.gaz_alphabet.get_index(longest_entity)
                    self.gaz_count[longest_entity_idx] = self.gaz_count.get(longest_entity_idx, 0) + 1

                    gaz_length = len(longest_entity)
                    # 删除匹配出的子词(实际上是在此次的匹配中忽略子词的的频率)，包括删除自己以防止死循环
                    for i in range(gaz_length):
                        for j in range(i+1, gaz_length+1):
                            covered_entity = longest_entity[i:j]
                            if covered_entity in matched_word_list:
                                matched_word_list.remove(covered_entity)

        print("The gaz alphabet is built done !")

    def done_alphabet(self):
        self.gaz_alphabet.close()
        self.label_alphabet.close()
        self.word_alphabet.close()

    def get_matched_with_gaz(self, data, name): # 数据的输入形式和上述相同，都是数据集中的data内容
        # all_sentences = [each_data['token'] for each_data in data]
        # all_labels = [each_data['label'] for each_data in data]
        if name == 'train':
            self.train_ids = read_instance_with_gaz(
                data, self.gaz,
                self.word_alphabet, self.gaz_alphabet, self.gaz_count,
                self.label_alphabet, self.max_seqlen
            )

        elif name == 'dev':
            self.dev_ids = read_instance_with_gaz(
                data, self.gaz,
                self.word_alphabet, self.gaz_alphabet, self.gaz_count,
                self.label_alphabet, self.max_seqlen
            )

        elif name == 'test':
            self.test_ids = read_instance_with_gaz(
                data, self.gaz,
                self.word_alphabet, self.gaz_alphabet, self.gaz_count,
                self.label_alphabet, self.max_seqlen
            )

    def build_word_embedding(self, emb_path):
        # 获取中文字符的每个字符的向量表示的字典
        self.word_embedding, self.word_emb_dim = read_pretrained_emb(emb_path, self.word_alphabet, 'w')

    def build_gaz_alphabet_embedding(self, emb_path):
        # # 获取匹配到的词的向量表示的字典
        self.gaz_alphabet_embedding, self.gaz_alphabet_emb_dim = read_pretrained_emb(emb_path, self.gaz_alphabet, 'g')
