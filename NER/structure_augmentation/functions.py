#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/11/6 下午8:25
# @Author  : PeiP Liu
# @FileName: functions.py
# @Software: PyCharm

import os
import re
import numpy as np
from transformers import BertTokenizer
import sys
sys.path.append('../..')
from config import root_dir as temp_addr
# from NER.structure_augmentation.alphabet import Alphabet
# from structure_augmentation.alphabet import Alphabet

NULLKEY = '-null-'


def read_instance_with_gaz(data, gaz, word_alphabet, gaz_alphabet, gaz_count, label_alphabet, max_seqlen):
    """
    :param data: the dataset from original file
    :param gaz: the gaz built by gazetteer and trie
    :param word_alphabet: the word dictionary from all train, dev and test
    :param gaz_alphabet: the word dictionary from gaz file
    :param gaz_count: the frequency of matched words for gaz and original dataset, gaz_count[word_id]: frequency
    :param label_alphabet: the label dictionary from all train, dev and test
    :param max_seqlen: the max len of all sequences
    :return:
    """
    # tokenizer = BertTokenizer.from_pretrained(bert_file, do_lower_case=True)
    # instance_texts = []  # 用于记录处理后的所有句子的信息
    instance_ids = []  # 用于记录处理后的所有句子的信息

    for each_data in data: # we deal with each sentence，遍历每个句子信息
        words = each_data['token']
        labels = each_data['label']
        assert len(words) == len(labels)

        word_ids = [word_alphabet.get_index(word) for word in words]
        labels_ids = [label_alphabet.get_index(label) for label in labels]

        if ((max_seqlen < 0) or (len(words) < max_seqlen)) and (len(words) > 0):
            # gaz_ids = []
            sentence_gazmasks = []
            sent_len = len(words)

            # 将每个匹配出来的词在gaz_alphabet中的id，表示到待匹配句子具体的token上
            gazs = [[[] for BMES in range(4)] for _ in range(sent_len)] # each token from the sentence has 4 matched set
            # 将每个匹配出来的词相应的frequency，表示到具体的token上
            gazs_count = [[[] for BMES in range(4)] for _ in range(sent_len)] # count the ele num of each matched set

            max_gazlist = 0  # 获取本条数据中B/M/E/S匹配集合中的词的最大数量值

            for word_iter in range(sent_len): # 对句子中的每个词进行遍历匹配
                # 匹配到的词列表
                matched_list = gaz.enumerateMatchList(words[word_iter:]) # match begin with words[word_iter] from gaz
                # 匹配到的词的长度的列表
                matched_length = [len(a) for a in matched_list] # the length of each matched token
                # 匹配到的词在gaz字典中的索引列表
                matched_id = [gaz_alphabet.get_index(each_matched_word) for each_matched_word in matched_list] # the ids of matched words in gaz alphabet

                for wm_iter in range(len(matched_id)):  # 遍历每个匹配出来的词
                    # gaz_word = matched_list[wm_iter]  # each matched word,如果进行字符匹配需要使用

                    # 以下对每个匹配出来的词进行归类，B,M,E,S
                    if matched_length[wm_iter] == 1:  # it means the single token matched, ##S
                        gazs[word_iter][3].append(matched_id[wm_iter])
                        gazs_count[word_iter][3].append(1)
                    else:
                        w_matched_length = matched_length[wm_iter]  # the matched word length, which is distance for the BMS

                        gazs[word_iter][0].append(matched_id[wm_iter])  # ##B
                        gazs_count[word_iter][0].append(gaz_count[matched_id[wm_iter]])

                        gazs[word_iter+w_matched_length-1][2].append(matched_id[wm_iter])  # ##E
                        gazs_count[word_iter+w_matched_length-1][2].append(gaz_count[matched_id[wm_iter]])

                        for w_matched_len_iter in range(1, w_matched_length-1):
                            gazs[word_iter + w_matched_len_iter][1].append(matched_id[wm_iter])  # ##M
                            # print('The matched_id is {}'.format(matched_id[wm_iter]))
                            # print('The gaz_count is {}'.format(gaz_count[matched_id[wm_iter]]))
                            gazs_count[word_iter + w_matched_len_iter][1].append(gaz_count[matched_id[wm_iter]])

                for bmes_iter in range(4):
                    if not gazs[word_iter][bmes_iter]:  # BMES中的某个匹配集合为空，即没有任何匹配到的gaz中的词
                        gazs[word_iter][bmes_iter].append(0)  # 注意，我们应该确定None的index为0
                        gazs_count[word_iter][bmes_iter].append(1)

                    max_gazlist = max(len(gazs[word_iter][bmes_iter]), max_gazlist) # 获取BMES集合中的最大数量

                """
                if matched_id:  # 当从gaz中能匹配出words[word_iter:]的相关词汇时
                    gaz_ids.append([matched_id, matched_length])  # 将匹配出的词的id和长度等信息进行记录
                else:
                    gaz_ids.append([])  # 如果没有匹配出任何的词汇，那么就记空
                """

            # 如果想要批处理，需要从此处开始.
            for word_iter in range(sent_len):  # 我们只处理一个句子的相关信息
                word_gazmask = []  # 每个词位置的BMES的mask信息（是否存在有效匹配）

                for bmes_iter in range(4):
                    label_matched_set_len = len(gazs[word_iter][bmes_iter])  # 获取每个token的在每个标签上的word匹配数量

                    count_set = set(gazs_count[word_iter][bmes_iter])  # 当前词在B/M/E/S位置上的匹配出的词的frequency，取set
                    if len(count_set) == 1 and 0 in count_set:  # 没匹配出任何的gaz中的词汇
                        gazs_count[word_iter][bmes_iter] = [1] * label_matched_set_len

                    # 对每个标签B/M/E/S进行mask
                    mask = label_matched_set_len * [0]  # 有效位置处的mask值为0
                    mask = mask + (max_gazlist - label_matched_set_len) * [1]  # 其余位置补充为1

                    gazs[word_iter][bmes_iter] = gazs[word_iter][bmes_iter] + (max_gazlist - label_matched_set_len) * [0]  # padding
                    gazs_count[word_iter][bmes_iter]= gazs_count[word_iter][bmes_iter] + (max_gazlist - label_matched_set_len) * [0]

                    word_gazmask.append(mask)  # [[0,0,1],[0,1,1],[0,0,0],[0,0,1]],0是存在有效匹配，1是没有匹配的padding

                sentence_gazmasks.append(word_gazmask)

            # bert_text = ['[CLS]'] + words + ['[SEP]']
            # bert_text_ids = tokenizer.convert_tokens_to_ids(bert_text)

            # instance_texts.append([words, gazs, labels]) # 句子及对应的标签，以及从gazetteer中匹配出的词的id
            # instance_ids.append([word_ids, gaz_ids, labels_ids, gazs, gazs_count, sentence_gazmasks, bert_text_ids])
            instance_ids.append([words, word_ids, labels, labels_ids, gazs, gazs_count, sentence_gazmasks, each_data])

    # return instance_texts, instance_ids
    return instance_ids


def load_embedding(file_path):
    embedding_dim = 200
    file_read = open(file_path, 'r', encoding='utf-8')
    emb_dict = dict()
    for line in file_read:
        line = line.strip()
        if line != '': # 存在实际信息
            line = line.split()
            if len(line) == 201: # 排除其他内容，构建可用的字典
                word = line[0]
                embedding = [float(x) for x in line[1:]]
                emb_dict[word] = np.array(embedding)
    return emb_dict, embedding_dim


def emb_norm(embedidng):
    root_sum_square = np.sqrt(np.sum(np.square(embedidng)))
    return embedidng / root_sum_square

"""
def read_pretrained_emb(emb_path, alphabet, signal): # 首次运行时，需要执行成功；第二次以后可以选用后面的同名函数来提高速度。
    emb_dict, emb_dim = load_embedding(emb_path)
    scale = np.sqrt(3.0 / emb_dim)
    emb_table = np.empty([alphabet.size(), emb_dim], dtype=np.float32)
    # emb_table[:2, :] = np.random.uniform(-scale, scale, [2, emb_dim])  # for UNK and PAD
    emb_table[0, :] = np.random.uniform(-scale, scale, [1, emb_dim])  # for UNK
    for instance, index in alphabet.items():
        if instance in emb_dict:
            instance_emb = emb_norm(emb_dict[instance])
        else:
            instance_emb = np.random.uniform(-scale, scale, [1, emb_dim])  # 对于不在向量字典里的向量，进行随机化
        emb_table[index, :] = instance_emb
    if signal == 'w':
        np.save(os.path.join(temp_addr, 'Result/word_alphabet_emb.npy'), emb_table)
        print('************The Word Alphabet Representation Is Saved!***********')
        return emb_table, emb_dim
    elif signal == 'g':
        np.save(os.path.join(temp_addr, 'Result/gaz_alphabet_emb.npy'), emb_table)
        print('************The Gaz Alphabet Representation Is Saved!***********')
        return emb_table, emb_dim
"""

def read_pretrained_emb(emb_path, alphabet, signal):
    if signal == 'w':
        print('************The Word Alphabet Representation Is Loading!***********')
        return np.load(os.path.join(temp_addr, 'Result/word_alphabet_emb.npy')), 200
    elif signal == 'g':
        print('************The Gaz Alphabet Representation Is Loading!***********')
        return np.load(os.path.join(temp_addr, 'Result/gaz_alphabet_emb.npy')), 200