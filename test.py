#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/12/2 上午11:43
# @Author  : PeiP Liu
# @FileName: test.py
# @Software: PyCharm
def extract_ents(best_path):
    """
    :param best_path: 在test阶段，ner_test输出的标注序列
    :return: 预测输出的实体构造的列表，每个实体包含了其其实位置

    best_path:
    """
    ent_start = 0
    ent_list = []
    seq_len = len(best_path)
    while ent_start < seq_len:
        while ent_start < seq_len and best_path[ent_start] == 'O':
            # 保证开始不能是'O'
            ent_start = ent_start + 1
        ent_end = ent_start + 1
        while ent_end < seq_len and best_path[ent_end] == 'I':
            # 维持中间位置都是'I'
            ent_end = ent_end + 1
        if ent_start != seq_len:
            ent_list.append((ent_start, ent_end))
        ent_start = ent_end

    return ent_list

test_str = 'BOBBIIIOOIBOOO'
print(extract_ents(test_str))
