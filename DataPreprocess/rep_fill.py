#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/7/23 上午11:10
# @Author  : PeiP Liu
# @FileName: rep_fill.py
# @Software: PyCharm

import re
import os
import data_preprocessing.utils.data as ud

abs_path = '/home/liupei/Papers/IPM/data_preprocessing/data_preprocessing/data'
orig_path = os.path.join(abs_path, 'rp_data1.txt')
update_path = os.path.join(abs_path, 'rp_data2.txt')

def replace(sent):
    sentence = re.sub(r'100%', "全部", sent) # 替换100%
    sentence = re.sub(r'50%', "大多", sentence) # 替换50%

    # 目标替换小于50%和大于50%
    num_pattern = re.compile('[0-9]+\\.?[0-9]*%')
    num = num_pattern.finditer(sentence)
    medium = 0.50000
    reduct = 1e-7
    for i in num:
        if float(i.group().strip('%')) / 100 - medium >= reduct:
            sentence = re.sub(i.group(), "大多", sentence)
        else:
            sentence = re.sub(i.group(), "少数", sentence)

    # 对分割形式的数字进行统一化处理
    sp_num_pattern = re.compile('[0-9]+[,?[0-9]+]*')
    sp_num = sp_num_pattern.finditer(sentence)
    for i in sp_num:
        sentence = re.sub(i.group(), i.group().replace(',', ''), sentence)
    if sent != sentence:
        print(sent)
        print(sentence)
    return sentence

def fill(rels):
    # 将原来的标注数据全部替换成三元组的形式
    triplet_rels = []

    # 本条数据的关系以及实体的实际获取
    rels_list = rels.split('|')
    for rel in rels_list:
        tmp = rel.split('，')
        flag = len(tmp) == 4
        if flag:
            triplet_rel = [tmp[1], tmp[2], tmp[3]]
        else:
            triplet_rel = [tmp[0], tmp[1], tmp[2]]
        if triplet_rel not in triplet_rels:
            triplet_rels.append(triplet_rel)
            # rel_str = rel_str + triplet_rel[0] + '，' + triplet_rel[1]+ '，' + triplet_rel[2] + '|'

    final_rels = triplet_rels.copy()

    # 多种间接关系的补充，部分补充可选
    for rel1 in triplet_rels:
        for rel2 in triplet_rels:
            if rel1 != rel2:
            # 以下进行多关系的合并
                s_name1 = rel1[0]
                rel_name1 = rel1[1]
                o_name1 = rel1[2]

                s_name2 = rel2[0]
                rel_name2 = rel2[1]
                o_name2 = rel2[2]

                # 共指作为关系的主语
                # A与Ｂ共指，Ｂ与Ｃ存在关系，则Ａ与Ｃ存在同样的关系; AB, BC, AC
                if rel_name1 == '共指' and o_name1 == s_name2:
                    # print(rel1)
                    # print(rel2)
                    final_rels.append([s_name1, rel_name2, o_name2])

                # A与Ｂ共指，A与Ｃ存在关系，则B与Ｃ存在同样的关系; AB, AC, BC
                if rel_name1 == '共指' and s_name1 == s_name2:
                    # print(rel1)
                    # print(rel2)
                    final_rels.append([o_name1, rel_name2, o_name2])

                #　共指作为关系的宾语
                # A与Ｂ存在关系，Ｂ与Ｃ共指，则Ａ与Ｃ存在同样的关系; AB, BC, AC
                if rel_name2 == '共指' and o_name1 == s_name2:
                    # print(rel1)
                    # print(rel2)
                    final_rels.append([s_name1, rel_name1, o_name2])

                # A与C存在关系，B与Ｃ共指，则A与B存在同样的关系; AC, BC, AB
                if rel_name2 == '共指' and o_name1 == o_name2:
                    # print(rel1)
                    # print(rel2)
                    final_rels.append([s_name1, rel_name1, s_name2])


                # print('-------------------------------------------------------------------')

                # √√√ Ａ与Ｂ存在关系，附属>控股>参股>关联
                if s_name1 == s_name2 and o_name1 == o_name2 and rel_name1 == '参股' and rel_name2 == '关联' and rel2 in final_rels:
                    final_rels.remove(rel2)
                    # print(rel2)
                if s_name1 == s_name2 and o_name1 == o_name2 and rel_name1 == '控股' and rel_name2 == '参股' and rel2 in final_rels:
                    final_rels.remove(rel2)
                    # print(rel2)
                if s_name1 == s_name2 and o_name1 == o_name2 and rel_name1 == '控股' and rel_name2 == '关联' and rel2 in final_rels:
                    final_rels.remove(rel2)
                    # print(rel2)
                if s_name1 == o_name2 and o_name1 == s_name2 and rel_name1 == '附属' and rel_name2 == '关联' and rel2 in final_rels:
                    final_rels.remove(rel2)
                    # print(rel2)
                if s_name1 == o_name2 and o_name1 == s_name2 and rel_name1 == '附属' and rel_name2 == '参股' and rel2 in final_rels:
                    final_rels.remove(rel2)
                    # print(rel2)
                if s_name1 == o_name2 and o_name1 == s_name2 and rel_name1 == '附属' and rel_name2 == '控股' and rel2 in final_rels:
                    final_rels.remove(rel2)
                    # print(rel2)

                # 合资>合作>关联
                if s_name1 == s_name2 and o_name1 == o_name2 and rel_name1 == '合作' and rel_name2 == '关联' and rel2 in final_rels:
                    final_rels.remove(rel2)
                    # print(rel2)
                if s_name1 == s_name2 and o_name1 == o_name2 and rel_name1 == '合资' and rel_name2 == '合作' and rel2 in final_rels:
                    final_rels.remove(rel2)
                    # print(rel2)
                if s_name1 == s_name2 and o_name1 == o_name2 and rel_name1 == '合资' and rel_name2 == '关联' and rel2 in final_rels:
                    final_rels.remove(rel2)
                    # print(rel2)
    # 进行去重
    new_final = []
    for each_rel in final_rels:
        if each_rel not in new_final:
            new_final.append(each_rel)

    if triplet_rels != new_final:
        print(triplet_rels)
        print(new_final)
        dvalue = []
        for i in new_final:
            if i not in triplet_rels:
                dvalue.append(i)
        print(dvalue)

    rel_str = '|'.join(['，'.join(each_rel) for each_rel in new_final])
    return rel_str

if __name__ == "__main__":
    new_data = ''
    with open(orig_path, "r", encoding="utf-8") as f:
        tmp = f.read().split("\n\n")
        data = []
        for info in tmp:
            print('*****************************************************************************')
            info_split = info.split("\n")
            sentence = replace(info_split[0])
            relations = fill(info_split[1])
            new_data = new_data + sentence + '\n' + relations + '\n\n'

    with open(update_path, 'w', encoding="utf-8") as wrt:
        wrt.write(new_data[:-1]) # 去掉最后一个换行字符