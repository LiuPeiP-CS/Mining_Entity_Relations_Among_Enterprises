#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/10/15 上午10:26
# @Author  : PeiP Liu
# @FileName: dcounter&deldul4txt.py
# @Software: PyCharm

import re
import os
import copy

# 该文件主要是为了删除共指延伸带来的重复关系，以及统计SEO、EPO、Normal的占比，

abs_path = '/home/liupei/Papers/IPM/DataProcessing20221012/data/RightLabel'
# orig_path = os.path.join(abs_path, 'data_ok.txt')
# update_path = os.path.join(abs_path, 'data_ok+.txt')

orig_path = os.path.join(abs_path, 'final_annot.txt')
update_path = os.path.join(abs_path, 'final_annot+.txt')

relations_statistic = {
        "合作": 0,
        "供应": 0,
        "参股": 0,
        "转让": 0,
        "控股": 0,
        "附属": 0,
        "合资": 0,
        "投资": 0,
        "授权": 0,
        "代管": 0,
        "合并": 0,
        "剥离": 0,
        "竞争": 0,
        "代工": 0,
        "委托": 0,
        "更名": 0,
        "共指": 0,
        "纠纷": 0,
        "关联": 0,
        "None": 0
    }


def deldup(rels):
    # 将原来的标注数据全部替换成三元组的形式
    print('The original relation is {}'.format(rels))
    sym_rel = ["合作", "合资", "合并", "竞争", '共指', '纠纷', '关联']

    triplet_rels = []

    # 本条数据的关系以及实体的实际获取
    entities = []
    rels_list = rels.split('|')
    for rel in rels_list:
        tmp = rel.split('，')
        flag = len(tmp) == 4
        if flag:
            triplet_rel = (tmp[1], tmp[2], tmp[3])
        else:
            triplet_rel = (tmp[0], tmp[1], tmp[2])
        if triplet_rel not in triplet_rels and triplet_rel[0] != triplet_rel[2]: # 初步去重，去掉头尾实体相同的关系
            triplet_rels.append(triplet_rel)
            # rel_str = rel_str + triplet_rel[0] + '，' + triplet_rel[1]+ '，' + triplet_rel[2] + '|'

        if triplet_rel[0] not in entities: # 构建实体字典值
            entities.append(triplet_rel[0])
        if triplet_rel[2] not in entities:
            entities.append(triplet_rel[2])
    print('The entities of this data is {}'.format(entities))

    # 去重
    print('Before remove the duplication of triplets {}'.format(triplet_rels))
    for rel1_iter, rel1 in enumerate(triplet_rels): # 防止[A,竞争,B]与[B,竞争,A]同时出现
        for rel2_iter in range(rel1_iter + 1, len(triplet_rels)):
            rel2 = triplet_rels[rel2_iter]
            if rel1 != rel2:
                if rel1[1] == rel2[1] and ((rel1[0] == rel2[0] and rel1[2] == rel2[2]) or (rel1[0]==rel2[2] and rel1[2] == rel2[0])):
                    triplet_rels[rel2_iter] = rel1
                    if rel1[1] not in sym_rel:
                        print('There is the wrong relation {} and {} !'.format(rel1, rel2))

    NewTriplet_rels = list(set(triplet_rels))
    print('After remove the duplication of triplets {}'.format(NewTriplet_rels))
    for each_rel_after_deldup in NewTriplet_rels:
        if each_rel_after_deldup[1] in relations_statistic:
            relations_statistic[each_rel_after_deldup[1]] += 1


    # 统计每种实体对出现在数据中的频次
    # 构造实体对
    construted_sb = []
    for ent1_iter, ent1 in enumerate(entities):
        for ent2_iter in range(ent1_iter+1, len(entities)):
            ent2 = entities[ent2_iter]
            assert ent1 != ent2
            construted_sb.append((ent1, ent2))
    NewSB = list(set(construted_sb))

    # 计算每个实体对出现的频次（只保留存在的，即大于0）
    f_epo = 0
    ent_sb_one_time = []  # 只出现过一次的实体对
    for each_ent_sb in NewSB:
        ent_sb_freq = 0
        for each_triplet_rel in NewTriplet_rels:
            if each_triplet_rel[0] == each_triplet_rel[2]:
                raise ValueError('There is the wrong relation!')
            if (each_ent_sb[0] == each_triplet_rel[0] and each_ent_sb[1] == each_triplet_rel[2]) or (each_ent_sb[0] == each_triplet_rel[2] and each_ent_sb[1] == each_triplet_rel[0]): #
                # 说明本条关系数据是由each_ent_sb实体对构造的
                ent_sb_freq += 1
        if ent_sb_freq > 1:
            f_epo+=ent_sb_freq # 直接计算epo
        elif ent_sb_freq == 1:
            ent_sb_one_time.append(each_ent_sb) # spo需要经过实体过滤

    seo_ent_sb_list = []
    for each_ent in entities:
        each_ent_freq = 0
        for each_ent_sb_one in ent_sb_one_time:
            assert each_ent_sb_one[0] != each_ent_sb_one[1]
            if each_ent == each_ent_sb_one[0] or each_ent == each_ent_sb_one[1]: # 如果实体出现了，那么它的频率增加
                each_ent_freq+=1
        if each_ent_freq > 1: # 说明该实体出现了多次，因此需要将其参与的关系实体对找出来
            for each_ent_sb_one in ent_sb_one_time:
                if each_ent == each_ent_sb_one[0] or each_ent == each_ent_sb_one[1]:
                    seo_ent_sb_list.append(each_ent_sb_one)

    f_seo = len(set(seo_ent_sb_list))

    f_normal = len(NewTriplet_rels) - f_epo - f_seo

    rel_str = '|'.join(['，'.join(each_rel) for each_rel in NewTriplet_rels])
    print('The number of all triplet_rels is {}'.format(f_seo+f_epo+f_normal))
    print('The number of epo is {}'.format(f_epo))
    print('The number of seo is {}'.format(f_seo))
    print('The number of normal is {}'.format(f_normal))
    return rel_str, f_seo, f_epo, f_normal, len(entities)

if __name__ == "__main__":
    new_data = ''
    ent_counter = 0
    tmp = None
    with open(orig_path, "r", encoding="utf-8") as f:
        tmp = f.read().split("\n\n")
        seo_all = 0
        epo_all = 0
        normal_all = 0
        for info in tmp:
            print('*****************************************************************************')
            info_split = info.split("\n")
            sentence = info_split[0]
            relations, seo, epo, normal, ent_num4info = deldup(info_split[1])
            new_data = new_data + sentence + '\n' + relations + '\n\n'
            seo_all += seo
            epo_all += epo
            normal_all += normal
            ent_counter += ent_num4info

    with open(update_path, 'w', encoding="utf-8") as wrt:
        wrt.write(new_data[:-1]) # 去掉最后一个换行字符
    print('The SEO--{}, EPO--{}, Normal--{}'.format(seo_all, epo_all, normal_all))
    print('The relations_statistic of all is {}'.format(relations_statistic))
    print('The average number of ent is from {}/{}={}'.format(ent_counter, len(tmp), ent_counter/(1.0*len(tmp))))