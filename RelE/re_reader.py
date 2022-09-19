#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/11/26 下午7:59
# @Author  : PeiP Liu
# @FileName: re_reader.py
# @Software: PyCharm

import torch
import torch.nn as nn
from collections import defaultdict
import sys
sys.path.append('..')
from RelE.re_data import RelationType, Entity, Relation, reDataSet


class InputReader:
    def __init__(self, config): # 输入是config.relation_dict，#注意更换原始数据中的relation_dict的组成形式 "合作":True

        self.rel_type_dict = defaultdict()
        self.index2rel_type = defaultdict()

        for i, (key, value) in enumerate(config.relation_dict.items()):
            rel_type_obj = RelationType(i, key, value)  # 第几个关系，关系的名称，关系是否可对称
            self.rel_type_dict[key] = rel_type_obj  # 关系名称作为key
            self.index2rel_type[i] = rel_type_obj

    def read(self, instance_data): # 输入每次都是一条数据，即instance[index][-1]

        dataset = reDataSet()  # 对每条数据构建一个对象

        entities = instance_data['entities']
        tokens = instance_data['token']
        relations = instance_data['relations']

        entity_obj_list = self.parser_entities(entities, tokens, dataset)  # 将所有的实体增加到dataset中
        relation_obj_list = self.parser_relations(relations, entity_obj_list, dataset)  # 将所有的关系添加到dataset中

        return dataset

    def parser_entities(self, entities, tokens, dataset):  # 输入是个字典，即原始数据中entities的values
        entity_obj_list = []

        for entity_range in entities:
            start, end = entity_range['start'], entity_range['end']
            entity_tags = ['B'] + ['I'] * (end-start-1)  # 增加实体的标签信息,如果是别的应用，应该更改标签内容和方式：['B-'+entity_type]
            entity_tokens = ' '.join(tokens[start:end])  # 将字符串起来得到实体名称
            entity_size = end-start
            entity_obj = dataset.create_entity([start, end], entity_tags, entity_tokens, entity_size)  # 实体对象包含的内容

            entity_obj_list.append(entity_obj)

        return entity_obj_list  # 每条数据中所有实体所构成的对象

    def parser_relations(self, relations, entity_objs, dataset):  # 输入是个列表，即原始数据中relations的values
        relation_obj_list = []

        for relation in relations:
            # rel_infor = relation['info']  # 该内容是关系的面向对象，此论文中暂不使用
            rel_obj = self.rel_type_dict[relation['rel_name']]  # 获取关系类对象
            head_entity_obj = entity_objs[relation['subject']]  # 获取实体类对象
            tail_entity_obj = entity_objs[relation['object']]  # 获取实体类对象

            relaion_obj = dataset.create_relation(head_entity_obj, rel_obj, tail_entity_obj)  # 关系对象包含的内容

            relation_obj_list.append(relaion_obj)

        return relation_obj_list
