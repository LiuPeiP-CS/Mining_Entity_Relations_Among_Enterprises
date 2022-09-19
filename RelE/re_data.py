#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/11/26 上午9:18
# @Author  : PeiP Liu
# @FileName: re_data.py
# @Software: PyCharm

import torch
from collections import defaultdict

"""
我们认为数据应该包含实体和关系，而关系应该包含实体和关系的属性特点
"""

class RelationType:
    def __init__(self, relation_id, relation_name, symmetric=False):
        """
        :param relation_id: the id of relation in the relation dictionary
        :param relation_name: the name of relation
        :param symmetric: whether the relation can be symmetric
        """
        self.rel_id = relation_id  # 指的是该关系在关系字典中的id
        self.rel_name = relation_name  # 指的是该关系的具体类型
        self.symmetric = symmetric  # True or False, 该关系是否对称

    @property
    def get_id(self):
        return self.rel_id

    @property
    def get_symmetric(self):
        return self.symmetric

    @property
    def get_name(self):
        return self.rel_name


class Entity:  # 将每个实体表示成一个对象。
    def __init__(self, span_range, entity_tags, entity_tokens, entity_size):
        """
        :param span_range: the list containing start and end of the span
        :param entity_tags: the tag of the span, which is a list ['B', 'I', 'I', ...]
        :param entity_tokens: a string which is the name of entity
        :param entity_size: a int constant which denotes the size of span
        :return:
        """
        self.range = span_range
        self.ent_tag = entity_tags
        self.ent_tokens = entity_tokens
        self.entity_size = entity_size

    @property
    def get_ent_tokens(self):
        # 获取实体的实际字符串
        return self.ent_tokens

    @property
    def get_ent_range(self):
        # 获取实体在原始句子中的头尾位置元组
        return self.range[0], self.range[1]

    @property
    def get_ent_tags(self):
        # 获取实体的标签
        return self.ent_tag  # 是一个标签列表

    @property
    def get_ent_size(self):
        return self.entity_size


class Relation:
    def __init__(self, head_entity_obj, rel_obj, tail_entity_obj):
        # head_obj和tail_obj都是entity实体类对象

        self.head_ent = head_entity_obj  # Entity类对象,指向头实体
        self.rel_obj = rel_obj  # RelationType类对象
        self.tail_ent = tail_entity_obj  # Entity类对象,指向尾实体

    @property
    def relation_triplet(self):
        head = self.head_ent
        tail = self.tail_ent

        head_start, head_end = head.get_ent_range
        tail_start, tail_end = tail.get_ent_range

        relation_detail = ((head_start, head_end, head.get_ent_tokens),
                           (tail_start, tail_end, tail.get_ent_tokens), self.rel_obj)

        return relation_detail

    @property
    def rel_head(self):
        return self.head_ent

    @property
    def rel_tail(self):
        return self.tail_ent

    @property
    def relation(self):
        return self.rel_obj

    """
    @property
    def relation_name(self):
        return self.rel_obj.get_name  # 获取本条关系的名称
    
    @property
    def reverse(self):
        return self.rel_obj.get_symmetric  # 获取本条关系的可对称属性
    """

class reDataSet:

    def __init__(self):
        # self.ent_dict = defaultdict()  # 构建一个用于存储该条数据所有entity的字典
        # self.relation_dict = defaultdict()  # 构建一个用于存储该条数据所有entity的字典
        self.ent_list = [] # 实体对象的列表，class Entity
        self.rel_list = [] # 关系对象的列表，class Relation

        # self.rel_id = 0
        # self.ent_id = 0  # 记录实体的数量

    def create_entity(self, entity_range, entity_tags, entity_tokens, entity_size):
        # 构造每条数据中的实体
        new_ent_obj = Entity(entity_range, entity_tags, entity_tokens, entity_size)
        # self.ent_dict[self.ent_id] = new_ent_obj
        # self.ent_id = self.ent_id + 1
        self.ent_list.append(new_ent_obj)  # 为了确认该实体在所有实体中的位置，当然我们也可以由entity_range来设置

        return new_ent_obj

    def create_relation(self, head_entity_obj, rel_obj, tail_entity_obj):
        # 构造每条数据中的关系
        new_rel_obj = Relation(head_entity_obj, rel_obj, tail_entity_obj)
        # self.relation_dict[self.rel_id] = new_rel_obj
        # self.rel_id = self.rel_id + 1
        self.rel_list.append(new_rel_obj)  # 为了确认该关系在所有关系中的位置

        return new_rel_obj

    @property
    def all_ents(self):
        return self.ent_list

    @property
    def all_rels(self):
        return self.rel_list