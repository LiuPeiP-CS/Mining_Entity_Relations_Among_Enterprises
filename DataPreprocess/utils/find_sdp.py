#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/10/11 下午4:24
# @Author  : PeiP Liu
# @FileName: find_sdp.py
# @Software: PyCharm

import re
from collections import OrderedDict
import copy

import spacy
from spacy.tokens import Doc
import networkx as nx
from spacy import displacy

nlp = spacy.load("zh_core_web_sm")
# rf https://blog.csdn.net/shenkunchang1877/article/details/109548721
# rf https://blog.csdn.net/qq_41542989/article/details/124242912

def split_by_ents(str_sentence, entities_list):
    """
    :param str_sentence: 原始输入的sentence句子
    :param entities_list: 同sentence_to_token()函数的参数
    :return:
    """

    re_s = entities_list[0]
    for i in range(1, len(entities_list)):
        re_s = re_s + '|' + entities_list[i]
    re_s = re_s.replace(
        "(", "\\(").replace(")", "\\)").replace("+", "\\+").replace("*", "\\*")
    entities_split_list = re.split("("+re_s+")+", str_sentence) # 通过实体对原始语句进行切分，并保留实体的内容和位置。
    print(entities_split_list)
    # ['我非常喜欢华夏文明，', '华为', 'Mate手机很不错，所以我选择了', '华为', '，而不是', '苹果', '和', 'Nokia Pro', '。', '(IBM)', '是一家很不错的企业，我们很欣赏', 'relme+', '的做事风格']

    non_ents = OrderedDict()
    ents = OrderedDict()

    # 区分实体和非实体
    pre_tokenID2entID = OrderedDict()
    ent_iter = 0

    for i, i_seg in enumerate(entities_split_list):
        if i_seg in entities_list:
            ents[i] = [i_seg] # 便于后续列表extend
            pre_tokenID2entID[i] = ent_iter # 解析之前，实体的位置是实体分割的位置
            ent_iter = ent_iter + 1
        else:
            non_ents[i] = i_seg # 便于后续字符串解析

    print(ents)
    print(non_ents)

    # 解析非实体的字符串
    non_ents_parse = OrderedDict()
    for non_ent_i, non_ent_seg in non_ents.items():
        parsed_tokens = []
        doc = nlp(non_ent_seg)
        for each_parsed_token_in_seg in doc:
            parsed_tokens.append(each_parsed_token_in_seg.text)

        non_ents_parse[non_ent_i] = copy.deepcopy(parsed_tokens)

    print(non_ents_parse)

    # 列表串接上所有的字符串
    entID2tokenID = OrderedDict()
    full_parser_token = []
    for i in range(len(entities_split_list)):
        if i in ents:
            entID2tokenID[pre_tokenID2entID[i]] = len(full_parser_token) # 从0计算，正好-1
            assert len(ents[i]) == 1
            full_parser_token.extend(ents[i])
        elif i in non_ents_parse:
            full_parser_token.extend(non_ents_parse[i])

    print(full_parser_token)

    assert ''.join(full_parser_token) == str_sentence, '{}'.format(str_sentence)

    return copy.deepcopy(entID2tokenID), copy.deepcopy(full_parser_token)


def get_nearest_mentions(token_list, so_id_list, rel_name):
    """
    :param token_list: spacy解析后的单词列表
    :param so_id_list: [(psent_id, poent_id), ...]，其中psent_id和poent_id分别是实体在token_list中的位置
    :return: 最优的(sent_id, oent_id)
    """
    doc = Doc(nlp.vocab, words = token_list)
    for name, tool in nlp.pipeline:
        tool(doc)

    edges = []

    for token in doc:
        print(token.text, token.i) # 构建{token.i:token.text的始末位置}的字典

    for token in doc:
        for child in token.children:
            edges.append((token.i, child.i)) # 将此处的token.i, child.i分别换成字典中的始末位置

    graph = nx.Graph(edges)

    dp_lens = [] # 所有可能组合的最短依存长度
    dp_strs = [] # 将所有的依存路径以单词路径的形式体现，主要是为了实现关系单词的匹配
    for each_ent_pair in so_id_list:
        entity1 = each_ent_pair[0]
        entity2 = each_ent_pair[1]
        dp_lens.append(nx.shortest_path_length(graph, source=entity1, target=entity2))

        dp_id = nx.shortest_path(graph, source=entity1, target=entity2)
        dp_strs.append([token_list[token_id] for token_id in dp_id])

    sorted_dp_lens = sorted(enumerate(dp_lens), key=lambda dp_lens: dp_lens[1]) # 对所有组合的最短依存路径长度进行排序，以便找到最优的组合
    dcp_inds = [x[0] for x in sorted_dp_lens] # 排序后最短依存长度的原始index,即为了找出对应哪条路径

    found = False
    for dcp_ind in dcp_inds:
        if rel_name in dp_strs[dcp_ind]: # 如果发现关系在某条路径中，则选择该条路径
            return so_id_list[dcp_ind]

    if not found:
        return so_id_list[dcp_inds[0]] # 如果没发现关系相关的路径，那么只好用所有组合中的最短路径作为匹配



def sentence_to_token(info, entities_list):
    # 根据entities_list提取实体位置，并将sentence转化为token
    token_pattern = re.compile('[0-9]+\\.?[0-9]*|[a-zA-Z]+|[\\s\\S]')
    token = token_pattern.findall(info)
    token_iter = token_pattern.finditer(info)  # 将数字或者连串字母看作一个整体

    re_s = entities_list[0]
    for i in range(1, len(entities_list)):
        re_s = re_s + '|' + entities_list[i]
    re_s = re_s.replace(
        "(", "\\(").replace(")", "\\)").replace("+", "\\+").replace("*", "\\*")
    entities_pattern = re.compile(re_s)
    entities_iter = entities_pattern.finditer(info)  # 构建实体迭代模式，找到所有实体

    token_info = []
    for i in token_iter:
        token_info.append({
            "start": i.start(),
            "end": i.end(),
            "match": i.group()
        })

    # print(token_info)

    entities_info = []
    for i in entities_iter:
        entities_info.append({
            "start": i.start(),
            "end": i.end(),
            "match": i.group()
        })

    # print(entities_info)

    entities = []
    entities_index = {}
    e_index = 0
    label = []

    flag = False
    for i in range(len(token_info)):
        if e_index < len(entities_info):
            if token_info[i]["start"] == entities_info[e_index]["start"]:
                entities.append({
                    "start": i,
                    "end": -1
                })
                e_name = entities_info[e_index]["match"]
                if e_name in entities_index.keys():
                    entities_index[e_name].append(e_index)
                else:
                    entities_index[e_name] = [e_index]
                label.append("B")
                if token_info[i]["end"] != entities_info[e_index]["end"]:
                    flag = True
                else:
                    entities[e_index]["end"] = i + 1
                    e_index += 1
                continue
            if flag:
                label.append("I")
            else:
                label.append("O")
            if token_info[i]["end"] == entities_info[e_index]["end"]:
                entities[e_index]["end"] = i + 1
                e_index += 1
                flag = False
        else:
            label.append("O")

    # print(token)
    print(label)
    print(entities)
    print(entities_index)
    return token, label, entities, entities_index, len(entities)

if __name__ == "__main__":
    str_sent = '我非常喜欢华夏文明，华为Mate手机很不错，所以我选择了华为，而不是苹果和Nokia Pro。(IBM)是一家很不错的企业，我们很欣赏relme+的做事风格'
    ent_list = ['华为', '苹果', 'Nokia Pro','(IBM)', 'relme+']
    sentence_to_token(str_sent, ent_list)
    # split_by_ents(str_sent, ent_list)


# 写作注意两点：英文单词以整体形式出现，英文单词之间的空格当做一个字符
# 数据标注时，我们使用最短依存解析找到多共指实体的关系匹配，比如：



