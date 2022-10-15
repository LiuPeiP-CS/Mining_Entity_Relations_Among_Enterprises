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
    # 该方法虽然能很好的解决实体对文本的分割，但对于连续实体没有办法
    # re_s = entities_list[0]
    # for i in range(1, len(entities_list)):
    #     re_s = re_s + '|' + entities_list[i]
    #
    # print('The split str is {}'.format(re_s))
    # re_s = re_s.replace(
    #     "(", "\\(").replace(")", "\\)").replace("+", "\\+").replace("*", "\\*")
    # entities_split_list = re.split("("+re_s+")+", str_sentence) # 通过实体对原始语句进行切分，并保留实体的内容和位置。
    # print('The split result by the re_s is {}'.format(entities_split_list))
    # ['我非常喜欢华夏文明，', '华为', 'Mate手机很不错，所以我选择了', '华为', '，而不是', '苹果', '和', 'Nokia Pro', '。', '(IBM)', '是一家很不错的企业，我们很欣赏', 'relme+', '的做事风格']

    # 解决连续实体的分割问题
    re_s = entities_list[0]
    for i in range(1, len(entities_list)):
        re_s = re_s + '|' + entities_list[i]
    re_s = re_s.replace(
            "(", "\\(").replace(")", "\\)").replace("+", "\\+").replace("*", "\\*")
    entities_pattern = re.compile(re_s)
    entities_iter = entities_pattern.finditer(str_sentence) # 构建实体迭代模式，找到所有实体

    entities_be_list = []
    for i in entities_iter:
        entities_be_list.append((i.start(), i.end())) # 找到所有实体的首尾位置

    entities_split_list = []
    new_begin = 0
    for ent_be_iter, each_ent_be in enumerate(entities_be_list): # 以每个实体的始末位置划分字符串
        if str_sentence[new_begin:each_ent_be[0]] != '':
            entities_split_list.append(str_sentence[new_begin:each_ent_be[0]]) # 两实体之间的内容
        entities_split_list.append(str_sentence[each_ent_be[0]:each_ent_be[1]]) # 实体内容
        new_begin = each_ent_be[1]
        if ent_be_iter+1 == len(entities_be_list): # 最后一个实体后的内容
            entities_split_list.append(str_sentence[new_begin:])

    print('The split result by the entities is {}'.format(entities_split_list))

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

    # print(ents)
    # print(non_ents)

    # 解析非实体的字符串
    non_ents_parse = OrderedDict()
    for non_ent_i, non_ent_seg in non_ents.items():
        parsed_tokens = []
        doc = nlp(non_ent_seg)
        for each_parsed_token_in_seg in doc:
            parsed_tokens.append(each_parsed_token_in_seg.text)

        non_ents_parse[non_ent_i] = copy.deepcopy(parsed_tokens)

    # print(non_ents_parse)

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

    print('The parsed tokens list from the original str_sentence is {}'.format(full_parser_token))
    print('The reconstructed str_sentence from parsed tokens is {}\n'.format(''.join(full_parser_token)))

    # assert ''.join(full_parser_token) == str_sentence, '{}'.format(str_sentence)

    return copy.deepcopy(entID2tokenID), copy.deepcopy(full_parser_token)


def get_nearest_from_addr(token_list, so_be_list, rel_name):
    """
    :param token_list: sentence_to_token()解析的token列表
    :param so_be_list: [((sent_b, sent_e), (oent_b, oent_e), iter_sent, iter_oent), ...]，(sent_b, sent_e)是实体sent在原始token_list中的位置
    :param rel_name: 关系的字符串表示
    :return:
    """
    def get_dis(s_be, o_be):
        s_b, s_e = s_be
        o_b, o_e = o_be
        if o_b > s_e: # sent在前面，oent在后面
            if rel_name in ''.join(token_list[s_e-3:o_b+6]): # 扩大关键字的查询范围
                return o_b - s_e, True
            else:
                return o_b - s_e, False

        elif s_b > o_e: # oent在前面，sent在后面
            if rel_name in ''.join(token_list[o_e-3:s_b+6]): # 扩大关键字的查询范围
                return s_b - o_e, True
            else:
                return s_b - o_e, False
        else:
            raise ValueError('The get_nearest_from_addr is wrong !')

    so_dis = [] # 两个实体span之间的距离
    so_rel_sig = [] # 字段内是否有关系字符串，有的话为'T'，没有的话为'F'
    so_ent_id = [] # 两个实体对应的实体标记位置，用于返回值
    for so_be_so_id in so_be_list:
        each_so_dis, each_so_rel_sig = get_dis(so_be_so_id[0], so_be_so_id[1])
        so_dis.append(each_so_dis)
        so_rel_sig.append(each_so_rel_sig)
        so_ent_id.append((so_be_so_id[2], so_be_so_id[3]))

    found_sg = False
    sorted_so_dis = sorted(enumerate(so_dis), key=lambda so_dis: so_dis[1])  # 对所有组合的最短依存路径长度进行排序，以便找到最优的组合
    so_dis_inds = [x[0] for x in sorted_so_dis]  # 排序后最短距离的原始index,即为了找出对应哪条路径

    for short_dis_ind in so_dis_inds:
        if so_rel_sig[short_dis_ind]: # 从最短距离开始找，如果找到就反馈，结束寻找
            return so_ent_id[short_dis_ind] # 找到对应的头尾实体索引

    if not found_sg: # 上述步骤没有反馈，说明没有找到。因此，直接最近的
        return so_ent_id[so_dis_inds[0]]


def get_nearest_mentions(token_list, parsed_token_list, so_id_list, so_be_list, rel_name):
    """
    :param token_list: sentence_to_token()解析的token列表
    :param parsed_token_list: spacy解析后的单词列表
    :param so_be_list: [((sent_b, sent_e), (oent_b,oent_e)), ...]，(sent_b, sent_e)是实体sent在原始token_list中的位置
    :param so_id_list: [(psent_id, poent_id), ...]，其中psent_id和poent_id分别是实体在parsed_token_list中的位置
    :param rel_name: 关系的字符串表示
    :return: 最优的(sent_id, oent_id)
    """
    doc = Doc(nlp.vocab, words = parsed_token_list)
    for name, tool in nlp.pipeline:
        tool(doc)

    edges = []

    for token in doc:
        print(token.text, token.i) # 构建{token.i:token.text的始末位置}的字典

    for token in doc:
        for child in token.children:
            edges.append((token.i, child.i)) # 将此处的token.i, child.i分别换成字典中的始末位置

    # print(displacy.render(doc,style='dep')) # 将该结果拷贝到文件中打开，就是依存图

    graph = nx.Graph(edges)

    dp_lens = [] # 所有可能组合的最短依存长度
    dp_strs = [] # 将所有的依存路径以单词路径的形式体现，主要是为了实现关系单词的匹配
    for each_ent_pair in so_id_list: # 按照输入列表，依次找到最优路径
        entity1 = each_ent_pair[0]
        entity2 = each_ent_pair[1]
        try:
            sdp_len = nx.shortest_path_length(graph, source=entity1, target=entity2)
            dp_id = nx.shortest_path(graph, source=entity1, target=entity2)
            dp_strs.append([parsed_token_list[token_id] for token_id in dp_id])
            print('The SDP-dis between ptoken-{} and ptoken-{} is {} \nand the SDP is {}\n'.format(entity1, entity2, sdp_len, dp_strs[-1]))
        except:
            sdp_len = 50000 # 没有最短路径时，设置最大的长度
            dp_strs.append(None)
            print('The non-SDP are from {} and {}\n'.format(entity1, entity2) )

        dp_lens.append(sdp_len)

    # print('The number of invalidation length(50000) is {}'.format(dp_lens.count(50000)))
    if dp_lens.count(50000) == len(so_id_list): # 没找到任何依存路径
        print('We have to use the dis-computation to get the relation!\n')
        return 'dis', get_nearest_from_addr(token_list, so_be_list, rel_name)
    else:
        # 存在依存路径
        print('We have used the SDP to get the relation!\n')
        sorted_dp_lens = sorted(enumerate(dp_lens), key=lambda dp_lens: dp_lens[1]) # 对所有组合的最短依存路径长度进行排序，以便找到最优的组合
        dcp_inds = [x[0] for x in sorted_dp_lens] # 排序后最短依存长度的原始index,即为了找出对应哪条路径

        found = False
        for dcp_ind in dcp_inds: # 从最小的路径开始，依次递增
            if dp_strs[dcp_ind] and rel_name in dp_strs[dcp_ind]: # 如果发现关系在某条路径中，则选择该条路径
                print('There is an appropriate SDP for the relation.\n')
                print(dp_strs[dcp_ind])
                return 'sdp', so_id_list[dcp_ind]

        if not found: # 存在依存关系，但自始至终没找到相应的关系词，那么我们选择第一个最短距离
            print('We have to choose the shortest SDP for the relation.\n')
            print(dp_strs[dcp_inds[0]])
            print('\n')
            return 'sdp', so_id_list[dcp_inds[0]] # 如果没发现关系相关的路径，那么只好用所有组合中的最短路径作为匹配
