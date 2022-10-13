#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/10/13 上午10:23
# @Author  : PeiP Liu
# @FileName: Test.py
# @Software: PyCharm

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


def get_ents_be_list(entities, tokens_list):
    """
    :param entities: # 包含始末位置的实体列表，processing_relations()的输入参数relations
    :param tokens_list: 解析出来的token，即 d["token"]
    :return:
    """

    str_entities = OrderedDict() # 表示每个字符串实体的所有提及的具体始末位置，表现形式为{'ent1':[(b1,e1),(b2,e2),...]}

    # 构建每个字符串实体的位置列表
    for each_ent_be in entities:# data['entities']是生成的表示实体始末的字典
        ent_str = ''.join(tokens_list[each_ent_be['start']:each_ent_be['end']]) # data['token']是字典里的token列表
        if ent_str not in str_entities:
            str_entities[ent_str] = [(each_ent_be['start'], each_ent_be['end'])]
        elif ent_str in str_entities:
            str_entities[ent_str].append((each_ent_be['start'], each_ent_be['end']))

    return copy.deepcopy(str_entities)

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
            if rel_name in ''.join(token_list[s_e:o_b+1]):
                return o_b - s_e, 'T'
            else:
                return o_b - s_e, 'F'

        elif s_b > o_e: # oent在前面，sent在后面
            if rel_name in ''.join(token_list[o_e:s_b+1]):
                return s_b - o_e, 'T'
            else:
                return s_b - o_e, 'F'
        else:
            raise ValueError('The get_nearest_from_addr is wrong !')

    so_dis = [] # 两个实体span之间的具体
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
        if so_rel_sig[short_dis_ind] == 'T': # 从最短距离开始找，如果找到就反馈，结束寻找
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

    print('The number of invalidation length(50000) is {}'.format(dp_lens.count(50000)))
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
                print('There is an appropriate SDP for the relation.')
                print(dp_strs[dcp_ind])
                return 'sdp', so_id_list[dcp_ind]

        if not found: # 存在依存关系，但自始至终没找到相应的关系词，那么我们选择第一个最短距离
            print('We have to choose the shortest SDP for the relation.')
            print(dp_strs[dcp_inds[0]])
            print('\n')
            return 'sdp', so_id_list[dcp_inds[0]] # 如果没发现关系相关的路径，那么只好用所有组合中的最短路径作为匹配


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
    # print(label)
    # print(entities)
    # print(entities_index)
    return token, label, entities, entities_index, len(entities)

if __name__ == "__main__":
    str_sent = '我非常喜欢华夏文明，华为Mate手机很不错，所以我选择了华为，而不是苹果和Nokia Pro。(IBM)是一家很不错的企业，我们很欣赏relme+的做事风格。随着美国对华为的持续打压，华为被迫剥离了荣耀，未来荣耀很有可能以竞争者的身份出现在手机市场。'
    ent_list = ['华为', '苹果', 'Nokia Pro','(IBM)', 'relme+', '荣耀']
    t, l, e, entities_index, length = sentence_to_token(str_sent, ent_list)
    print('The token is {}---\n'.format(t))
    print('The label is {}---\n'.format(l))
    print('The entities is {}---\n'.format(e))
    print('The entities_index is {}---\n'.format(entities_index))
    print('The length is {}---'.format(length))
    dict4ents_be_list = get_ents_be_list(e, t)
    entID2ptokenID, parsed_token_list = split_by_ents(str_sent, ent_list)
    print('The entID2ptokenID is {}---\n'.format(entID2ptokenID))
    print('The parsed_token_list is {}---\n'.format(parsed_token_list))

    ptokenID2entID = {v: k for k, v in entID2ptokenID.items()}

    so_id_list = []
    s_name = '华为'
    o_name = '荣耀'
    rel_name = '竞争'
    sent_ids_list = entities_index[s_name]  # 找到每个头实体字符串出现的实体位置列表
    print('The sent_ids_list is --- {}\n'.format(sent_ids_list))
    oent_ids_list = entities_index[o_name]  # 找到每个尾实体字符串出现的实体位置列表
    print('The oent_ids_list is --- {}\n'.format(oent_ids_list))
    for each_sent_id in sent_ids_list:
        for each_oent_id in oent_ids_list:
            so_id_list.append((entID2ptokenID[each_sent_id],
                               entID2ptokenID[each_oent_id]))  # 此处构成元组的是实体列表中的实体id，我们应该将其转化为spacy解析句子后的token_id

    so_be_list = []
    sent_be_list = dict4ents_be_list[s_name]  # 找到每个头实体字符串出现的token位置元组列表
    print('The sent_be_list is --- {}\n'.format(sent_be_list))
    oent_be_list = dict4ents_be_list[o_name]  # 找到每个尾实体字符串出现的token位置元组列表
    print('The oent_be_list is --- {}\n'.format(oent_be_list))
    for iter_sent, each_sent_be in enumerate(sent_be_list):
        for iter_oent, each_oent_be in enumerate(oent_be_list):
            so_be_list.append((each_sent_be, each_oent_be, iter_sent, iter_oent))

    assert len(sent_ids_list) == len(sent_be_list)
    assert len(oent_ids_list) == len(oent_be_list)

    cmp_sig, best_so = get_nearest_mentions(t, parsed_token_list, copy.deepcopy(so_id_list),
                                                  copy.deepcopy(so_be_list), rel_name)
    sent_id, oent_id = best_so
    if cmp_sig == 'sdp':
        s_i, o_i = ptokenID2entID[sent_id], ptokenID2entID[oent_id]
    elif cmp_sig == 'dis':
        # 使用的是序列相对距离
        s_i, o_i = sent_ids_list[sent_id], oent_ids_list[oent_id]

    print('The chose SDP of subject and object are the {}--th and {}--th ent of all.\n'.format(s_i, o_i))

    s_name_fromIndex = 'None'
    for ent, indexs in entities_index.items():
        if s_i in indexs:
            s_name_fromIndex = ent

    o_name_fromIndex = 'None'
    for ent, indexs in entities_index.items():
        if o_i in indexs:
            o_name_fromIndex = ent

    print('The s_name_fromIndex is {}.'.format(s_name_fromIndex))
    print('The o_name_fromIndex is {}.'.format(o_name_fromIndex))

    assert s_name == s_name_fromIndex, 'Error: the s_name_fromIndex is {}'.format(s_name_fromIndex)
    assert o_name == o_name_fromIndex, 'Error: the o_name_fromIndex is {}'.format(o_name_fromIndex)
    # split_by_ents(str_sent, ent_list)


# 写作注意两点：英文单词以整体形式出现，英文单词之间的空格当做一个字符
# 数据标注时，我们使用最短依存解析找到多共指实体的关系匹配，比如：



