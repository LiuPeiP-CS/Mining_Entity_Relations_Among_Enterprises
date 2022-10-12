import re
from collections import OrderedDict
import copy
from find_sdp import *

'''
文本分析处理部分
'''


def get_ents_list(rel_list):
    relations_list = ["合作", "供应", "参股", "转让", "控股", "附属", "合资", "投资", "授权", "代管", "合并",
                      "剥离", "竞争", "代工", "委托", "更名", "共指", "纠纷", "关联", "None"]
    entities = []
    rels = []
    for relation in rel_list:
        tmp = relation.split("，")
        if len(tmp) == 3:
            rel = {
                "subject": tmp[0],
                "rel_name": tmp[1],
                "object": tmp[2]
            }
            if tmp[1] not in relations_list:
                print(tmp)
        elif len(tmp) == 4:
            rel = {
                "info": tmp[0],
                "subject": tmp[1],
                "rel_name": tmp[2],
                "object": tmp[3]
            }
            if tmp[2] not in relations_list:
                print(tmp)
        else:
            print(tmp)
        rels.append(rel)
    et = {}
    for relation in rels:
        et[relation["subject"]] = {"start": -1, "end": -1}
        et[relation["object"]] = {"start": -1, "end": -1}
    for e in et:
        if e not in entities:
            entities.append(e)

    entities.sort(key=lambda x: len(x))
    return entities


def sentence_to_token(info, entities_list):
    # 根据entities_list提取实体位置，并将sentence转化为token
    token_pattern = re.compile('[0-9]+\\.?[0-9]*|[a-zA-Z]+|[\\s\\S]')
    token = token_pattern.findall(info)
    token_iter = token_pattern.finditer(info) # 将数字或者连串字母看作一个整体

    re_s = entities_list[0]
    for i in range(1, len(entities_list)):
        re_s = re_s + '|' + entities_list[i]
    re_s = re_s.replace(
            "(", "\\(").replace(")", "\\)").replace("+", "\\+").replace("*", "\\*")
    entities_pattern = re.compile(re_s)
    entities_iter = entities_pattern.finditer(info) # 构建实体迭代模式，找到所有实体

    token_info = []
    for i in token_iter:
        token_info.append({
            "start": i.start(),
            "end": i.end(),
            "match": i.group()
        })

    entities_info = []
    for i in entities_iter:
        entities_info.append({
            "start": i.start(),
            "end": i.end(),
            "match": i.group()
        })
    

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

    return token, label, entities, entities_index, len(entities) # entities表示的每个实体的{start,end}, entities_index表示{字符名称,[该实体是entitiesz中的第几个]}

def get_entities(sentence, entities_list):
    # 获得实体列表
    entities = []
    for entity in entities_list:
        if entity in sentence:
            if entity not in entities:
                if entity == "58" or entity == '360':
                    tmp = re.compile('[0-9]+\\.?[0-9]*').findall(sentence)
                    for t in tmp:
                        if t == entity:
                            entities.append(entity)
                else:
                    entities.append(entity)
    entities.sort(key = lambda x: len(x), reverse=True)
    return entities # 将匹配出的实体按照长度，从大到小进行排列


def processing_relations(relations, entities, entities_index, token_list, str_sentence, entities_list): # 返回关系列表，每条关系的形式是：{实体id, 关系名称, 实体id}
    """
    :param relations: 标注的文本数据，利用|拆分后构成列表
    :param entities: 实体列表，每个实体包括其start和end
    :param entities_index: 一个字典，key是该实体的字符串，value是一个列表，指的是该实体在entities中的编号（第几个）
    :param token_list: sentence_to_token解析后的token
    :param str_sentence: 原始的输入字符串句子，同sentence_to_token()的参数
    :return:已经按长度从大到小排列的实体序列，同sentence_to_token()的参数
    """

    dict4ents_be_list = get_ents_be_list(entities, token_list)
    entID2ptokenID, parsed_token_list = split_by_ents(str_sentence, entities_list)
    ptokenID2entID = {v:k for k,v in entID2ptokenID.item()}

    l_relations = []
    for relation in relations:
        tmp = relation.split("，") # 对每个关系元组进行分析
        if len(tmp) == 3: # 3元组
            rel = {
                "subject": tmp[0],
                "rel_name": tmp[1],
                "object": tmp[2]
            }
        else: # 四元组
            rel = {
                "info": tmp[0],
                "subject": tmp[1],
                "rel_name": tmp[2],
                "object": tmp[3]
            }
        l_relations.append(rel)
    full_rel = []
    rel_tt = ["合作", "合资", "合并", "竞争", '共指', '纠纷', '关联']

    for relation in relations: # 遍历该条数据的每条关系信息
        tmp = relation.split("，")
        flag4 = len(tmp) == 4
        if flag4:
            info = tmp[0],
            s_name = tmp[1]
            rel_name = tmp[2]
            o_name = tmp[3]
        else:
            s_name = tmp[0]
            rel_name = tmp[1]
            o_name = tmp[2]
            info = None


        if len(entities_index[s_name]) > 1 or len(entities_index[o_name]) > 1:
            # 找到该两个字符串实体存在关系的是位于全部实体的哪个
            so_id_list = []
            sent_ids_list = entities_index[s_name]  # 找到每个头实体字符串出现的实体位置列表
            oent_ids_list = entities_index[o_name]  # 找到每个尾实体字符串出现的实体位置列表
            for each_sent_id in sent_ids_list:
                for each_oent_id in oent_ids_list:
                    so_id_list.append((entID2ptokenID[each_sent_id], entID2ptokenID[each_oent_id])) # 此处构成元组的是实体列表中的实体id，我们应该将其转化为spacy解析句子后的token_id

            ptoken_sid, ptoken_oid = get_nearest_mentions(parsed_token_list, copy.deepcopy(so_id_list), rel_name)
            s_i, o_i = ptokenID2entID[ptoken_sid], ptokenID2entID[ptoken_oid]

        elif len(entities_index[s_name]) == 1 and len(entities_index[o_name]) == 1:
            s_i = entities_index[s_name][0] # 找出每条关系中每个实体（字符）所处的全部位置（在该条数据的实体列表中的id）
            o_i = entities_index[o_name][0]

        else:
            raise ValueError('The number of subject or object is wrong !')

        # 当A字符串和B字符串存在关系时，我们默认A的所有共指与B的所有共指存在这种关系
        if flag4:
            new_rel = {
                "subject": s_i,
                "rel_name": rel_name,
                "object": o_i,
                "info": info
            }
        else:
            new_rel = {
                "subject": s_i,
                "rel_name": rel_name,
                "object": o_i
            }
        if new_rel not in full_rel:
            full_rel.append(new_rel)
        if rel_name in rel_tt: # 特殊的，对于对称关系，使用了sub和obj的逆转添加
            if flag4:
                new_rel = {
                    "subject": o_i,
                    "rel_name": rel_name,
                    "object": s_i,
                    "info": info
                }
            else:
                new_rel = {
                    "subject": o_i,
                    "rel_name": rel_name,
                    "object": s_i
                }
            if new_rel not in full_rel:
                full_rel.append(new_rel)

    # 建立所有的实体对关系，将不存在事实关系的全部列为None关系
    for s_i in range(len(entities)):
        for o_i in range(len(entities)):
            if s_i == o_i:
                continue
            flag = False
            for rel in full_rel: # 检查每个事实关系，除了事实关系外的都定义为None关系
                if rel["object"] == o_i and rel["subject"] == s_i:
                    flag = True # 标记这两个实体存在事实关系
                    break
            if not flag:
                full_rel.append({
                    "subject": s_i,
                    "rel_name": "None",
                    "object": o_i
                })
    return full_rel


def relation_statistic(relations): # 统计每个文本关系所对应的数量
    rel_stats = {}
    for rel in relations:
        if rel["rel_name"] in rel_stats.keys():
            rel_stats[rel["rel_name"]] += 1
        else:
            rel_stats[rel["rel_name"]] = 1
    return rel_stats


def count_relations(data):
    relations = {}
    for d in data:
        for rel in d["relations_statistic"].keys():
            if rel in relations.keys():
                relations[rel] += d["relations_statistic"][rel]
            else:
                relations[rel] = d["relations_statistic"][rel]
    return relations


def data_to_txt(info): # 主要是用于随机化之后的文本重现
    ts = info["sentence"]
    tr = ""
    ent = []
    ent_index = info["entities"]
    for e_i in ent_index:
        ent.append("".join(info["token"][e_i["start"]:e_i["end"]]))
    relations = info["relations"]
    for rel in relations:
        if rel["rel_name"] != "None":
            if len(rel) == 3:
                tr += ent[rel["subject"]] +'，'+ rel["rel_name"]+'，'+ ent[rel["object"]]+ '|'
            else:
                print('************************************************')
    tr = tr[:-1]
    return ts, tr


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


