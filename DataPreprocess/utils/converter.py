import json
from operator import le
# import jsonlines


def convert_to_casrel_and_TPLinker(ori_data_file, save_path_casrel, save_path_tplinker):
    with open(ori_data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 获得rel2id.json
    rel = data["relations_dict"]
    dict1 = {}
    dict2 = {}
    rel_list = list(rel.keys())
    for i in range(len(rel_list)):
        dict1[i] = rel_list[i]
        dict2[rel_list[i]] = i

    rel2id = [dict1, dict2]

    with open(save_path_casrel / "rel2id.json", "w", encoding="utf-8") as f:
        json.dump(rel2id, f, ensure_ascii=False)

    import re
    pattern = re.compile('[a-zA-Z]+|[\s\S]')
    # 获得一般数据
    info = data["data"]

    triple_info = []

    for i in info:
        entities = i["entities"]
        triple_list = []
        for rel in i["relations"]:
            if rel["rel_name"] != "None":
                sub_i = rel["subject"]
                rel_name = rel["rel_name"]
                obj_i = rel["object"]
                sub = "".join(i["token"][entities[sub_i]["start"]:entities[sub_i]["end"]])
                obj = "".join(i["token"][entities[obj_i]["start"]:entities[obj_i]["end"]])
                triple_list.append([
                    sub,
                    rel_name,
                    obj
                ])
        token = i["token"]
        text = " ".join(token)
        
        arm = {
            "text": " " + text + " ",
            "triple_list": triple_list
        }
        triple_info.append(arm)

    with open(save_path_casrel / "all_triple.json", "w", encoding="utf-8") as f:
        json.dump(triple_info, f, ensure_ascii=False)

    len_train = int(len(info) * 0.8)
    len_test = int(len(info)*0.1)

    train_triples = triple_info[0:len_train]
    test_triples = triple_info[len_train:len_train+len_test]
    dev_triples = triple_info[len_train+len_test:]

    with open(save_path_casrel / "train_triples.json", "w", encoding="utf-8") as f:
        json.dump(train_triples, f, ensure_ascii=False)

    with open(save_path_casrel / "test_triples.json", "w", encoding="utf-8") as f:
        json.dump(test_triples, f, ensure_ascii=False)

    with open(save_path_casrel / "dev_triples.json", "w", encoding="utf-8") as f:
        json.dump(dev_triples, f, ensure_ascii=False)

    with open(save_path_tplinker / "train_data.json", 'w', encoding='utf-8') as f:
        json.dump(train_triples, f, ensure_ascii=False)
    
    with open(save_path_tplinker / "valid_data.json", 'w', encoding='utf-8') as f:
        json.dump(dev_triples, f, ensure_ascii=False)
    
    with open(save_path_tplinker / "test_data/test_triples.json", 'w', encoding='utf-8') as f:
        json.dump(test_triples, f, ensure_ascii=False)


def convert_to_spert(ori_data_file, save_path):
    with open(ori_data_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 生成types
    relations = data["relations_dict"]
    types = {
        "entities": {"Org": {"short": "Org", "verbose": "Organization"}},
        "relations": {}
    }

    for rel in relations:
        if rel == "None":
            continue
        types["relations"][rel] = {
            "short": rel,
            "verbose": rel,
            "symmetric": relations[rel]
        }

    with open(save_path / "types.json", "w", encoding="utf-8") as f:
        json.dump(types, f, ensure_ascii=False)

    def get_data(o_data, index = 0):
        res_data = []
        for d in o_data:
            es = []
            for e in d["entities"]:
                es.append({
                    "type": "Org",
                    "start": e["start"],
                    "end": e["end"]
                })
            rs = []
            for r in d["relations"]:
                if r["rel_name"] == "None":
                    continue
                rs.append({
                    "type": r["rel_name"],
                    "head": r["subject"],
                    "tail": r["object"] 
                })
            res_data.append({
                "tokens": d["token"],
                "entities": es,
                "relations": rs,
                "orig_id": index
            })
            index += 1
        return res_data, index

    # 生成train
    train_data_len = int(len(data["data"]) * 0.8)
    ori_data = data["data"][0:train_data_len]

    train_data, index = get_data(ori_data)

    with open(save_path / "train.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False)

    # 生成valid
    valid_data_len = int(len(data["data"]) * 0.1)
    ori_data = data["data"][train_data_len:train_data_len + valid_data_len]

    valid_data, index = get_data(ori_data, index)

    with open(save_path / "valid.json", "w", encoding="utf-8") as f:
        json.dump(valid_data, f, ensure_ascii=False)

    # 生成test
    test_data_len = int(len(data["data"]) * 0.1)
    ori_data = data["data"][train_data_len + valid_data_len:-1]

    test_data, index = get_data(ori_data, index)
    with open(save_path / "test.json", "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False)


def convert_to_tablesequence(ori_data_file, save_path):
    with open(ori_data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    def get_data(o_data):
        res_data = []
        for d in o_data:
            es = []
            for e in d["entities"]:
                es.append([
                    e["start"],
                    e["end"],
                    "Org"
                ])
            rs = []
            for r in d["relations"]:
                if r["rel_name"] == "None":
                    continue
                rs.append([
                    d["entities"][r["subject"]]["start"],
                    d["entities"][r["subject"]]["end"],
                    d["entities"][r["object"]]["start"],
                    d["entities"][r["object"]]["end"],
                    r["rel_name"]
                ])
            res_data.append({
                "tokens": d["token"],
                "entities": es,
                "relations": rs
            })
        return res_data

    # 生成train
    train_data_len = int(len(data["data"]) * 0.8)
    # train_data_len = 600
    ori_data = data["data"][0:train_data_len]

    train_data = get_data(ori_data)

    with open(save_path / "train.exp.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False)

    # 生成valid
    valid_data_len = int(len(data["data"]) * 0.1)
    # valid_data_len = 80
    ori_data = data["data"][train_data_len:train_data_len + valid_data_len]

    valid_data = get_data(ori_data)

    with open(save_path / "valid.exp.json", "w", encoding="utf-8") as f:
        json.dump(valid_data, f, ensure_ascii=False)

    # 生成test
    test_data_len = int(len(data["data"]) * 0.1)
    # test_data_len = 80
    ori_data = data["data"][train_data_len + valid_data_len:train_data_len + valid_data_len+test_data_len]

    test_data = get_data(ori_data)
    with open(save_path / "test.exp.json", "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False)
