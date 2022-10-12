import os
import argparse
from itertools import count

import data_preprocessing.utils.data as ud
import data_preprocessing.utils.file as uf
# import utils.converter as uc
# import utils.replace as ur

# from tqdm import tqdm
from pathlib import Path


def main(args):
    # load labeled data
    dp = args.data_path
    ep = args.entities_path
    sp = args.save_path
    data = uf.load_label_data(dp)

    # entities_list = uf.load_json_data(ep) # 导入全部的公司实体

    max_len = 0

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

    relations_dict = {
        "合作": True,
        "供应": False,
        "参股": False,
        "转让": False,
        "控股": False,
        "附属": False,
        "合资": True,
        "投资": False,
        "授权": False,
        "代管": False,
        "合并": True,
        "剥离": False,
        "竞争": True,
        "代工": False,
        "委托": False,
        "更名": False,
        "共指": True,
        "纠纷": True,
        "关联": True,
        "None": False
    }

    # count = 0
    txt = ""
    # analyze data
    for d in data:
        # 构建属于本条数据的实体列表
        entities_list = ud.get_ents_list(d["relations"]) # 从关系字符串里面提取出实体
        try:
            try:
                t_entities = ud.get_entities(d["sentence"], entities_list) # 从原始的文本句子中提取出实体列表
                d["token"], d["label"], d["entities"], entities_index, d["entities_num"] = ud.sentence_to_token(d["sentence"], t_entities) # 从原始文本序列中抽取出相应的实体、关系和标签等
                d["relations"] = ud.processing_relations(d["relations"], d["entities"], entities_index, d["token"], d["sentence"], t_entities)
                d["relations_statistic"] = ud.relation_statistic(d["relations"])
                for key in d["relations_statistic"].keys():
                    relations_statistic[key] += d["relations_statistic"][key]
                d["length"] = len(d["token"])
                if max_len < d["length"]:
                    max_len = d["length"]
            except:
                t_entities = ud.get_entities(d["sentence"], entities_list)
                d["token"], d["label"], d["entities"], entities_index, d["entities_num"] = ud.sentence_to_token(d["sentence"], t_entities)
                d["relations"] = ud.processing_relations(d["relations"], d["entities"], entities_index, d["token"], d["sentence"], t_entities)
                d["relations_statistic"] = ud.relation_statistic(d["relations"])
                for key in d["relations_statistic"].keys():
                    relations_statistic[key] += d["relations_statistic"][key]
                d["length"] = len(d["token"])
                if max_len < d["length"]:
                    max_len = d["length"]
        except:
            t_entities = ud.get_entities(d["sentence"], entities_list)
            d["token"], d["label"], d["entities"], entities_index, d["entities_num"] = ud.sentence_to_token(d["sentence"], t_entities)
            d["relations"] = ud.processing_relations(d["relations"], d["entities"], entities_index, d["token"], d["sentence"], t_entities)
            d["relations_statistic"] = ud.relation_statistic(d["relations"])
            for key in d["relations_statistic"].keys():
                relations_statistic[key] += d["relations_statistic"][key]
            d["length"] = len(d["token"])
            if max_len < d["length"]:
                max_len = d["length"]

    attributes = [
        "sentence",
        "token",
        "label",
        "length",
        "entities_num",
        "entities",
        "relations",
        "relations_statistic"
    ]
    result = {
        "attributes": attributes,
        "relations_statistic": relations_statistic,
        "relations_dict": relations_dict,
        "max_len": max_len,
        "data": data
    }

    # print(count)

    # #Save file
    # sp = Path(sp)
    spf = os.path.join(sp, "final_data.json")
    uf.save_to_json(result, spf)
    # uc.convert_to_casrel_and_TPLinker(spf, sp/"CasRel", sp/"TPLinker")
    # uc.convert_to_spert(spf, sp/"spert")
    # uc.convert_to_tablesequence(spf, sp/"Table-sequence")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="preprocessing labeled data")
    parser.add_argument("-p", "--data-path", default="/home/liupei/Papers/IPM/data_preprocessing/data_preprocessing/data/data_ok_07251.txt", type=str, help="Path to labeled data file")
    parser.add_argument("-e", "--entities-path", default="/home/liupei/Papers/IPM/data_preprocessing/data_preprocessing/data/entities.json", type=str, help="Path to entities data file")
    parser.add_argument("-s", "--save-path", type=str, default="/home/liupei/Papers/IPM/data_preprocessing/data_preprocessing/data", help="Output path of preprocessing result file")
    main(parser.parse_args())
