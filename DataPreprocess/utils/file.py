import json

'''
文件处理部分
'''


def load_label_data(path):
    # 载入txt数据
    with open(path, "r", encoding="utf-8") as f:
        tmp = f.read().split("\n\n")
        data = []
        for info in tmp:
            info_split = info.split("\n")
            data.append({
                "sentence": info_split[0],
                "relations": info_split[1].split("|")
            })
    return data # 包括文本句子，文本关系的列表


# 加载json数据
def load_json_data(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        return data


# 保存文件到json文件
def save_to_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        data_json = json.dumps(data, ensure_ascii=False, indent=4)
        f.write(data_json)


def save_to_text(data, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write(data)