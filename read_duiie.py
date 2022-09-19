# -*- coding: UTF-8 -*-
import json

"""
paths = ['/home/liupei/Papers/IPM/Dataset/du2019/train_data.json',
         '/home/liupei/Papers/IPM/Dataset/du2019/dev_data.json',
         ]
wrt_file = '/home/liupei/Papers/IPM/Dataset/du2019/fusion_du.txt'

data_dict = {}
for file_path in paths:
    print(file_path)
    file_read = open(file_path, 'r', encoding='utf-8')
    for line in file_read.readlines():
        try:
            user_dict = json.loads(line)
            print(user_dict['text'])
            print(user_dict['spo_list'])
            data_dict[user_dict['text']] = user_dict['spo_list']
        except:
            continue

prefined_relation_dataset = ['合作','供应','参股','转让','控股','附属','合资','投资','授权',
                             '代管','合并','剥离','竞争','代工','委托','更名','简称','诉讼',]

for text_data, relations in data_dict.items():
    wrt = open(wrt_file, 'a+', encoding='utf-8')
    rel_text = ''
    rel_num = len(relations)
    for i, relation_dict in enumerate(relations):
        if relation_dict['predicate'] in prefined_relation_dataset:
            sub = relation_dict['subject'].strip()
            rel = relation_dict['predicate'].strip()
            # obj = relation_dict['object']['@value'].strip() # for du_iie_data
            obj = relation_dict['object'].strip()
            rel_text = rel_text+sub+'，'+rel+'，'+obj
            if i+1 != rel_num:
                rel_text = rel_text+'|'
    if rel_text is not '':
        wrt.write(text_data.strip())
        wrt.write('\n')
        wrt.write(rel_text)
        wrt.write('\n\n')
    wrt.close()
"""
prefined_relation_dataset = ['合作','供应','参股','转让','控股','附属','合资','投资','授权',
                             '代管','合并','剥离','竞争','代工','委托','更名','简称','诉讼','关联']
path = "/home/liupei/Papers/IPM/Dataset/final_dataset.txt"
file_read = open(path, 'r', encoding='utf-8')
while file_read:
    wrt_str = ''
    try:
        content_line = file_read.readline().strip()

        triplets_text = next(file_read).strip()
        try:
            triplet_text_list = triplets_text.split('|')
        except:
            triplet_text_list = triplets_text.split('｜')
        for triplet in triplets_list:
            try:
                f_e = triplet.split(',')
            except:
                f_e = triplet.split('，')
            if len(f_e) == 4 and f_e[-2] in prefined_relation_dataset:
                wrt_str = wrt_str + f_e[1] + '，' + f_e[-2] + '，' + f_e[-1] + '|'
            elif len(f_e) == 3 and f_e[-2] in prefined_relation_dataset:
                wrt_str = wrt_str + f_e[1] + '，' + f_e[-2] + '，' + f_e[-1] + '|'
            else:
                continue
        empty_line = next(file_read)

    except:
        continue





