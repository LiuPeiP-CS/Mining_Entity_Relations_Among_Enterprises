from operator import le
import random

def sentence_replace(sentence, relations, entities, all_entities):
    
    entities.sort(key = lambda x: len(x), reverse=True)
    new_entities = []
    index = random.sample(range(len(all_entities)), len(entities))
    index.sort(reverse=True)
    for i in range(len(entities)):
        if entities[i] in ['巴基斯坦', '阿联酋', '摩洛哥', '土耳其', '阿根廷', '墨西哥', '中国', '巴林', '埃及', '巴西', '泰国', '秘鲁']:
            new_entities.append(entities[i])
            continue
        while all_entities[index[i]] == entities[i] or all_entities[index[i]] in new_entities: # 避免出现原始数据，实现完全不同的随机替换
            index[i] += 1
        sentence = sentence.replace(entities[i], all_entities[index[i]])
        new_rels = []
        for relation in relations:
            tmp = relation.split("，")
            while entities[i] in tmp:
                tmp[tmp.index(entities[i])] = all_entities[index[i]]
            relation = "，".join(tmp)
            new_rels.append(relation)
            
        relations = new_rels
        new_entities.append(all_entities[index[i]])
    
    return sentence, relations, new_entities