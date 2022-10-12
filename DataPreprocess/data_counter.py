from data_preprocessing.utils import file as uf
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import colors
import random

data = uf.load_json_data("data/data.json")

rels = data["relations_statistic"]
rels_sum = 0
for k in rels.keys():
    if k != "None":
        rels_sum += rels[k]
rels_avg = rels_sum/len(data["data"])
print("sum: {rels_sum};avg: {rels_avg}")

ent_sum = 0
for d in data["data"]:
    ent_sum += d["entities_num"]
ent_avg = ent_sum/len(data["data"])
print("sum: {ent_sum}; avg: {ent_avg}")

sent_max = 0
sent_min = 500
sent_sum = 0
for d in data["data"]:
    if d["length"] > sent_max:
        sent_max = d["length"]
    if d["length"] < sent_min:
        sent_min = d["length"]
    sent_sum += d["length"]
sent_avg = sent_sum/len(data["data"])
print("max: {sent_max}, min: {sent_min}, avg: {sent_avg}")

'''绘图部分'''

matplotlib.rcParams['font.sans-serif'] = ['KaiTi']
matplotlib.rcParams['figure.figsize'] = (5.0, 3.0)

rs = data["relations_statistic"]
x = []
y = []
for k in rs.keys():
    if k != "None":
        x.append(k)
        y.append(rs[k])
c_i = random.sample(range(0, len(colors.cnames.keys())), 18)
bar_colors = []
for i in c_i:
    bar_colors.append(list(colors.cnames.keys())[i])
plt.bar(x, y, color=bar_colors)

plt.show()
