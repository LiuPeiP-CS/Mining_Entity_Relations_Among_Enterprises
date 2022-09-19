#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/11/12 下午7:17
# @Author  : PeiP Liu
# @FileName: BertEvaluate.py
# @Software: PyCharm
import time
import datetime
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
import sys
sys.path.append('../..')
from data_utils import *

def time_format(time_diff):
    seconds = int(round(time_diff))
    return str(datetime.timedelta(seconds=seconds))

def macro_f1(y_true, y_pred):
    p_macro, r_macro, f_macro, support_macro \
      = precision_recall_fscore_support(y_true, y_pred, average='macro')
    f_macro = 2*p_macro*r_macro/(p_macro+r_macro)
    return p_macro, r_macro, f_macro

def bert_evaluate(eval_model, eval_data, label_alphabet, eval_epoch, batch_size, eval_device, eval_data_name):
    eval_model.eval()
    all_pred_labels = []
    all_true_labels = []
    total = 0
    correct = 0
    start = time.time()
    with torch.no_grad():
        for eval_batch in gen_batch(eval_data, batch_size):
            # batch_data = eval_batch.to(eval_device)

            pred_label_ids = eval_model.decoder(eval_batch['token'])
            print(pred_label_ids)
            all_pred_labels.extend(pred_label_ids)
            dev_pred_tensor = torch.tensor(pred_label_ids, dtype=torch.long)

            dev_true_tensor = torch.tensor([label_alphabet.get_index(label) for label in eval_batch['label']],
                                           dtype=torch.long)
            dev_true = dev_true_tensor.tolist()
            print(dev_true)
            all_true_labels.extend(dev_true)

            assert len(all_true_labels) == len(all_pred_labels)

            total = total + len(dev_true)
            assert total == len(all_pred_labels)

            correct = correct + dev_pred_tensor.eq(dev_true_tensor).sum().item()

    assert len(all_true_labels) == len(all_pred_labels)
    average_acc = correct / total
    p, r, f1 = macro_f1(np.array(all_pred_labels), np.array(all_true_labels))
    end = time.time()
    print("This is %s: \n Epoch: %d\n Precision: %.2f\n Recall: %.2f\n F1: %.2f\n Spending: %s"%
          (eval_data_name, eval_epoch, p*100, r*100, f1*100, time_format(end-start)))
    return average_acc, f1



