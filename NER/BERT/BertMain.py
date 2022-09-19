#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/11/12 下午7:16
# @Author  : PeiP Liu
# @FileName: BertMain.py
# @Software: PyCharm
import os
import torch
import time
import datetime
import numpy as np
from tqdm import trange
import sys
sys.path.append('../..')
from config import BasicArgs
from NER.BERT.BertModel import Bert_CRF
from NER.BERT.BertData import BertData
from NER.BERT.BertEvaluate import bert_evaluate
from data_utils import *
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score


if __name__ =='__main__':
    Bert_output_dir = BasicArgs.Bert_saved_path
    device = BasicArgs.device
    batch_size = BasicArgs.batch_size
    data = BasicArgs.data # data是一个列表，列表里面的每个元素是一个字典，代表了一条数据

    Bert_data = BertData()
    Bert_data.build_alphabet(data)
    label_alphabet = Bert_data.label_alphabet

    # train_sentences = [each_data['token'] for each_data in BasicArgs.train_data]
    # train_labels = [each_data['label'] for each_data in BasicArgs.train_data]
    train_data = BasicArgs.train_data

    # dev_sentences = [each_data['token'] for each_data in BasicArgs.dev_data]
    # dev_labels = [each_data['label'] for each_data in BasicArgs.dev_data]
    dev_data = BasicArgs.dev_data

    # test_sentences = [each_data['token'] for each_data in BasicArgs.test_data]
    # test_labels = [each_data['label'] for each_data in BasicArgs.test_data]
    test_data = BasicArgs.test_data

    learning_rate = BasicArgs.learning_rate
    weight_decay_crf_fc = BasicArgs.weight_decay_crf_fc
    weight_decay_finetune = BasicArgs.weight_decay_finetune
    lr_crf_fc = BasicArgs.lr_crf_fc
    train_epoch = BasicArgs.total_epoch
    warmup_proportion = BasicArgs.warmup_proportion
    gradient_accumulation_steps = BasicArgs.gradient_accumulation_steps

    model = Bert_CRF(BasicArgs.BertPath, label_alphabet, device).to(device)

    params_list = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in params_list if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay_finetune},
        {'params': [p for n, p in params_list if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    start_epoch = 0
    dev_acc_pre = 0
    dev_f1_pre = 0
    if not os.path.exists(Bert_output_dir + 'bert_crf_ner.checkpoint.pt'):
        os.mknod(Bert_output_dir + 'bert_crf_ner.checkpoint.pt')

    # the total_train_steps referring to the loss computing
    total_train_steps = int(len(train_data)//batch_size//gradient_accumulation_steps)*train_epoch
    warmup_steps = int(warmup_proportion*total_train_steps)
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-8)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_train_steps)

    train_average_loss = []
    dev_acc_score = []
    dev_f1_score = []
    for epoch in trange(start_epoch, train_epoch, desc='Epoch'):
        train_loss = 0
        train_start = time.time()
        model.train()
        # clear the gradient
        model.zero_grad()
        batch_start = time.time()

        for step, batch in enumerate(gen_batch(train_data, batch_size)): # the batch is a dict
            # we show the time cost ten by ten batches
            if step % 100 == 0 and step != 0:
                print('100 batches cost time : {}'.format(time_format(time.time()-batch_start)))
                batch_start = time.time()
            # print(batch)
            # batch_data = batch.to(device)
            object_loss = model.neg_log_likehood(batch['token'], batch['label'])

            if gradient_accumulation_steps > 1:
                object_loss = object_loss / gradient_accumulation_steps

            object_loss.backward()
            train_loss+= object_loss.cpu().item()

            if (step+1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1.0, norm_type=2)
                optimizer.step()
                lr_scheduler.step()
                model.zero_grad()

            if step % 100 == 0 and step != 0:
                print("Epoch:{}-{}, Object-loss:{}".format(epoch, step, object_loss))
        ave_loss = train_loss / len(train_data)
        train_average_loss.append(ave_loss)

        print("Epoch: {} is completed, the average loss is: {}, spend: {}".format(epoch, ave_loss, time_format(time.time()-train_start)))
        print("***********************Let us begin the validation of epoch {}******************************".format(epoch))

        dev_acc, dev_f1 = bert_evaluate(model, dev_data, label_alphabet, epoch, batch_size, device, 'dev')
        dev_acc_score.append(dev_acc)
        dev_f1_score.append(dev_f1)

        if dev_f1 > dev_f1_pre:
            torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'dev_acc': dev_acc,
                        'dev_f1': dev_f1},
                       os.path.join(Bert_output_dir + 'bert_crf_ner.checkpoint.pt'))
            dev_f1_pre = dev_f1

    print("**********************************************\n"
          "********     The training is over.    ********\n"
          "**********************************************")

    checkpoint = torch.load(Bert_output_dir + 'bert_crf_ner.checkpoint.pt', map_location='cpu')
    # parser the model params
    epoch = checkpoint['epoch']
    dev_f1_prev = checkpoint['dev_f1']
    dev_acc_prev = checkpoint['dev_acc']
    pretrained_model_dict = checkpoint['model_state']
    # get the model param names
    model_state_dict = model.state_dict()
    # get the params interacting between model_state_dict and pretrained_model_dict
    selected_model_state = {k: v for k, v in pretrained_model_dict.items() if k in model_state_dict}
    model_state_dict.update(selected_model_state)
    # load the params into model
    model.load_state_dict(model_state_dict)
    # show the details about loaded model
    print('Loaded the pretrained BERT_CRF model, epoch:', checkpoint['epoch'],
          'dev_acc:', checkpoint['dev_acc'], 'dev_f1:',checkpoint['dev_f1'])
    model.to(device)
    test_acc, test_f1 = bert_evaluate(model, test_data, label_alphabet, epoch, batch_size, device, 'test')