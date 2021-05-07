# -*- coding: utf-8 -*-
"""
Created on Thu May  6 15:05:45 2021
@author: Vraj
"""
import torch
import torch.utils.data
from torch.autograd import Variable

import os
import pickle
import time
import math
import numpy as np
from preprocess.make_vocab import Dict
import utils

from preprocess import make_vocab
from seq2seq import seq2seq
from optims import Optim

path = r"../Data"

config_dict = {
        "model_name" : "seq2seq",
        "optim" : "adam",
        "learning_rate" : 0.0003,
        "max_grad_norm" : 10,
        "learning_rate_decay" : 0.5,
        "start_decay_at" : 5,
        "epoch": 10,
        "batch_size": 64,
        "num_label": 5,
        "cell": 'lstm',
        "attention": 'None',
        "emb_size": 256,
        "hidden_size": 256,
        "dec_num_layers": 1,
        "enc_num_layers": 1,
        "bidirectional": True,
        "dropout": 0.0,
        "max_time_step": 50,
        "eval_interval": 1000,
        "save_interval": 3000,
        "metrics": ['rouge'],
        "shared_vocab": True,
        "beam_size": 1,
        "unk": True,
        "use_cuda" : False,
        "max_split" : 0
    }

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

config_dict = AttrDict(config_dict)

def load_data():
    print('Loading Data...\n')
    
    pickle_data = pickle.load(open(path + r'/data.pkl', 'rb'))
    pickle_data['train']['length'] = int(pickle_data['train']['length'])
        
    trainset = utils.LabelDataset(pickle_data['train'])
    validset = utils.LabelDataset(pickle_data['test'])

    src_vocab = pickle_data['dict']['src']
    tgt_vocab = pickle_data['dict']['tgt']

    global config_dict
    config_dict["src_vocab_size"]  = src_vocab.size()
    config_dict["tgt_vocab_size"]  = tgt_vocab.size()
    

    trainloader = torch.utils.data.DataLoader(dataset = trainset,
                                              batch_size = 64,
                                              shuffle = True,
                                              num_workers = 0,
                                              collate_fn = utils.label_padding)
    
    
    validloader = torch.utils.data.DataLoader(dataset = validset,
                                              batch_size = 64,
                                              shuffle = False,
                                              num_workers = 0,
                                              collate_fn = utils.label_padding)
    
    res = {
            'trainset': trainset, 
            'validset': validset,
            'trainloader': trainloader, 
            'validloader': validloader,
            'src_vocab': src_vocab, 
            'tgt_vocab': tgt_vocab
    }
    
    return res

def build_model(config_dict):
    print('Building Model...\n')
    model = seq2seq(config_dict)
    optim = Optim(config_dict.optim, config_dict.learning_rate, config_dict.max_grad_norm,
                             lr_decay = config_dict.learning_rate_decay, start_decay_at = config_dict.start_decay_at)
    
    optim.set_parameters(model.parameters())
    
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
        
    return model, optim

def train_model(model, datas, optim, epoch, params):
    model.train()
    trainloader = datas['trainloader']

    for src, tgt, label, src_len, tgt_len, original_src, original_tgt in trainloader:
        model.zero_grad()

        src = Variable(src)
        tgt = Variable(tgt)
        label = Variable(label)
        
        src_len = Variable(src_len)
        
        if config_dict.use_cuda:
            src = src.cuda()
            tgt = tgt.cuda()
            label = label.cuda()
            src_len = src_len.cuda()
        
        lengths, indices = torch.sort(src_len, dim=0, descending=True)
        src = torch.index_select(src, dim=0, index=indices)
        tgt = torch.index_select(tgt, dim=0, index=indices)
        label = torch.index_select(label, dim=0, index=indices)
        dec = tgt[:, :-1]
        targets = tgt[:, 1:]

        try:
            loss, outputs = model(src, lengths, dec, targets)

            if outputs is not None:
                pred = outputs.max(2)[1]
                targets = targets.t()
                num_correct = pred.data.eq(targets.data).masked_select(targets.ne(make_vocab.PAD).data).sum()
                num_total = targets.ne(make_vocab.PAD).data.sum()
            else:
                num_correct = 0
                num_total = 1

            loss.backward()
            optim.step()

            params['report_loss'] += loss.data
            params['report_correct'] += num_correct
            params['report_total'] += num_total

        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory')
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                raise e

        params['updates'] += 1

        if params['updates'] % config_dict.eval_interval == 0:
            score = eval_model(model, datas, params)

            if type(score) == 'dict':
                for metric in config_dict.metrics:
                    params[metric].append(score[metric])

            model.train()
            params['report_loss'], params['report_time'] = 0, time.time()
            params['report_correct'], params['report_total'] = 0, 0

    optim.updateLearningRate(score=0, epoch=epoch)

def eval_model(model, datas, params):
    model.eval()
    reference, candidate, source, alignments = [], [], [], []
    correct_2, correct_5 = 0, 0
    count, total_count = 0, len(datas['validset'])
    validloader = datas['validloader']
    tgt_vocab = datas['tgt_vocab']

    for src, tgt, label, src_len, tgt_len, original_src, original_tgt in validloader:

        src = Variable(src, volatile=True)
        src_len = Variable(src_len, volatile=True)
        label = Variable(label, volatile=True)
        if config_dict.use_cuda:
            src = src.cuda()
            src_len = src_len.cuda()
            label = label.cuda()

        if config_dict.beam_size > 1:
            samples, alignment, c_5, c_2 = model.beam_sample(src, src_len, label, beam_size = config_dict.beam_size)
        else:
            samples, alignment, c_5, c_2 = model.sample(src, src_len, label)

        if samples is not None:
            candidate += [tgt_vocab.convertToLabels(s, make_vocab.EOS) for s in samples]
            source += original_src
            reference += original_tgt

        if alignment is not None:
            alignments += [align for align in alignment]

        count += len(original_src)
        correct_2 += c_2
        correct_5 += c_5

    if config_dict.unk and config_dict.attention != 'None' and len(candidate) > 0:
        cands = []
        for s, c, align in zip(source, candidate, alignments):
            cand = []
            for word, idx in zip(c, align):
                if word == make_vocab.UNK_WORD and idx < len(s):
                    try:
                        cand.append(s[idx])
                    except:
                        cand.append(word)
                        print("%d %d\n" % (len(s), idx))
                else:
                    cand.append(word)
            cands.append(cand)
        candidate = cands

    accuracy_five = correct_5 * 100.0 / total_count
    accuracy_two = correct_2 * 100.0 / total_count
    print("acc = %.2f, %.2f" % (accuracy_five, accuracy_two))

data = load_data()
model, optim = build_model(config_dict)

params = {
        'updates': 0, 
        'report_loss': 0, 
        'report_total': 0,
        'report_correct': 0, 
        'report_time': time.time()
    }
        
for metric in config_dict.metrics:
    params[metric] = []

for i in range(1, config_dict.epoch + 1):
    train_model(model, data, optim, i, params)
    
for metric in config_dict.metrics:
    print("Best %s score: %.2f\n" % (metric, max(params[metric])))
















