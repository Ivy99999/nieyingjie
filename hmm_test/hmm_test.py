#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 20-4-2 上午9:35
# @Author  : ivy_nie
# @File    : hmm_test.py
# @Software: PyCharm
import re

from hmm_model import *

STATES = ['B', 'M', 'E', 'S']
obj=HMM(STATES)
obj.get_train_data('data/train.txt')
def add_tags(word):
    if len(word)==1:
        tags=['S']
    elif len(word)==2:
        tags=['B','E']
    else:
        tags =['M']*len(word)
        tags[0] = 'B'
        tags[-1] = 'S'
    char_lst=[]
    for i  in word:
        char_lst.append(i)
    return tags,char_lst

for line in obj.train_data:
    if len(line)==0:
        continue
    word_list=re.split('\s+',line.strip())
    observes = []
    states = []
    for w in word_list:
        tags,char_lst=add_tags(w)
        observes.extend(char_lst)
        states.extend(tags)
    obj.train(observes, states)
obj.count2prob()