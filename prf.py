#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 20-4-1 下午5:36
# @Author  : ivy_nie
# @File    : prf.py
# @Software: PyCharm
from test_hmm import *
def text2tuple(path, cut=True, J=False):
    import jieba
    with open(path) as f:
        dic = {}
        i = 0
        for line in f:
            line = line.strip()
            if cut:
                res = line.split()
            else:
                if J:
                    res = jieba.cut(line)
                else:
                    res = hmm.cut(line)
                    print(res)
            dic[i] = []
            num = 0
            for s in res:
                dic[i].append((num, num + len(s) - 1))
                num += len(s)
            i += 1

    return dic


def test(test, gold, J=False):
    dic_test = text2tuple(test, cut=False, J=J)
    dic_gold = text2tuple(gold, J=J)

    linelen = len(dic_test)
    assert len(dic_test) == len(dic_gold)

    num_test = 0
    num_gold = 0
    num_right = 0
    for i in range(linelen):
        seq_test = dic_test[i]
        seq_gold = dic_gold[i]
        num_test += len(seq_test)
        num_gold += len(seq_gold)
        for t in seq_test:
            if t in seq_gold:
                num_right += 1

    P = num_right / num_test
    R = num_right / num_gold
    F1 = P * R / (P + R)
    return P, R, F1


P, R, F1 = test('./msr_test.utf8', './msr_test_gold.utf8')
print("HMM的精确率：", round(P, 3))
print("HMM的召回率：", round(R, 3))
print("HMM的F1值：", round(F1, 3))

P, R, F1 = test('./msr_test.utf8', './msr_test_gold.utf8', J=True)
print("jieba的精确率：", round(P, 3))
print("jieba的召回率：", round(R, 3))
print("jieba的F1值：", round(F1, 3))