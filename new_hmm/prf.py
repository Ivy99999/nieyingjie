#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 20-4-1 下午5:36
# @Author  : ivy_nie
# @File    : prf.py
# @Software: PyCharm
import re
import jieba
from new_hmm import hmm
def cut_word(sen,tags):
	s=list(sen)
	# print(tags)
	tags=''.join(tags)
	lst=re.finditer('BE{1}|BME{1}|S{1}',tags)
	count=0  # 已插入空格的数量
	for i in lst:
		start=i.span()[0]+count
		count+=1
		end=i.span()[1]+count
		s.insert(start,' ')
		s.insert(end,' ')
		count+=1
	s=''.join(s)
	result=re.split('\s+',s.strip())
	return result
STATES = ['B', 'M', 'E', 'S']
hmm_obj = hmm.HMM(STATES)
hmm_obj.load()
def text2tuple(path, cut=True, J=False):
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
                    # test_tags = obj.predict(test)
                    tag = hmm_obj.predict(line)
                    res=cut_word(line,tag)
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


P, R, F1 = test('msr_test.utf8', 'msr_test_gold.utf8')
print("HMM的精确率：", round(P, 3))
print("HMM的召回率：", round(R, 3))
print("HMM的F1值：", round(F1, 3))

P, R, F1 = test('msr_test.utf8', 'msr_test_gold.utf8', J=True)
print("jieba的精确率：", round(P, 3))
print("jieba的召回率：", round(R, 3))
print("jieba的F1值：", round(F1, 3))