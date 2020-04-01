#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 20-4-1 下午8:12
# @Author  : ivy_nie
# @File    : test_new.py
# @Software: PyCharm

# encoding=utf-8




import os


def data_path(filename):
    return os.path.join(os.path.dirname(__file__), "%s" % filename)

#
# def test_seg():
#     cases = [
#         "给你们传授一点人生的经验",
#         "我来到北京清华大学",
#         "长春市长春节讲话",
#         "我们在野生动物园玩",
#         "我只是做了一些微小的工作",
#         "国庆节我在研究中文分词",
#         "比起生存还是死亡来忠诚与背叛可能更是一个问题"
#     ]
#     for case in cases:
#         result = dict_cut(case)
#         print(result)

#
# def test_tag():
#     cases = [
#         "给你们传授一点人生的经验",
#         "我来到北京清华大学",
#         "长春市长春节讲话",
#         "我们在野生动物园玩",
#         "我只是做了一些微小的工作",
#         "国庆节我在研究中文分词",
#         "比起生存还是死亡来忠诚与背叛可能更是一个问题"
#     ]
#     for case in cases:
#         result = tag(case)
#         print(result)
#

def test_abstract():
    fr = open(data_path('tmp/news.txt'), encoding='utf-8')

    case = ''
    for line in fr:
        line = line.strip()
        if line == '####':
            result = get_abstract(case)
            print(result)
            case = ''
        else:
            case += line


def test_word2vec():
    data = [
        'Merge multiple sorted inputs into a single sorted output',
        'The API below differs from textbook heap algorithms in two aspects'
    ]
    wv = Word2Vec(vec_len=50)
    wv.train(data, model='cbow')
    print(wv['into'])


def test():
    print("test seg:")
    test_seg()
    print("==========")
    print("test tag:")
    test_tag()
    print("==========")
    print("test abstract:")
    test_abstract()
    print("==========")
    print("test word2vec:")
    test_word2vec()
    print("==========")


if __name__ == '__main__':
    test()
