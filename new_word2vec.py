#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 20-4-7 上午10:34
# @Author  : ivy_nie
# @File    : new_word2vec.py
# @Software: PyCharm
import random

import numpy as np
word_list_all=[]
with open('test_new.txt','r',encoding='utf-8') as f:
    for lines in f.readlines():
        print(lines)
        word_list = lines.strip().split('  ')
        # print(word_list)
        for idx, word in enumerate(word_list):
            word_list[idx] = word_list[idx]
            word_list_all.append(word_list[idx])
print(len(word_list_all))

import collections
vocabulary_size = 200000
count = [['UNK', -1]]
count.extend(collections.Counter(word_list_all).most_common(vocabulary_size - 1))
dictionary = dict()

for word, _ in count:
    dictionary[word] = len(dictionary)
data = list()
unk_count = 0
for word in word_list_all:
    if word in dictionary:
        index = dictionary[word]
    else:
        index = 0  # dictionary['UNK']
        unk_count = unk_count + 1
    data.append(index)

count[0][1] = unk_count
reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
del word_list_all

data_index = 0

def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buf = collections.deque(maxlen=span)
    for _ in range(span):
        buf.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [ skip_window ]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buf[skip_window]
            labels[i * num_skips + j, 0] = buf[target]
        buf.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels


import tensorflow as tf
import collections
import numpy as np

batch_size = 128
embedding_size = 128  # 生成向量维度.
skip_window = 2  # 左右窗口.
num_skips = 2  # 同一个keyword产生label的次数.
num_sampled = 64  # 负样本抽样数.

graph = tf.Graph()

with graph.as_default(), tf.device('/cpu:0'):
    train_dataset = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    softmax_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / np.sqrt(embedding_size)))
    softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))

    embed = tf.nn.embedding_lookup(embeddings, train_dataset)
    loss = tf.reduce_mean(
        tf.nn.sampled_softmax_loss(weights=softmax_weights, biases=softmax_biases, inputs=embed,
                                   labels=train_labels, num_sampled=num_sampled, num_classes=vocabulary_size))

    optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)

    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
num_steps = 500001
import random
with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    average_loss = 0
    for step in range(num_steps):
        batch_data, batch_labels = generate_batch(batch_size, num_skips, skip_window)
        feed_dict = {train_dataset : batch_data, train_labels : batch_labels}
        _, l = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += l
        if step % 100000 == 0 and step > 0:
            print('Average loss at step %d: %f' % (step, average_loss / 100000))
            average_loss = 0
    word2vec = normalized_embeddings.eval()
num_steps = 500001
import random
with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    average_loss = 0
    for step in range(num_steps):
        batch_data, batch_labels = generate_batch(batch_size, num_skips, skip_window)
        feed_dict = {train_dataset : batch_data, train_labels : batch_labels}
        _, l = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += l
        if step % 100000 == 0 and step > 0:
            print('Average loss at step %d: %f' % (step, average_loss / 100000))
            average_loss = 0
    word2vec = normalized_embeddings.eval()


distances = -word2vec[dictionary[u'我们']].reshape((1, -1)).dot(word2vec.T)
inds = np.argsort(distances.ravel())[1:6]
print(' '.join([reverse_dictionary[i] for i in inds]))
# ----------------------------------------------
# 资料 统计 显示 信息 证据

