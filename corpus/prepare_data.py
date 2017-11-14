#!/usr/bin/env python
# coding:utf-8

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Functions for downloading and reading MNIST data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import tempfile
import string
import numpy as np
import tensorflow as tf
import pymongo
import pdb

from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets


def get_mnist_data():
    # 首先导入数据，看一下数据的形式
    mnist = read_data_sets('/home/sst/Documents/socialcredits/data/tenserflow/mnist', one_hot=True)
    #print mnist.train.images.shape
    return mnist


class Corpus(object):
    def __init__(self, db, collection, usage, char2id):
        self._db = db
        self._collection = collection
        self._usage = usage
        self._char2id = char2id
        self._cursor = 0

    def __get_data(self):
        """ 从mongo读取数据iteration
        """
        conn = pymongo.MongoClient(['127.0.0.1'])
        db = conn[self._db]
        data = db[self._collection].find({}, {'word':1, 'tag':1, '_id':0})
        count = data.count()
        if self._usage == 'train':
            data = db[self._collection].find({}, {'word':1, 'tag':1, '_id':0})[:int(count*0.8)]
        elif self._usage == 'test':
            data = db[self._collection].find({}, {'word':1, 'tag':1, '_id':0})[int(count*0.2):]
        return data

    def next_batch(self, size=100):
        """ 获取下一个batch的数据
        """
        data = self.__get_data()
        res = self.__process_data(data[self._cursor:self._cursor+size])
        self._cursor += size
        return res

    def __process_data(self, iter_data):
        """ batch数据转换成(X, Y)的二元组, 其中X和Y为ndarray格式
        """
        X, Y = [], []
        for doc in iter_data:
            word_list = doc['word']
            cur_sentence = []
            cur_labels = []

            for word in word_list:
                # 对每个词（双十一/你好 等）中的每个字给标签
                word_len = len(word)
                if word_len == 1:
                    cur_labels += [1]
                elif word_len == 2:
                    cur_labels += [2, 4]
                else:
                    cur_labels += [2] + [3] * (word_len - 2) + [4]
                for char in word:
                    if char not in [u',', u'，', u';', u'；', u'。', u'?', u'？', u'!', u'！']:
                        cur_sentence.append(self._char2id.get(char, 0))
                    else:
                        X.append(self.__fill_symbols(cur_sentence, embedding_size=64))
                        Y.append(self.__fill_symbols(cur_labels, embedding_size=5))
                        cur_sentence = []
                        cur_labels = []
        return (np.array(X, dtype=np.int32), np.array(Y))

    def __fill_symbols(self, sentence, embedding_size):
        """ 不足size的补0，超出size的截断
        """
        wd_list = sentence[:size] + [0] * (32-len(sentence))
        vec_list = 


class DataSets(object):
    def __init__(self):
        self._char2id = {}
        self._id2char = {}
        self.__mapping_id('corpus', 'news')
        self.train = Corpus('corpus', 'news', 'train', self._char2id)
        self.test = Corpus('corpus', 'news', 'test', self._char2id)

    def __mapping_id(self, db, collection):
        conn = pymongo.MongoClient(['127.0.0.1'])
        db = conn[db]
        data = db[collection].find({}, {'word':1, 'tag':1, '_id':0})
        self.count = data.count()
        for doc in data:
            word_list = doc['word']
            for word in word_list:
                for char in word:
                    if char not in self._char2id:
                        self._char2id.update({char : len(self._char2id)+1})
                        self._id2char.update({len(self._char2id)+1 : char})
        self.vocab_size = len(self._char2id)



def sent2vec2(sent, vocab, ctxWindows = 5):

    charVec = []
    for char in sent:
        if char in vocab:
            charVec.append(vocab[char])
        else:
            charVec.append(vocab['retain-unknown'])
    #首尾padding
    num = len(charVec)
    pad = int((ctxWindows - 1)/2)
    for i in range(pad):
        charVec.insert(0, vocab['retain-padding'] )
        charVec.append(vocab['retain-padding'] )
    X = []
    for i in range(num):
        X.append(charVec[i:i + ctxWindows])
    return X

def sent2vec(sent, vocab, ctxWindows = 5):
    chars = []
    words = sent.split()
    for word in words:
        #包含两个字及以上的词
        if len(word) > 1:
            #词的首字
            chars.append(word[0] + '_b')
            #词中间的字
            for char in word[1:(len(word) - 1)]:
                chars.append(char + '_m')
            #词的尾字
            chars.append(word[-1] + '_e')
        #单字词
        else: 
            chars.append(word + '_s')

    return sent2vec2(chars, vocab, ctxWindows = ctxWindows)

