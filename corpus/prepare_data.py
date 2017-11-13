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
import numpy
import tensorflow as tf
import pymongo
import pdb

from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

# 首先导入数据，看一下数据的形式
mnist = read_data_sets('/home/sst/Documents/socialcredits/data/tenserflow/mnist', one_hot=True)
#print mnist.train.images.shape


class Corpus(object):
    def __init__(self, db, collection, char2id):
        self._data = self.__get_data(db, collection)
        self._char2id = char2id
        self._cursor = 0

    def __get_data(self, db, collection):
        conn = pymongo.MongoClient(['127.0.0.1'])
        db = conn[db]
        data = db[collection].find({}, {'word':1, 'tag':1, '_id':0})
        return data

    def next_batch(self, size=100):
        res = self.__process_data(self._data[self._cursor:self._cursor+size])
        self._cursor += size
        return res

    def __process_data(self, iter_data):
        X, Y = [], []
        for doc in inter_data:
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
                        cur_sentence.append(self._char2id.get(char))
                    else:
                        X.append(self.__fill_symbols(cur_sentence))
                        Y.append(self.__fill_symbols(cur_labels))
                        cur_sentence = []
                        cur_labels = []
        return (np.array(X), np.array(Y))

    def __fill_symbols(self, sentence, size=32):
        return sentence[:size] + [0] * (size-len(sentence))


class DataSets(object):
    def __init__(self):
        self._char2id = {}
        self._id2char = {}
        self.__mapping_id'corpus', 'news_train')
        self.train = Corpus('corpus', 'news_train', self._char2id)
        self.test = Corpus('corpus', 'news_test', self._char2id)

    def __mapping_id(self, db, collection):
        conn = pymongo.MongoClient(['127.0.0.1'])
        db = conn[db]
        data = db[collection].find({}, {'word':1, 'tag':1, '_id':0})
        for doc in data:
            word_list = doc['word']
            for word in word_list:
                for char in word:
                    if char not in self._char2id:
                        self._char2id.update({char: len(self._char2id)})
                        self._id2char.update({len(self._char2id): char})




