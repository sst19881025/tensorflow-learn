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
import gensim
import traceback
import pdb

from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

from scpy.logger import get_logger
logger = get_logger(__file__)


def get_mnist_data():
    # 首先导入数据，看一下数据的形式
    mnist = read_data_sets('/home/sst/Documents/socialcredits/data/tenserflow/mnist', one_hot=True)
    #print mnist.train.images.shape
    return mnist





class Corpus(object):
    def __init__(self, db, collection, usage, word2vec):
        self._db = db
        self._collection = collection
        self._usage = usage
        #self._char2id = char2id
        self._word2vec = word2vec
        self._cursor = 0
        self._embedding_size = 64
        self._label_size = 5

    def __get_data(self):
        """ 从mongo读取数据iteration
        """
        conn = pymongo.MongoClient(['127.0.0.1'])
        db = conn[self._db]
        data = db[self._collection].find({}, {'word':1, 'tag':1, '_id':0})
        count = data.count()
        if self._usage == 'train':
            data = db[self._collection].find({}, {'word':1, 'tag':1, '_id':1})[:int(count*0.8)]
        elif self._usage == 'test':
            data = db[self._collection].find({}, {'word':1, 'tag':1, '_id':1})[int(count*0.2):]
        return data

    def next_batch(self, size=100):
        """ 获取下一个batch的数据
        """
        data = self.__get_data()
        res = self.__process_data(list(data[self._cursor:self._cursor+size]))
        self._cursor += size
        return res

    def __process_data(self, iter_data):
        """ batch数据转换成(X, Y)的二元组, 其中X和Y为ndarray格式
        """
        X, Y = [], []
        for doc in iter_data:
            cur_sentence = doc['word']
            cur_labels = doc['tag']
            X.append(self.__fill_x_symbols(cur_sentence))
            Y.append(self.__fill_y_symbols(cur_labels))
        return (np.row_stack(X), np.row_stack(Y))

    def __fill_x_symbols(self, sentence):
        """ 不足size的补0，超出size的截断
        """
        try:
            result = []
            # 补位 / 截断后
            sentence = filter(lambda x:x in self._word2vec.wv, sentence)
            sentence = sentence[:32] + ['unknown'] * (32-min(32, len(sentence)))
            for wd in sentence:
                word_vec = self._word2vec.wv[wd]
                result.append(word_vec)

            res_arr = np.array(result)
            result = np.row_stack(res_arr.reshape((1, res_arr.size)))
        except:
            err = traceback.format_exc()
            logger.error(err)
            pdb.set_trace()
        return result

    def __fill_y_symbols(self, sentence):
        """ 不足size的补0，超出size的截断
        """
        try:
            result = []
            sentence = sentence[:32] + [0] * (32-min(32, len(sentence)))
            # 数据列，one-hot编码之后非零的列
            result = np.zeros((32, self._label_size))
            for i, pos in enumerate(sentence):
                result[i][pos] = 1

            res_arr = np.array(result)
            result = np.row_stack(res_arr.reshape((1, res_arr.size)))
        except:
            err = traceback.format_exc()
            logger.error(err)
            pdb.set_trace()
        return result


class DataSets(object):
    def __init__(self):
        self._word2vec = gensim.models.Word2Vec.load("./word2vec/news.text.model")
        #self.__mapping_id('corpus', 'news')
        self.train = Corpus('corpus', 'news_sentence', 'train', self._word2vec)
        self.test = Corpus('corpus', 'news_sentence', 'test', self._word2vec)





