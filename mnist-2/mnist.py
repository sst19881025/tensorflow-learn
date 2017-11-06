#!/usr/bin/env python
# coding:utf-8

'''1.导入模块
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf
'''2.系统参数，让程序更直观，方便修改
'''
# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 10

# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

'''3.神经网络图
输入：输入图片，隐藏层1神经元个数，隐藏层2神经元个数
输出：神经网络输出
'''
def inference(images, hidden1_units, hidden2_units):
    # http://www.tensorfly.cn/tfdoc/api_docs/python/constant_op.html#truncated_normal
    # weights就是我们要训练的参数, 一般用tf.Variable定义
    # Hidden 1
    with tf.name_scope('hidden1'):
        weights = tf.Variable(
            # 784维度(IMAGE_PIXELS) -> 128维度(hidden1_units)
            tf.truncated_normal([IMAGE_PIXELS, hidden1_units],
                                stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))),
                                name='weights')
        biases = tf.Variable(tf.zeros([hidden1_units]), name='biases')
        # relu是激活函数
        hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
    # Hidden 2
    with tf.name_scope('hidden2'):
        weights = tf.Variable(
            # 128维度(hidden1_units) -> 32维度(hidden2_units)
            tf.truncated_normal([hidden1_units, hidden2_units],
                                stddev=1.0 / math.sqrt(float(hidden1_units))),
                                name='weights')
        biases = tf.Variable(tf.zeros([hidden2_units]), name='biases')
        # relu是激活函数
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
    # Linear
    with tf.name_scope('softmax_linear'):
        weights = tf.Variable(
            # 32维度(hidden2_units) -> 类数(NUM_CLASSES)
            tf.truncated_normal([hidden2_units, NUM_CLASSES],
                                stddev=1.0 / math.sqrt(float(hidden2_units))),
                                name='weights')
        biases = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases')
        # logits是未归一化的概率分布
        logits = tf.matmul(hidden2, weights) + biases
    return logits

'''4.输出损失计算方法
输入： logits：网络输出，为float类型，[batch_size,NUM_CLASSES]
        labels：目标标签，为int32类型，[batch_size]
输出：损失，float类型
'''
def loss(logits, labels):
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                labels=labels, logits=logits, name='xentropy')
    return tf.reduce_mean(cross_entropy, name='xentropy_mean')

'''5.训练方法
输入：损失，学习速率
输出：训练op
训练方法为梯度下降。
'''
def training(loss, learning_rate):
    # Add a scalar summary for the snapshot loss.
    tf.summary.scalar('loss', loss)
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

'''6.评估训练效果
输入：logits：网络输出，float32，[batch_size, NUM_CLASSES]
            labels：标签，int32，[batch_size]
输出：预测正确的数量
'''
def evaluation(logits, labels):
    # label在top-k中为True，否则为False
    correct = tf.nn.in_top_k(logits, labels, k=1)
    return tf.reduce_sum(tf.cast(correct, tf.int32))



