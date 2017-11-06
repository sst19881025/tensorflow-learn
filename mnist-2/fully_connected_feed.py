#!/usr/bin/env python
# coding:utf-8


'''1.导入模块
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=missing-docstring
import argparse
import os.path
import sys
import time
import pdb

from six.moves import xrange    # pylint: disable=redefined-builtin
import tensorflow as tf
import mnist
import input_data

# Basic model parameters as external flags.
FLAGS = None

'''2.占位符
目的：产生图片及标签的占位符
输入：batch_size
输出：Images placehodler（float32），Labels placeholder（int32）
'''
def placeholder_inputs(batch_size):
    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, mnist.IMAGE_PIXELS))
    labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
    return images_placeholder, labels_placeholder

'''3.填充喂养字典
目的：在训练时对应次数自动填充字典
输入：数据源data_set，来自input_data.read_data_sets
     图片holder，images_pl,来自placeholder_inputs()
     标签holder,labels_pl,来自placeholder_inputs()
输出：喂养字典feed_dict.
'''
def fill_feed_dict(data_set, images_pl, labels_pl):
    images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size, FLAGS.fake_data)
    feed_dict = {
        images_pl: images_feed,
        labels_pl: labels_feed,
    }
    return feed_dict

'''4.评估
目的：每循环1000次或结束进行一次评估。
输入：sess: 模型训练所使用的Session
    eval_correct: 预测正确的样本数量
    images_placeholder: images placeholder.
    labels_placeholder: labels placeholder.
    data_set: 图片和标签数据，来自input_data.read_data_sets().
输出：打印测试结果。
'''
def do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            data_set):
    true_count = 0    # 记录预测正确的数目。
    steps_per_epoch = data_set.num_examples // FLAGS.batch_size
    num_examples = steps_per_epoch * FLAGS.batch_size
    for step in xrange(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set, images_placeholder, labels_placeholder)
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = float(true_count) / num_examples
    print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
                (num_examples, true_count, precision))

'''5.训练过程
'''
def run_training():
    # 获取数据
    data_sets = input_data.read_data_sets(FLAGS.input_data_dir, FLAGS.fake_data)
    # 在默认Graph下运行.
    with tf.Graph().as_default():
        # 配置graph
        images_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size)
        # logits是shape为(batch_size，NUM_CLASSES)，表示每条数据预测后未归一化的概率分布
        logits = mnist.inference(images_placeholder,
                                    FLAGS.hidden1,
                                    FLAGS.hidden2)
        # 损失函数
        loss = mnist.loss(logits, labels_placeholder)
        train_op = mnist.training(loss, FLAGS.learning_rate)
        eval_correct = mnist.evaluation(logits, labels_placeholder)

        # 汇聚tensor
        summary = tf.summary.merge_all()
        # 建立初始化机制
        init = tf.global_variables_initializer()
        # 建立保存机制
        saver = tf.train.Saver()
        # 建立Session
        sess = tf.Session()

        # 建立一个SummaryWriter输出汇聚的tensor
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

        # 开始执行

        # 执行变量
        sess.run(init)

        # 开始训练，2000次循环
        for step in xrange(FLAGS.max_steps):
            start_time = time.time()

            #获取当次循环的数据
            feed_dict = fill_feed_dict(data_sets.train,
                                        images_placeholder,
                                        labels_placeholder)

            # 丢弃了train数据, 记录loss数据, sess.run(train_op)是最简单的训练代码
            # run(self, fetches, feed_dict=None, options=None, run_metadata=None)
            # fetches格式很自由，详情见help(tf.Session.run)
            _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
            duration = time.time() - start_time

            # 每训练100次输出当前损失，并记录数据
            if step % 100 == 0:
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            # 每1000次测试模型
            if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=step)
                # Evaluate against the training set.
                print('Training Data Eval:')
                do_eval(sess,
                        eval_correct,
                        images_placeholder,
                        labels_placeholder,
                        data_sets.train)
                # Evaluate against the validation set.
                print('Validation Data Eval:')
                do_eval(sess,
                        eval_correct,
                        images_placeholder,
                        labels_placeholder,
                        data_sets.validation)
                # Evaluate against the test set.
                print('Test Data Eval:')
                do_eval(sess,
                        eval_correct,
                        images_placeholder,
                        labels_placeholder,
                        data_sets.test)


def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    run_training()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.01,
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--max_steps',
        type=int,
        default=2000,
        help='Number of steps to run trainer.'
    )
    parser.add_argument(
        '--hidden1',
        type=int,
        default=128,
        help='Number of units in hidden layer 1.'
    )
    parser.add_argument(
        '--hidden2',
        type=int,
        default=32,
        help='Number of units in hidden layer 2.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='Batch size. Must divide evenly into the dataset sizes.'
    )
    parser.add_argument(
        '--input_data_dir',
        type=str,
        default='/home/sst/Documents/socialcredits/data/tenserflow/mnist/',
        help='Directory to put the input data.'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='./py_log/',
        help='Directory to put the log data.'
    )
    parser.add_argument(
        '--fake_data',
        default=False,
        help='If true, uses fake data for unit testing.',
        action='store_true'
    )
    # FLAGS用以记录parser.argument的结果
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


