#!/usr/bin/env python
# coding:utf-8


""" Bi-directional Recurrent Neural Network.
A Bi-directional Recurrent Neural Network (LSTM) implementation example using 
TensorFlow library.
Links:
    [Long Short Term Memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)
"""

from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import gensim
import pdb

# Import news data
from prepare_data import DataSets
news_data = DataSets()

'''
To classify images using a bidirectional recurrent neural network, we consider
every image row as a sequence of pixels. Because MNIST image shape is 28*28px,
we will then handle 28 sequences of 28 steps for every sample.
'''

# Training Parameters
learning_rate = 0.001
training_steps = 10000
batch_size = 128
display_step = 200

# Network Parameters
embedding_size = 50 # embedding size
timesteps = 32 # timesteps
num_hidden = 128 # hidden layer num of features
num_classes = 5 # MNIST total classes (0-4 digits)

# tf Graph input
x_input = tf.placeholder(tf.int32, [None, timesteps])
y_input = tf.placeholder(tf.int32, [None, timesteps])

# Define weights
weights = {
    # Hidden layer weights => 2*n_hidden because of forward + backward cells
    'out': tf.Variable(tf.random_normal([2*num_hidden, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}

w2v_model = gensim.models.Word2Vec.load("/usr/app/tensorflow-learn/corpus/word2vec/news.text.model")

def BiRNN(inputs, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, embedding_size)
    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, embedding_size)
    #x = tf.unstack(x, timesteps, 1)

    # ** 0.char embedding，请自行理解 embedding 的原理！！做 NLP 的朋友必须理解这个
    embedding = tf.get_variable("embedding", [10988, embedding_size], dtype=tf.float32)
    # X_inputs.shape = [batchsize, timesteps]  ->  inputs.shape = [batchsize, timesteps, embedding_size]
    inputs = tf.nn.embedding_lookup(embedding, inputs)

    # Define lstm cells with tensorflow
    # Forward direction cell
    lstm_fw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    # Backward direction cell
    lstm_bw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)


    # ** 2.dropout
    lstm_fw_cell = rnn.DropoutWrapper(cell=lstm_fw_cell, input_keep_prob=1.0, output_keep_prob=1.0)
    lstm_bw_cell = rnn.DropoutWrapper(cell=lstm_bw_cell, input_keep_prob=1.0, output_keep_prob=1.0)
    # ** 3.多层 LSTM
    cell_fw = rnn.MultiRNNCell([lstm_fw_cell]*2, state_is_tuple=True)
    cell_bw = rnn.MultiRNNCell([lstm_bw_cell]*2, state_is_tuple=True)

    # ** 4.初始状态
    initial_state_fw = cell_fw.zero_state(batch_size, tf.float32)
    initial_state_bw = cell_bw.zero_state(batch_size, tf.float32)  


    """
    # Get lstm cell output
    try:
        outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                              dtype=tf.float32)
    except Exception: # Old TensorFlow version only returns outputs not states
        outputs = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                        dtype=tf.float32)
    """

    # ** 5. bi-lstm 计算（展开）
    with tf.variable_scope('bidirectional_rnn'):
        # *** 下面，两个网络是分别计算 output 和 state 
        # Forward direction
        outputs_fw = list()
        state_fw = initial_state_fw
        with tf.variable_scope('fw'):
            for timestep in range(timesteps):
                if timestep > 0:
                    tf.get_variable_scope().reuse_variables()
                pdb.set_trace()
                (output_fw, state_fw) = cell_fw(inputs[:, timestep, :], state_fw)
                outputs_fw.append(output_fw)

        # backward direction
        outputs_bw = list()
        state_bw = initial_state_bw
        with tf.variable_scope('bw') as bw_scope:
            inputs = tf.reverse(inputs, [1])
            for timestep in range(timesteps):
                if timestep > 0:
                    tf.get_variable_scope().reuse_variables()
                (output_bw, state_bw) = cell_bw(inputs[:, timestep, :], state_bw)
                outputs_bw.append(output_bw)
        # *** 然后把 output_bw 在 timestep 维度进行翻转
        # outputs_bw.shape = [timesteps, batch_size, hidden_size]
        outputs_bw = tf.reverse(outputs_bw, [0])
        # 把两个oupputs 拼成 [timesteps, batch_size, hidden_size*2]
        output = tf.concat([outputs_fw, outputs_bw], 2)  
        # output.shape 必须和 y_input.shape=[batch_size,timesteps] 对齐
        output = tf.transpose(output, perm=[1,0,2])
        output = tf.reshape(output, [-1, hidden_size*2])
    # ***********************************************************
    softmax_w = weight_variable([hidden_size * 2, class_num]) 
    softmax_b = bias_variable([class_num]) 
    logits = tf.matmul(output, softmax_w) + softmax_b

    # Linear activation, using rnn inner loop last output
    #return tf.matmul(outputs[-1], weights['out']) + biases['out']
    return logits


logits = BiRNN(x_input, weights, biases)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_input, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# ** 0.char embedding，请自行理解 embedding 的原理！！做 NLP 的朋友必须理解这个
#embedding = tf.get_variable("embedding", [news_data.vocab_size, embedding_size], dtype=tf.float32)


# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, training_steps+1):
        batch_x, batch_y = news_data.train.next_batch(batch_size)

        # Reshape data to get 32(timestemps) seq of 64(embbeding_size) elements
        pdb.set_trace()
        batch_x = batch_x.reshape((batch_size, timesteps, embedding_size))
        # X_inputs.shape = [batchsize, timesteps]  ->  inputs.shape = [batchsize, timesteps, embedding_size]
        #batch_x = tf.nn.embedding_lookup(embedding, batch_x)  
        #pdb.set_trace()
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={x_input: batch_x, y_input: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={x_input: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for 128 mnist test images
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, timesteps, embedding_size))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy:", \
sess.run(accuracy, feed_dict={x_input: test_data, Y: test_label}))
