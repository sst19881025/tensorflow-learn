# -- coding: utf-8 --

import tensorflow as tf
import pdb
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("../data/MNIST_data/", one_hot=True)

# 定义变量
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# 明确模型
y = tf.nn.softmax(tf.matmul(x,W) + b)


# 定义真实值
y_ = tf.placeholder("float", [None,10])

# 定义交叉熵
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

# 定义梯度下降来迭代一步
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)


# 在运行计算之前，我们需要添加一个操作来初始化我们创建的变量：
init = tf.global_variables_initializer()
# 现在我们可以在一个Session里面启动我们的模型，并且初始化变量：
sess = tf.Session()
sess.run(init)


# 然后开始训练模型，这里我们让模型循环训练1000次！
for i in range(1000):
    # 该循环的每个步骤中，我们都会随机抓取训练数据中的100个批处理数据点
    batch_xs, batch_ys = mnist.train.next_batch(100)
    # 然后我们用这些数据点作为参数替换之前的占位符来运行train_step。 
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
pdb.set_trace()
print accuracy.eval(feed_dict = {x: mnist.test.images, y_: mnist.test.labels}, session=sess)


