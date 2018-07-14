from A001_asisst_function import *

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import ctypes
import time

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/data/", one_hot=True)

# Parameters
learning_rate = 0.5
training_epochs = 1000
batch_size = 100
display_step = 1

# tf Graph Input
x = tf.placeholder(tf.float32, [None, 784])  # mnist data image of shape 28*28=784
y = tf.placeholder(tf.float32, [None, 10])  # 0-9 digits recognition => 10 classes
W1 = tf.Variable(tf.random_normal([784, 256]))
W2 = tf.Variable(tf.random_normal([256, 10]))
# B1 = tf.Variable(tf.random_normal([10]))
# B2 = tf.Variable(tf.random_normal([10]))
FP_operation_W_sum = np.zeros(1)
# FP_operation_W2_sum = np.zeros(1)
Q_operation_Wb_sum = np.zeros(1)
# Q_operation_W2b_sum = np.zeros(1)

W1b=quantize(W1)
W2b=quantize(W2)
# W1b=W1
# W2b=W2

# W1b_pre=tf.Variable(initial_value=tf.zeros([784,10]))
# W1b_pre_place=tf.placeholder(dtype=tf.float32,shape=[784,10])
# W1b_pre_assign=W1b_pre.assign(W1b_pre_place)
# W2b_pre=tf.Variable(initial_value=tf.zeros([10,10]))
# W2b_pre_place=tf.placeholder(dtype=tf.float32,shape=[10,10])
# W2b_pre_assign=W2b_pre.assign(W2b_pre_place)
#
# W1_pre=tf.Variable(initial_value=tf.zeros([784,10]))
# W1_pre_place=tf.placeholder(dtype=tf.float32,shape=[784,10])
# W1_pre_assign=W1_pre.assign(W1_pre_place)
# W2_pre=tf.Variable(initial_value=tf.zeros([10,10]))
# W2_pre_place=tf.placeholder(dtype=tf.float32,shape=[10,10])
# W2_pre_assign=W2_pre.assign(W2_pre_place)

# L1 = tf.nn.relu(tf.add(tf.matmul(x, W1),B1))  #ReLU
L1 = tf.nn.relu(tf.matmul(x, W1b))  #ReLU
# pred = tf.add(tf.matmul(L1, W2), B2)  # Softmax
pred = tf.matmul(L1, W2b)  # Softmax
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=pred))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    with open("results/MNIST/784_256_10/8bit.txt", 'w') as f:
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(mnist.train.num_examples / batch_size)
            for i in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                # t_i=time.time()
                # W1_value, W2_value, W1b_value, W2b_value = sess.run([W1, W2, W1b, W2b])

                _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs, y: batch_ys})


                # W1_pre_value, W2_pre_value, W1b_pre_value, W2b_pre_value = sess.run([W1, W2, W1b, W2b])
                #
                # FP_operation_W_sum += FP_operation_W(W1_pre_value, W1_value) + FP_operation_W(W2_pre_value, W2_value)
                # Q_operation_Wb_sum += Q_operation_Wb(W1b_pre_value, W1b_value) + Q_operation_Wb(W2b_pre_value, W2b_value)
                # # print(time.time() - t_i)

                avg_cost += c / total_batch
            if (epoch + 1) % display_step == 0:

                correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(pred), 1), tf.argmax(y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                acc=accuracy.eval({x: mnist.test.images[:10000], y: mnist.test.labels[:10000]})
                # data=str(print("Ep:", '%03d' % (epoch + 1), "cost:", "{:.9f}".format(avg_cost), "Q_op:", Q_operation_Wb_sum, "FP_op:", FP_operation_W_sum, "Acc:",acc))
                data="Ep: " + '%03d ' % (epoch + 1) + "cost: " + "{:.9f} ".format(avg_cost) + "Q_op: "+ str(Q_operation_Wb_sum) + " FP_op: " +str(FP_operation_W_sum)+ " Acc: " +str(acc)
                print(data)
                f.write("%s\n" % data)


