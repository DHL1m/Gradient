import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import ctypes

from A001_asisst_function import *

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/data/", one_hot=True)

# Parameters
learning_rate = 0.1
training_epochs = 1000
batch_size = 100
display_step = 1

# tf Graph Input
x = tf.placeholder(tf.float32, [None, 784])  # mnist data image of shape 28*28=784
y = tf.placeholder(tf.float32, [None, 10])  # 0-9 digits recognition => 10 classes
W1 = tf.Variable(tf.random_normal([784, 256]))
W2 = tf.Variable(tf.random_normal([256, 10]))
B1 = tf.Variable(tf.random_normal([256]))
B2 = tf.Variable(tf.random_normal([10]))
W1b=quantize(W1)
W2b=quantize(W2)
# W1b=W1
# W2b=W2

# L1 = tf.nn.relu(tf.add(tf.matmul(x, W1),B1))  #ReLU
L1 = tf.nn.relu(tf.matmul(x, W1b))  #ReLU
# pred = tf.add(tf.matmul(L1, W2), B2)  # Softmax
pred = tf.matmul(L1, W2b)  # Softmax
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=pred))
# cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(tf.nn.softmax(pred)+0.000001)))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples / batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs, y: batch_ys})
            avg_cost += c / total_batch
        if (epoch + 1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
    print("Optimization Finished!")
    correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(pred), 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({x: mnist.test.images[:10000], y: mnist.test.labels[:10000]}))