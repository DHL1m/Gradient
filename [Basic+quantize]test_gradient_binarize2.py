import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# def xavier_initializer(n_inputs, n_outputs, uniform = True):
#     if uniform:
#         init_range = tf.sqrt(6.0/ (n_inputs + n_outputs))
#         return tf.random_uniform_initializer(-init_range, init_range)

#     else:
#         stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
#         return tf.truncated_normal_initializer(stddev=stddev)

# Import MNIST data

from tensorflow.examples.tutorials.mnist import input_data
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('dataset', 'MNIST_rand',
                           """Name of dataset used.""")
import funcData
# mnist = input_data.read_data_sets("/data/", one_hot=True)


# Parameters

learning_rate = 0.1

training_epochs = 1

batch_size = 100

display_step = 1

# tf Graph Input

x = tf.placeholder(tf.float32, [None, 784])  # mnist data image of shape 28*28=784

y = tf.placeholder(tf.float32, [None, 10])  # 0-9 digits recognition => 10 classes

# Set model weights

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
# a = tf.Variable(tf.ones([10]))
# d = tf.Variable(tf.ones([1]))

# Wb = (tf.cast(W > 0, tf.float32)-0.5)
# Wb = (tf.cast(W > 0, tf.float32)-0.5)*d

Wb= W

# Wb = W * 0
# fff = 0.0625 * 0.0625
# Wb = Wb + tf.cast(fff*2.0 < W, tf.float32) * (fff)
# Wb = Wb + tf.cast(fff < W, tf.float32) * (fff)
# Wb = Wb + tf.cast(0.0 < W, tf.float32) * (fff)
# Wb = Wb + tf.cast(0.0 >= W, tf.float32) * (-fff)
# Wb = Wb + tf.cast(-fff > W, tf.float32) * (-fff)
# Wb = Wb + tf.cast(-fff*2.0 > W, tf.float32) * (-fff)
# Wb = Wb + tf.cast(-fff*3.0 >= W, tf.float32) * (-fff)
# Wb = Wb + tf.cast(fff*3.0 < W, tf.float32) * (fff)
# Wb = Wb + tf.cast(-fff*4.0 >= W, tf.float32) * (W +4.0* fff)
# Wb = Wb + tf.cast(fff*4.0 < W, tf.float32) * (W -4.0* fff)
# Wb = Wb + tf.cast(-fff*5.0 >= W, tf.float32) * (-W -5.0* fff)
# Wb = Wb + tf.cast(fff*5.0 < W, tf.float32) * (-W +5.0* fff)


# inference only
#   0.125*0.25       0.125*0.125       0.25*0.25
# h      8997            8787              9160
#       9042            8779              9136
#       9049            8774              9170

# F                                        9138, 9138, 9185
# 3                                        9097
#                                         9148
#                                         9124


# Wb = Wb + tf.cast(fff * 2.0 < W, tf.float32) * (fff)
# Wb = Wb + tf.cast(fff < W, tf.float32) * (fff)
# Wb = Wb + tf.cast(0.0 < W, tf.float32) * (fff)
# Wb = Wb + tf.cast(0.0 >= W, tf.float32) * (-fff)
# Wb = Wb + tf.cast(-fff > W, tf.float32) * (-fff)
# Wb = Wb + tf.cast(-fff * 2.0 > W, tf.float32) * (-fff)
# Wb = Wb + tf.cast(-fff * 3.0 >= W, tf.float32) * (-fff)
# Wb = Wb + tf.cast(fff * 3.0 < W, tf.float32) * (fff)

# Wc = tf.Variable(tf.zeros([784, 10]))
# Wc = Wb


# Construct model

pred = tf.nn.softmax(tf.matmul(x, Wb) + b)  # Softmax
# pred2 = tf.nn.softmax(tf.matmul(x, W)+b)  # Softmax
# pred = tf.nn.softmax(tf.matmul(x, W))  # Softmax

# Minimize error using cross entropy

cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred+0.00001), reduction_indices=1))
# cost2 = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred2), reduction_indices=1))

# optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# # Calculate the gradient using tf.gradient and define the update code#

# grad_W, grad_b = tf.gradients(xs=[W,b], ys=cost2)
# grad_W, grad_a, grad_d = tf.gradients(xs=[W, a,d], ys=cost)
# grad_d = 0.01


grad_W = -tf.matmul(tf.transpose(x), y - pred)
grad_b = -tf.reduce_mean(tf.matmul(tf.transpose(x), y - pred), reduction_indices=0)
# grad_a = -tf.reduce_mean(tf.matmul(tf.transpose(x), y - pred), reduction_indices=0)

# W_up=(W-learning_rate*grad_W)

new_W = W.assign(W - learning_rate * grad_W)
new_b = b.assign(b - learning_rate * grad_b)
# new_a=a.assign(a-learning_rate*grad_a)
# new_d=d.assign(d-learning_rate*grad_d)

# Wbb = (tf.cast(W_up > 0, tf.float32)-0.5)

# Wbb= W_up*0
# fff= 0.125*0.25
# Wbb = Wbb + tf.cast(fff*2.0 < W_up, tf.float32) * (fff)
# Wbb = Wbb + tf.cast(fff < W_up, tf.float32) * (fff)
# Wbb = Wbb + tf.cast(0.0 < W_up, tf.float32) * (fff)
# Wbb = Wbb + tf.cast(0.0 >= W_up, tf.float32) * (-fff)
# Wbb = Wbb + tf.cast(-fff > W_up, tf.float32) * (-fff)
# Wbb = Wbb + tf.cast(-fff*2.0 > W_up, tf.float32) * (-fff)
# Wbb = Wbb + tf.cast(-fff*3.0 >= W_up, tf.float32) * (-fff)
# Wbb = Wbb + tf.cast(fff*3.0 < W_up, tf.float32) * (fff)
# Wbb = Wbb + tf.cast(-fff*4.0 >= W_up, tf.float32) * (W_up +4.0* fff)
# Wbb = Wbb + tf.cast(fff*4.0 < W_up, tf.float32) * (W_up -4.0* fff)
# Wbb = Wbb + tf.cast(-fff*5.0 >= W_up, tf.float32) * (-W_up -5.0* fff)
# Wbb = Wbb + tf.cast(fff*5.0 < W_up, tf.float32) * (-W_up +5.0* fff)


# Wbb = Wbb + tf.cast(fff*2.0 < W_up, tf.float32) * (fff)
# Wbb = Wbb + tf.cast(fff < W_up, tf.float32) * (fff)
# Wbb = Wbb + tf.cast(0.0 < W_up, tf.float32) * (fff)
# Wbb = Wbb + tf.cast(0.0 >= W_up, tf.float32) * (-fff)
# Wbb = Wbb + tf.cast(-fff > W_up, tf.float32) * (-fff)
# Wbb = Wbb + tf.cast(-fff*2.0 > W_up, tf.float32) * (-fff)
# Wbb = Wbb + tf.cast(-fff*3.0 >= W_up, tf.float32) * (-fff)
# Wbb = Wbb + tf.cast(fff*3.0 < W_up, tf.float32) * (fff)

# Wbb = Wbb + tf.cast(fff*2.0 < W_up, tf.float32) * (fff)
# Wbb = Wbb + tf.cast(fff < W_up, tf.float32) * (fff)
# Wbb = Wbb + tf.cast(0.0 < W_up, tf.float32) * (fff)
# Wbb = Wbb + tf.cast(0.0 >= W_up, tf.float32) * (-fff)
# Wbb = Wbb + tf.cast(-fff > W_up, tf.float32) * (-fff)
# Wbb = Wbb + tf.cast(-fff*2.0 > W_up, tf.float32) * (-fff)
# Wbb = Wbb + tf.cast(-fff*3.0 >= W_up, tf.float32) * (W + 3.0*fff)
# Wbb = Wbb + tf.cast(fff*3.0 < W_up, tf.float32) * (W - 3.0*fff)
# Wbb = Wbb + tf.cast(-fff*4.0 >= W_up, tf.float32) * (-W -4.0* fff)
# Wbb = Wbb + tf.cast(fff*4.0 < W_up, tf.float32) * (-W +4.0* fff)


# new_W = W.assign(Wbb)

#       0.25*0.25      0.125*0.25    0.125*0.125
# 0.01    6292            7268            6867
# 0.001                   6531            6074
# 0.003                   6348
# 0.1                     7372
# ---------------------------------------------------------------------#


# Dara to plot
training_cost = []
dataset=funcData.get_data_provider(FLAGS.dataset,training=True)
dataset_test=funcData.get_data_provider(FLAGS.dataset,training=False)
x_data,y_data=dataset.next_batch(batch_size)
y_data=tf.one_hot(y_data,depth=10)
x_data_test,y_data_test=dataset_test.next_batch(funcData.images_test.shape[0])
y_data_test=tf.one_hot(y_data_test,depth=10)
# Start training

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # Training cycle

    for epoch in range(training_epochs):

        avg_cost = 0.

        # total_batch = int(mnist.train.num_examples / batch_size)
        total_batch = int(funcData.images_train.shape[0]/batch_size)
        # Loop over all batches

        for i in range(total_batch):
            # batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            batch_xs, batch_ys =sess.run([x_data,y_data])
            # Fit training using batch data

            # _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs, y: batch_ys})
            _, _, c = sess.run([new_W, new_b, cost], feed_dict={x: batch_xs, y: batch_ys})
            #             _, _, _, c = sess.run([new_W, new_a, new_d, cost], feed_dict={x: batch_xs, y: batch_ys})
            #             _, c = sess.run([new_W, cost], feed_dict={x: batch_xs, y: batch_ys})

            #             print(__w)

            # Compute average loss

            avg_cost += c / total_batch

        # Display logs per epoch step

        if (epoch + 1) % display_step == 0:
            #             print(sess.run(W))

            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
            training_cost.append(avg_cost)
    #         if (epoch + 1) % display_step*10 == 0:
    #             learning_rate=learning_rate*0.2

    print("Optimization Finished!")
    coord.request_stop()
    coord.join(threads)
    coord.clear_stop()
    # Test model

    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # correct_prediction = tf.equal(tf.argmax(pred, 1), tf.cast(y, dtype=tf.int64))

    # Calculate accuracy for 3000 examples

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # print("Accuracy:", accuracy.eval({x: mnist.test.images[:10000], y: mnist.test.labels[:10000]}))
    print("Accuracy:", accuracy.eval({x: funcData.images_test.reshape(50000,784)[:50000], y: tf.one_hot(funcData.labels_test,10)}))


def plotCost(jvec):
    plt.figure(figsize=(6, 4))
    plt.plot(range(len(jvec)), jvec, 'bo')
    plt.grid(True)
    plt.title("Convergence of Cost Function")
    plt.xlabel("Epoch")
    plt.ylabel("Cost function")


plotCost(training_cost)