import tensorflow as tf

# Import MNIST data

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/data/", one_hot=True)

# Parameters

learning_rate = 0.01

training_epochs = 10

batch_size = 100

display_step = 1

# tf Graph Input

x = tf.placeholder(tf.float32, [None, 784])  # mnist data image of shape 28*28=784
y = tf.placeholder(tf.float32, [None, 10])  # 0-9 digits recognition => 10 classes

# Set model weights

# W1 = tf.Variable(tf.ones([784, 256]))
# W2 = tf.Variable(tf.ones([256, 10]))
# b = tf.Variable(tf.ones([256]))
# d = tf.Variable(tf.ones([10]))

# W1 = tf.Variable(tf.zeros([784, 256]))
# W2 = tf.Variable(tf.zeros([256, 10]))
# b = tf.Variable(tf.zeros([256]))
# d = tf.Variable(tf.zeros([10]))


W1 = tf.Variable(tf.random_normal([784, 256]))
W2 = tf.Variable(tf.random_normal([256, 10]))
B1 = tf.Variable(tf.random_normal([256]))
B2 = tf.Variable(tf.random_normal([10]))

# W1b = (tf.cast(W1 > 0, tf.float32) - 0.5)*2
# W2b = (tf.cast(W2 > 0, tf.float32) - 0.5)*2
# Wb = W

# Construct model

L1 = tf.nn.relu(tf.add(tf.matmul(x, W1),B1))  #ReLU
# pred = tf.nn.softmax(tf.add(tf.matmul(L1, W2), B2))  # Softmax
pred = tf.add(tf.matmul(L1, W2), B2)  # Softmax

# pred = tf.nn.softmax(tf.matmul(tf.nn.softmax(tf.matmul(x, W1)),W2))  # Softmax
# pred = tf.nn.softmax(tf.matmul(x, W1)+b)  # Softmax
# pred = tf.nn.softmax(tf.nn.sigmoid(tf.matmul(x, W1)+b))  # 이건 됨!
# pred = tf.nn.softmax(tf.matmul(tf.nn.softmax(tf.matmul(x, W1)), W2)+b)  #
# Minimize error using cross entropy


# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred)) # Softmax loss


# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=pred))
# cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred)))
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(tf.nn.softmax(pred))),  reduction_indices=1)

# optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# # Calculate the gradient using tf.gradient and define the update code#
# grad_W1, grad_W2 = tf.gradients(xs=[W1, W2], ys=cost)


# grad_W1, grad_b = tf.gradients(xs=[W1, b], ys=cost)  # 이건 되는데???
# grad_W1, grad_b, grad_W2 = tf.gradients(xs=[W1,b,W2], ys=cost)
# grad_W1, grad_B1, grad_W2, grad_B2 = tf.gradients(xs=[W1b,B1,W2b,B2], ys=cost)
grad_W1, grad_B1, grad_W2, grad_B2 = tf.gradients(xs=[W1,B1,W2,B2], ys=cost)

# grad_W1 = tf.gradients(xs=[W1], ys=cost)
# grad_b = tf.gradients(xs=[b], ys=cost)


# grad_b = tf.gradients(xs=b, ys=cost)
# grad_W1, grad_W2 = tf.gradients(xs=[W1, W2], ys=cost)
# grad_W1 = -tf.matmul(tf.transpose(x), y - pred)
# grad_b = -tf.reduce_mean(tf.matmul(tf.transpose(x), y - pred), axis=0)
# grad_d = tf.gradients(xs=d, ys=cost)
# grad_d=-0.001

# new_W1=W1.assign(W1-learning_rate*grad_W1)
# new_W2=W2.assign(W2-learning_rate*grad_W2)
# new_B1=B1.assign(B1-learning_rate*grad_B1)
# new_B2=B2.assign(B2-learning_rate*grad_B2)

#---------------------------------------------------------------------#

init = tf.global_variables_initializer()

# Start training

with tf.Session() as sess:
    sess.run(init)

    # Training cycle

    for epoch in range(training_epochs):

        avg_cost = 0.

        total_batch = int(mnist.train.num_examples / batch_size)

        # Loop over all batches

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)

            # Fit training using batch data

            # _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs, y: batch_ys})
            # _, _, c = sess.run([new_W1, new_b, cost], feed_dict={x: batch_xs, y: batch_ys})
            _, _, _, _, c = sess.run([new_W1, new_B1, new_W2, new_B2, cost], feed_dict={x: batch_xs, y: batch_ys})

            # _, c = sess.run([new_W1, cost], feed_dict={x: batch_xs, y: batch_ys})

            #             print(__w)



            # Compute average loss

            avg_cost += c / total_batch

        # Display logs per epoch step

        if (epoch + 1) % display_step == 0:
            #             print(sess.run(W))

            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")

    # Test model

    correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(pred), 1), tf.argmax(y, 1))

    # Calculate accuracy for 3000 examples

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print("Accuracy:", accuracy.eval({x: mnist.test.images[:3000], y: mnist.test.labels[:3000]}))