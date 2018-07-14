import tensorflow as tf
tf.set_random_seed(333)
a=tf.random_normal([10])
with tf.Session() as sess:
    print(sess.run(a))