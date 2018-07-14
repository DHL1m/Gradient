from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data

import os
import sys
import tarfile
import gzip
import tensorflow as tf
import math
from six.moves import urllib

# import csv
# import glob
# import re
import numpy as np
DATA_DIR = './Datasets/'
URLs = {
    'cifar10': 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz',
    'cifar100': 'http://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz',
    'MNIST_train_image': 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
    'MNIST_train_lable': 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
    'MNIST_test_image': 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
    'MNIST_test_label': 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
    }


def __maybe_download(data_url, dest_directory, apply_func=None):
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = data_url.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(data_url, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    if apply_func is not None:
        apply_func(filepath)


def __read_MNIST(training=True):
    """Reads and parses examples from MNIST data files."""

    mnist = input_data.read_data_sets("Datasets/MNIST", one_hot=False)
    if training:
        images=tf.cast(tf.convert_to_tensor(mnist.train.images.reshape([55000,784])),tf.float32)
        labels=tf.cast(tf.convert_to_tensor(mnist.train.labels),dtype=tf.int32)
    else:
        images = tf.cast(tf.convert_to_tensor(mnist.test.images.reshape([10000,784])),dtype=tf.float32)
        labels = tf.cast(tf.convert_to_tensor(mnist.test.labels),dtype=tf.int32)
    return images,labels
FLAGS = tf.app.flags.FLAGS
# tf.set_random_seed(FLAGS.seed)
if FLAGS.dataset=="MNIST_back":
    images_train =np.load("./MNIST_back/MNIST_back_train_images.npy")

    images_test = np.load("./MNIST_back/MNIST_back_test_images.npy")
    labels_train =np.load("./MNIST_back/MNIST_back_train_labels.npy")
    labels_test = np.load("./MNIST_back/MNIST_back_test_labels.npy")
elif FLAGS.dataset== "MNIST_rand":

    images_train =np.load("./MNIST_backrand/MNIST_rand_train_images.npy")
    images_test = np.load("./MNIST_backrand/MNIST_rand_test_images.npy")
    labels_train =np.load("./MNIST_backrand/MNIST_rand_train_labels.npy")
    labels_test = np.load("./MNIST_backrand/MNIST_rand_test_labels.npy")

def __read_MNIST_back(training=True):
    FLAGS = tf.app.flags.FLAGS
    # np.random.seed(FLAGS.seed)
    """Reads and parses examples from MNIST data files."""
    if training:
        images=tf.cast(tf.convert_to_tensor(images_train.reshape(12000,784)),tf.float32)
        labels=tf.cast(tf.convert_to_tensor(labels_train.reshape(12000,)),dtype=tf.int32)
    else:
        images=tf.cast(tf.convert_to_tensor(images_test.reshape(50000,784)),tf.float32)
        labels=tf.cast(tf.convert_to_tensor(labels_test.reshape(50000,)),dtype=tf.int32)
    return images,labels


"""
설명5:
클래스 안에 매서드가 한개있다! 클래스응용에 참 좋은 예제인 것 같다.
별거 없다. 데이터셋 받아서 그걸 일정 배치사이즈로 shuffle해주는거다.
min_queue_examples=1000, num_threads=8 는 미리 정해준다, 그리고 웬만하면 이 초깃값을 이용해주는 듯 하다.
만약 배치사이즈가 100이라면, 1300개를 queue에 쌓아놓고, 거기서 100개를 뽑아준다. 
아닌가? num_threads=8이니까 한번에 800개를 뽑으려나..? #미해결
어쩄든 min_queue_examples 가 크면 클수록 큰사이즈의 queue에서 데이터를 뽑게 되므로 잘섞어서 뽑는 셈이 된다.
"""

class DataProvider:
    def __init__(self, data, size=None, training=True,MNIST=False):
        self.size = size or [None]*4
        self.data = data
        self.training = training
        self.enqueue_many=MNIST

    def next_batch(self, batch_size, min_queue_examples=3000, num_threads=8):
        """Construct a queued batch of images and labels.

        Args:
        image: 3-D Tensor of [height, width, 3] of type.float32.
        label: 1-D Tensor of type.int32
        min_queue_examples: int32, minimum number of samples to retain
        in the queue that provides of batches of examples.
        batch_size: Number of images per batch.

        Returns:
        images: Images. 4D tensor of [batch_size, height, width, 3] size.
        labels: Labels. 1D tensor of [batch_size] size.
        """
        # Create a queue that shuffles the examples, and then
        # read 'batch_size' images + labels from the example queue.

        image, label = self.data
        if self.training:
            images, label_batch = tf.train.shuffle_batch(
                [image, label],
                batch_size=batch_size,
                num_threads=num_threads,
                capacity=5000,
                min_after_dequeue=min_queue_examples, enqueue_many=self.enqueue_many)
        else:
            images, label_batch = tf.train.batch(
                [image, label],
                batch_size=batch_size,
                num_threads=num_threads,
                capacity=min_queue_examples + 3 * batch_size, enqueue_many=self.enqueue_many)

        return images, tf.reshape(label_batch, [batch_size])

def get_data_provider(name,training=True):

    if name == 'MNIST':
        if training:
            return DataProvider(__read_MNIST(training=True), size=[55000, 28, 28, 1], training=True, MNIST=True)
        else:
            return DataProvider(__read_MNIST(training=False), size=[10000, 28, 28, 1], training=False,
                                MNIST=True)
    elif name == 'MNIST_rand' or 'MNIST_back':
        if training:
            return DataProvider(__read_MNIST_back(training=True), size=[12000, 28, 28, 1], training=True, MNIST=True)
        else:
            return DataProvider(__read_MNIST_back(training=False), size=[50000, 28, 28, 1], training=False,
                                MNIST=True)
            # if training:
        #     return DataProvider(__read_MNIST(training=True),size=[55000, 28,28, 1], training=True,argum=False)
        # else:
        #     return DataProvider(__read_MNIST(training=False),size=[10000, 28,28, 1],training=False,argum=False)

