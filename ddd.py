import numpy as np

images_train = np.load("./MNIST_backrand/MNIST_rand_train_images.npy")
images_test = np.load("./MNIST_backrand/MNIST_rand_test_images.npy")
labels_train = np.load("./MNIST_backrand/MNIST_rand_train_labels.npy")
labels_test = np.load("./MNIST_backrand/MNIST_rand_test_labels.npy")
print(np.shape(labels_test))

print(labels_test[2,])

print(np.shape(images_test.reshape(50000,784)))



