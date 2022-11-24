from neuralnetwork   import Network 
from activations     import *

from keras.datasets import mnist
from keras.utils import np_utils

import numpy as np

(train_batch, train_labels), (test_batch, test_labels) = mnist.load_data()

# Reshape Training Data
train_batch = train_batch.reshape(train_batch.shape[0], 1, 28*28).astype("float32")
train_batch /= 255 # normalize pixel values

train_labels = np_utils.to_categorical(train_labels) # reformat into vector labels

# Reshape Testing Data
test_batch = test_batch.reshape(test_batch.shape[0], 1, 28*28).astype("float32")
test_batch /= 255 # normalize pixel values

test_labels = np_utils.to_categorical(test_labels) # reformat into vector labels



# Train
nn = Network(lr=.1)

nn.config((28*28, 100, 50, 10), tanh, tanh_prime)
nn.train(test_batch, test_labels, 10)

print("\nTest:")
for sample in nn.predict(test_batch[:3]): 
    print(sample)

print("\nLabels:")
for label in test_labels[:3]: 
    print(label)
