from network import *
import numpy as np

print("Loading Data...")
from keras.datasets import mnist
from keras.utils import np_utils

n_test = 4000
(train_batch, train_labels), (test_batch, test_labels) = mnist.load_data()

# Reshape Data
train_batch = train_batch.reshape(train_batch.shape[0], 1, 28*28).astype("float32")[:n_test]
test_batch  = test_batch .reshape(test_batch .shape[0], 1, 28*28).astype("float32")[:n_test]

train_batch /= 255 # normalize pixel values
test_batch  /= 255

train_labels = np_utils.to_categorical(train_labels)[:n_test] # reformat to vector labels
test_labels  = np_utils.to_categorical(test_labels )[:n_test]
print("Data Loaded.")

nn = Net(mse_grad, mse)
nn.add(FCLayer(28**2, 100))
nn.add(TanhLayer())
nn.add(FCLayer(100, 50))
nn.add(TanhLayer())
nn.add(FCLayer(50, 10))

nn.train(train_batch, train_labels, 5, .001, 1, 1, test_batch, test_labels)
# Gets ~98% acc w/ 5 epochs, lr=.001 and n_test = 4000
