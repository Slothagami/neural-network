from network.neuralnet   import *
from network.activations import *

from keras.datasets import mnist
from keras.utils import np_utils

import numpy as np

print("Loading Data...")
(train_batch, train_labels), (test_batch, test_labels) = mnist.load_data()

# Reshape Data
# train_batch = train_batch.reshape(train_batch.shape[0], 1, 28, 28).astype("float32")
# test_batch  = test_batch .reshape(test_batch .shape[0], 1, 28, 28).astype("float32")
train_batch = train_batch.reshape(train_batch.shape[0], 1, 28*28).astype("float32")
test_batch  = test_batch .reshape(test_batch .shape[0], 1, 28*28).astype("float32")

train_batch /= 255 # normalize pixel values
test_batch  /= 255

train_labels = np_utils.to_categorical(train_labels) # reformat to vector labels
test_labels  = np_utils.to_categorical(test_labels )


# Train
nn = NeuralNet(lr=.01)
nn.config((28*28, 100, 50, 10), Tanh)

# Config Layers of ConvNet
# depth = 3
# nn.add(ConvLayer((1, 28, 28), 5, depth))
# nn.add(ActivationLayer(ReLU))
# nn.add(ReshapeLayer((depth, 26, 26), (1, depth * 26 * 26)))

# nn.add(FCLayer(depth * 26 * 26, 100))
# nn.add(ActivationLayer(ReLU))

# nn.add(FCLayer(100, 10))
# nn.add(Softmax())

print("Beginning Training...")
nn.train(train_batch, train_labels, 5)


# Evaluate Accuracy
correct = 0
nsamples = 2000
for sample, label in zip(nn.predict(test_batch[:nsamples]), test_labels[:nsamples]): 
    prediction = np.abs(np.round(sample))[0]
    if np.array_equal(label, prediction):
        correct += 1 

accuracy = correct / nsamples * 100
print(f"\nAcuracy: {accuracy:.2f}% ({correct}/{nsamples})")
