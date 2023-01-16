from network.neuralnet   import *
from network.activations import *

from mimic_dataset import make_data

samples, labels = make_data()
in_size = samples[0].shape[0]
out_size = labels[0].shape[0]

# Train
nn = NeuralNet(lr=.002)
nn.config((in_size, 100, 50, 100, out_size), ReLU)

print("Beginning Training...")
nn.train(samples, labels, 10)
