from network.neuralnetwork  import Network 
from network.visualize      import show_output_space
from network.activations    import *

import numpy as np

batches = np.array([
    [[0, 0]],
    [[0, 1]],
    [[1, 0]],
    [[1, 1]],
])
labels = np.array([
    [[0]],
    [[1]],
    [[1]],
    [[0]],
])

nn = Network(lr=.1)

nn.config((2, 3, 1), tanh, tanh_prime)
nn.train(batches, labels, 500)

print(nn.predict(batches))
show_output_space(nn, key=lambda x: x[0][0])
