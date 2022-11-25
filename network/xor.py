from neuralnetwork  import Network 
from activations    import *
from layers         import FCLayer

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
nn.train(batches, labels, 1000)

print(nn.predict(batches))
