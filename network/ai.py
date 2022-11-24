from layers          import FCLayer, ActivationLayer
from neuralnetwork   import Network 
from activations     import *

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

nn.add(FCLayer(2, 3))
nn.add(ActivationLayer(func=tanh, derivative=tanh_prime))
nn.add(FCLayer(3, 1))
nn.add(ActivationLayer(func=tanh, derivative=tanh_prime))

nn.train(batches, labels, 1000)
print(nn.predict(batches))
