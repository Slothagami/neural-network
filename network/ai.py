from layers          import FCLayer, ActivationLayer
from neuralnetwork   import Network 
from activations     import MSE, MSE_prime, sigmoid, sigmoid_prime

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
nn.add(ActivationLayer())
nn.add(FCLayer(3, 1))
nn.add(ActivationLayer())

nn.train(batches, labels, 1000)
print(nn.predict(batches))
