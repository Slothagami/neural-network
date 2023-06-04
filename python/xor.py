from network import *
import numpy as np

nn = Net(mse_grad, mse)

nn.add(FCLayer(2, 3))
nn.add(TanhLayer())
nn.add(FCLayer(3, 2))
nn.add(TanhLayer())

batch  = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
labels = np.array([[[0, 1]], [[1, 0]], [[1, 0]], [[0, 1]]])
train_data = (batch, labels), (batch, labels)

nn.train(train_data, 500, .1, 4, 100)
