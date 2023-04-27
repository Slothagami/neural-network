from network import *
import numpy as np

test = np.arange(9).reshape((3, 3))
test = Matrix.from_numpy(test)
test = Matrix.to_numpy(test)
print(test)

nn = Net(mse_grad, mse)

nn.add(FCLayer(2, 3))
nn.add(TanhLayer())
nn.add(FCLayer(3, 2))
nn.add(TanhLayer())

batch  = np.array([[[0], [0]], [[0], [1]], [[1], [0]], [[1], [1]]])
labels = np.array([[[0], [1]], [[1], [0]], [[1], [0]], [[0], [1]]])

nn.train(batch, labels, 500, .1, 4, 100)
