from network import *
from datasets import mnist_fc

mnist = mnist_fc(1000)

nn = Net(mse_grad, mse)
nn.add(FCLayer(28**2, 100))
nn.add(TanhLayer())
nn.add(FCLayer(100, 50))
nn.add(TanhLayer())
nn.add(FCLayer(50, 10))

nn.train(mnist, 5, .001, 1, 1)
# Gets ~98% acc w/ 5 epochs, lr=.001 and n_test = 4000
