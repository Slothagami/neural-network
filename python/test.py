from network import *
import numpy as np

nn = Net(mse_grad, mse)

nn.add(FCLayer(2, 3))
nn.add(TanhLayer())
nn.add(FCLayer(3, 2))
nn.add(TanhLayer())
