import numpy as np

class NNFunction:
    def function(x):   raise NotImplementedError
    def derivative(x): raise NotImplementedError

class Sigmoid(NNFunction):
    def function(x):   return 1/(1 + np.exp(-x))
    def derivative(x): return Sigmoid.function(x) * (1 - Sigmoid.function(x))

class Tanh(NNFunction):
    def function(x):   return np.tanh(x)
    def derivative(x): return 1 - np.tanh(x) ** 2

class MSE(NNFunction):
    def function(target, prediction):   return np.mean(np.square(target - prediction))
    def derivative(target, prediction): return 2 * (prediction - target) / target.size
