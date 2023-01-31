import numpy as np

# Activation Functions #
class NNFunction:
    def function(x):   raise NotImplementedError
    def derivative(x): raise NotImplementedError

class Sigmoid(NNFunction):
    def function(x):   return 1/(1 + np.exp(-x))
    def derivative(x): return Sigmoid.function(x) * (1 - Sigmoid.function(x))

class Tanh(NNFunction):
    def function(x):   return np.tanh(x)
    def derivative(x): return 1 - np.tanh(x) ** 2

class ReLU(NNFunction):
    def function(x):   return np.maximum(0, x)
    def derivative(x): return (x >= 0).astype(int) # 1 iff x >= 0

# Loss Functions #
class MSE(NNFunction):
    def function(target, prediction):   return np.mean(np.square(target - prediction))
    def derivative(target, prediction): return 2 * (prediction - target) / target.size

class CategoricalCrossEntropy(NNFunction):
    def function(target, prediction):   return np.sum(target * np.log(prediction)) # issue: log of 0 encountered
    def derivative(target, prediction): return prediction - target # not sure this is the derivative
    
class BinaryCrossEntropy(NNFunction):
    def function(target, prediction):   return -(target * np.log(prediction) + (1-target) * np.log(1-prediction))
    def derivative(target, prediction): return target / prediction + (1 - target) / (1 - prediction)
