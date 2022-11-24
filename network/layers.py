import numpy as np

# based on towardsdatascience.com/math-neural-network-from-scratch-in-python-d6da9f29ce65
class Layer:
    def __init__(self): self.input = self.output = None 
    def forward():  raise NotImplementedError
    def backprop(): raise NotImplementedError

class FCLayer(Layer):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weights = np.random.rand(in_size, out_size) - .5
        self.bias    = np.random.rand(out_size) - .5

    def forward(self, input):
        self.input = input 
        self.output = np.dot(input, self.weights) + self.bias 
        return self.output

    def backprop(self, out_error, lr):
        # Calc Error
        in_error = np.dot(out_error, self.weights.T)
        weights_error = np.dot(self.input.T, out_error)

        # Update Weights
        self.weights -= lr * weights_error
        self.bias    -= lr * out_error

        return in_error

class ActivationLayer(Layer):
    def __init__(self, func, derivative):
        super().__init__()
        self.func = func
        self.derivative = derivative

    def forward(self, input):
        self.input = input
        self.output = self.func(input)
        return self.output

    def backprop(self, out_error, lr):
        return self.derivative(self.input) * out_error
