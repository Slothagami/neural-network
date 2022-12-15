from network.activations import NNFunction, MSE, Sigmoid
import numpy as np

class NeuralNet:
    def __init__(self, /, loss: NNFunction = MSE, lr=.1):
        self.layers = []

        self.lr = lr
        self.loss = loss

    def add(self, layer): self.layers.append(layer)

    def config(self, layer_sizes, activation: NNFunction, /, layertypes=None):
        for i in range(1, len(layer_sizes)):
            layertype = layertypes[i-1] if layertypes is not None else FCLayer
            self.add(layertype(layer_sizes[i-1], layer_sizes[i]))
            self.add(ActivationLayer(activation))

    def predict(self, samples):
        # Predict a Batch of samples
        results = []
        for sample in samples:
            output = sample

            for layer in self.layers:
                output = layer.forward(output)

            results.append(output)

        return results

    def train(self, samples, labels, epochs):
        for epoch in range(epochs):
            disp_error = 0 

            for sample, label in zip(samples, labels):
                # Forward Propogation
                output = sample 
                for layer in self.layers:
                    output = layer.forward(output)

                # Display Error
                disp_error += self.loss.function(label, output)

                # Backprop
                error = self.loss.derivative(label, output)
                for layer in reversed(self.layers):
                    error = layer.backprop(error, self.lr)

            # Calc Average Error
            disp_error /= len(samples)

            print(f"Epoch: {epoch + 1}, Error: {disp_error}")


# Layers #
class Layer:
    def __init__(self): self.input = self.output = None 
    def forward():  raise NotImplementedError
    def backprop(): raise NotImplementedError

class FCLayer(Layer):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weights = np.random.rand(in_size, out_size) - .5
        self.bias    = np.random.rand(1, out_size) - .5

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
    def __init__(self, activation: NNFunction = Sigmoid):
        super().__init__()
        self.activation = activation

    def forward(self, input):
        self.input = input
        self.output = self.activation.function(input)
        return self.output

    def backprop(self, out_error, lr):
        return self.activation.derivative(self.input) * out_error
