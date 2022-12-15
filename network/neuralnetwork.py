from network.activations import NNFunction, MSE
from network.layers import *

class Network:
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
