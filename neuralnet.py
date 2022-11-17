from activations import *
import numpy as np 

class NeuralNetwork:
    def __init__(self, layers, /, lr=.01):
        self.layers = layers
        self.lr = lr 

        self.weights = []
        self.biases = np.random.rand(len(layers))

        for i, layer in enumerate(layers[:-1]):
            self.weights.append( # +1 to account for bias node
                np.random.rand(layer, layers[i + 1]) )

    def predict(self, input):
        out = np.array(input)

        for layer, bias in zip(self.weights, self.biases): 
            out = np.dot(out, layer) + bias

        return out

    def feed(self, input, target):
        out = self.predict(input)

        # Mean Squared Error
        error = np.square(out - target)
        print("Error:", error)

        # Error Prime
        error_prime = 2 * (out - target)

        self.weights[-1] -= error_prime * self.lr
        print(out)


if __name__ == "__main__":
    nn = NeuralNetwork((3, 4, 2))
    # print(nn.predict([2, 3, 4]))
    for _ in range(10): nn.feed([2, 3, 4], [1, 2])
