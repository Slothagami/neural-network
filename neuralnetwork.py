from activations import *
import numpy as np 
from visualize import show_output_space

class NeuralNetwork:
    def __init__(self, layers, /, learn_rate=.01, activation=sigmoid, activation_prime=sigmoid_prime):
        self.layers = layers
        self.lr = learn_rate
        self.activation = activation
        self.activation_prime = activation_prime

        self.activations = []
        self.weights = []
        self.biases  = np.random.rand(len(layers))

        for i, layer in enumerate(layers[:-1]):
            self.weights.append( # +1 to account for bias node
                np.random.rand(layer, layers[i + 1]) )

    def predict(self, input):
        self.activations = []
        out = np.array(input)

        for layer, bias in zip(self.weights, self.biases): 
            out = np.dot(out, layer) + bias
            self.activations.append(out)
            out = self.activation(out)

        return out

    def feed(self, input, target):
        out = self.predict(input)

        # Mean Squared Error
        error = np.sum( np.square(out - target) )

        # Error Prime
        error_prime = 2 * (out - target)

        for weight, activation, bias in zip(
            reversed(self.weights), 
            reversed(self.activations),
            reversed(self.biases)
        ):
            weight -= error_prime * self.lr
            bias   -= error_prime * self.lr
            error_prime = np.dot((self.activation_prime(activation) * activation).T, error_prime)

        print(f"Error: {error:.4f} → {target}? → {out}")


if __name__ == "__main__":
    nn = NeuralNetwork((2, 1), learn_rate=.1)
    for n in range(10_000_000):
        if n % 500_000 == 0:
            nn.feed([1, 1], 1)
            nn.feed([1, 0], 1)
            nn.feed([0, 1], 1)
            nn.feed([0, 0], 0)

    print("\nFinal Predictions")
    print(nn.predict([1,1]))
    print(nn.predict([1,0]))
    print(nn.predict([0,1]))
    print(nn.predict([0,0]))

    show_output_space(nn)
