import matplotlib.pyplot as plt
import numpy as np
from activations import sigmoid, heaviside

class Perceptron:
    def __init__(self, /, lr=.1, activation=heaviside):
        self.weights = np.random.rand(3)
        self.lr = lr
        self.activation = activation

    def feed(self, input, target):
        input = np.append(input, 1) # add bias component
        out = np.dot(input, self.weights)
        out = self.activation(out)

        error = target - out
        self.weights += error * input * self.lr

    def predict(self, input):
        input = np.append(input, 1) # add bias component
        out   = np.dot(input, self.weights)
        return self.activation(out)

if __name__ == "__main__":
    nn = Perceptron(activation=sigmoid)

    # train network
    for _ in range(1000):
        nn.feed([1, 1], 1)
        nn.feed([1, 0], 1)
        nn.feed([0, 1], 1)
        nn.feed([0, 0], 0)

    print("Final Predictions:")
    inputs = [[1, 1], [1, 0], [0, 1], [0, 0]]
    for input in inputs:
        print(f"{input}: {nn.predict(input):.6f}")

    print(f"\nFinal Weights:\n{nn.weights}")

    # Image of predictions for all values
    image_size = 100
    image = np.zeros((image_size, image_size))
    for x in range(image_size):
        for y in range(image_size):
            image[x, y] = nn.predict([x / image_size, y / image_size])

    plt.imshow(image, "Greys")
    plt.show()
    
