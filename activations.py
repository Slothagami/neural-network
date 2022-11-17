import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


def heaviside(x):
    return 1 if x > 0 else 0
