import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))



def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1 - np.tanh(x) ** 2



def MSE(prediction, target):
    return np.mean(np.square(target - prediction))

def MSE_prime(prediction, target):
    return 2 * (prediction - target) / target.size


def heaviside(x):
    return 1 if x > 0 else 0
