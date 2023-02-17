from network.activations import *
from scipy import signal
from skimage.measure import block_reduce
import numpy as np

class NeuralNet:
    def __init__(self, file=None, /, loss: NNFunction = MSE, lr=.1, lr_falloff=1):
        self.layers = []
        self.file = file

        self.lr = lr
        self.lr_falloff = lr_falloff
        self.loss = loss

    def add(self, layer): self.layers.append(layer)

    def config(self, layer_sizes, activation: NNFunction = Tanh, /, layertypes=None):
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

    def train(self, samples, labels, epochs, minibatch_size=100):
        error_graph = [] # for plotting error
        samples = np.array(samples)
        labels  = np.array(labels)
        
        for epoch in range(epochs):
            disp_error = 0 

            # randomize order of samples
            dataset = list(zip(samples, labels))
            np.random.shuffle(dataset)

            considered_samples = 0
            for sample, label in dataset:
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

                # update batch if at end of 
                considered_samples += 1
                if considered_samples >= minibatch_size:
                    considered_samples = 0
                    for layer in self.layers: layer.update()

            # Calc Average Error
            disp_error /= len(samples)

            error_graph.append(disp_error)
            print(f"Epoch: {epoch + 1}, Error: {disp_error}")
            self.save(self.file)

            self.lr *= self.lr_falloff
        return error_graph

    def save(self, filen):
        # TODO: Save conv layers
        if filen == None: return
        weights, biases = [], []
        for layer in self.layers:
            if isinstance(layer, FCLayer):
                weights.append(layer.weights)
                biases.append(layer.bias)

        np.save(filen, (weights, biases), allow_pickle=True)

    def load(self):
        weights, biases = np.load(self.file, allow_pickle=True)

        ind = 0
        for layer in self.layers:
            if isinstance(layer, FCLayer):
                layer.weights = weights[ind]
                layer.bias    = biases[ind]
                ind += 1

# utility #
def rand(*args):
    return (np.random.rand(*args) - .5) * 2

def norm(*args):
    return np.random.randn(*args)

# Layers #
class Layer:
    def __init__(self): self.input = self.output = None 
    def forward(self):  raise NotImplementedError
    def backprop(self): raise NotImplementedError
    def update(self): pass

class FCLayer(Layer):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weights = rand(in_size, out_size)
        self.bias    = rand(1, out_size)

        self.delta_weights = np.zeros_like(self.weights)
        self.delta_bias    = np.zeros_like(self.bias)
        self.delta_n = 0 # number of samples being averaged over (for division in average)

    def forward(self, input):
        self.input = input 
        self.output = np.dot(input, self.weights) + self.bias 
        return self.output

    def backprop(self, out_error, lr):
        # Calc Error
        in_error = np.dot(out_error, self.weights.T)
        weights_error = np.dot(self.input.T, out_error)

        # Update Weights
        self.delta_weights += lr * weights_error
        self.delta_bias    += lr * out_error
        self.delta_n += 1

        return in_error
    
    def update(self):
        # apply the average if the in backprop since last call
        self.weights -= self.delta_weights / self.delta_n
        self.bias    -= self.delta_bias    / self.delta_n

        # reset everything
        self.delta_n = 0
        self.delta_weights.fill(0) # reuse array to avoid allocating memory
        self.delta_bias   .fill(0)

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

class Softmax(Layer):
    def forward(self, input): 
        # shift input (has no effect on output, but prevents overflow in exp)
        input -= np.max(input)
        exp = np.exp(input)
        self.output = exp / np.sum(exp)
        return self.output

    def backprop(self, out_error, lr):
        out_size = np.size(self.output)
        tmp = np.vstack([self.output] * out_size)
        return np.dot(
            tmp * (np.identity(out_size) - tmp.T),
            out_error.T
        ).reshape((1, out_size))

class ConvLayer(Layer):
    def __init__(self, in_shape, kernel_size, depth):
        in_depth, in_height, in_width = in_shape
        self.in_shape  = in_shape
        self.depth     = depth
        self.in_depth = in_depth

        self.out_shape = (
            depth,
            in_height - kernel_size + 1,
            in_width  - kernel_size + 1
        )
        self.kernels_shape = (depth, in_depth, kernel_size, kernel_size)

        self.kernels = rand(*self.kernels_shape)
        self.biases  = rand(*self.out_shape)

        self.delta_kernels = np.zeros_like(self.kernels)
        self.delta_biases  = np.zeros_like(self.biases)
        self.delta_n = 0 # number of samples being averaged over (for division in average)

    def forward(self, input):
        self.input = input
        self.output = np.copy(self.biases) # Equal to adding them

        for depth in range(self.depth):
            for in_depth in range(self.in_depth):
                self.output[depth] += signal.correlate2d(
                    self.input[in_depth], 
                    self.kernels[depth, in_depth], 
                    "valid"
                )

        return self.output

    def backprop(self, out_gradient, lr):
        kernel_delta = np.zeros(self.kernels_shape)
        input_delta  = np.zeros(self.in_shape)

        for depth in range(self.depth):
            for in_depth in range(self.in_depth):
                kernel_delta[depth, in_depth] = signal.correlate2d(
                    self.input[in_depth], 
                    out_gradient[depth], 
                    "valid"
                )
                input_delta[in_depth] = signal.convolve2d(
                    out_gradient[depth],
                    self.kernels[depth, in_depth], 
                    "full"
                )

                self.delta_kernels += lr * kernel_delta
                self.delta_biases  += lr * out_gradient
                self.delta_n += 1

                return input_delta
            
    def update(self):
        self.kernels -= self.delta_kernels / self.delta_n
        self.biases  -= self.delta_biases  / self.delta_n

        # reset everything
        self.delta_n = 0
        self.delta_kernels.fill(0)
        self.delta_biases .fill(0)

class ReshapeLayer(Layer):
    def __init__(self, in_shape, out_shape):
        self.in_shape = in_shape
        self.out_shape = out_shape

    def forward(self, input):
        return np.reshape(input, self.out_shape)

    def backprop(self, out_gradient, lr):
        return np.reshape(out_gradient, self.in_shape)

class PoolingLayer(Layer):
    # Note: Probably only works with 2d arrays atm due to block_reduce taking n dimensional blocks, across multiple color channels
    def __init__(self, kernel_size=2, func=np.max):
        super.__init__()
        self.kernel_size = kernel_size
        self.func = func
        
    def forward(self, img): 
        self.input = img
        self.output = block_reduce(img, (self.kernel_size, self.kernel_size), self.func)
        return self.output
    
    def backprop(self, out_gradient, lr):
        width, height = self.input.shape
        gradient = np.zeros_like(self.input)
        
        for x in range(0, width, self.kernel_size):
            for y in range(0, height, self.kernel_size):
                kernel = self.input[x: x+self.kernel_size, y: y+self.kernel_size]
                max_x, max_y = np.unravel_index(np.argmax(kernel), kernel.shape)
                
                grad_kernel = np.zeros_like(kernel)
                grad_kernel[max_x, max_y] = 1
                
                gradient[x:x+self.kernel_size, y: y+self.kernel_size] = grad_kernel
        return gradient * out_gradient
