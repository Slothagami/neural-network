from network.neuralnet   import *
from network.activations import *
from network.eval        import *

from keras.datasets import mnist
from keras.utils import to_categorical

print("Loading Data...")

# Reshape Data
(train_batch, train_labels), (test_batch, test_labels) = mnist.load_data()
train_batch = train_batch.reshape(train_batch.shape[0], 1, 28, 28).astype("float32")
test_batch  = test_batch .reshape(test_batch .shape[0], 1, 28, 28).astype("float32")
train_labels = to_categorical(train_labels)
test_labels  = to_categorical(test_labels )

# Train
parameters = {
    "lr": .525,
    "lr_falloff": .9,
    "loss": CategoricalCrossEntropy,
    # "file": "models/mnist_conv.npy" # remove to train fresh model and don't save
}
nn = NeuralNet(**parameters) # Gets ~88% acc w/ 5 epochs
print(nn.lr, nn.lr_falloff)

channels = 1 # mnist is a grayscale dataset
nn.layers = [
    ConvLayer((1, 28, 28), 3, channels),
    ReshapeLayer((channels, 26, 26), (1, channels * 26 * 26)),
    ActivationLayer(Tanh),

    FCLayer(channels * 26 * 26, 100),
    ActivationLayer(Tanh),
    FCLayer(100, 50),
    ActivationLayer(Tanh),
    FCLayer(50, 10),

    Softmax()
]

nn.load()
calc_accuracy(nn, test_batch, test_labels, print_acc=True)

print("Beginning Training...")
error_graph = nn.train(train_batch, train_labels, 5, 64)

calc_accuracy(nn, test_batch, test_labels, print_acc=True)
show_error_graph(error_graph, ylim=2)
