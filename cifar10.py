from network.neuralnet   import *
from network.activations import *
from network.eval        import *

from keras.datasets import cifar10
from keras.utils import to_categorical

print("Loading Data...")

# Reshape Data
(train_batch, train_labels), (test_batch, test_labels) = cifar10.load_data()
train_batch = train_batch.reshape(train_batch.shape[0], 3, 32, 32).astype("float32")
test_batch  = test_batch .reshape(test_batch .shape[0], 3, 32, 32).astype("float32")
train_labels = to_categorical(train_labels)
test_labels  = to_categorical(test_labels )

# Train
parameters = {
    "lr": .525,
    "loss": CategoricalCrossEntropy,
    "file": "models/cifar10.npy" # remove to train fresh model and don't save
}
nn = NeuralNet(**parameters)
print(nn.lr, nn.lr_falloff)

channels = 3
nn.layers = [
    ConvLayer((1, 32, 32), 3, channels),
    ReshapeLayer((channels, 30, 30), (1, channels * 30 * 30)),
    ActivationLayer(Tanh),

    FCLayer(channels * 30 * 30, 100),
    ActivationLayer(Tanh),
    FCLayer(100, 50),
    ActivationLayer(Tanh),
    FCLayer(50, 10),

    Softmax()
]

nn.load()
calc_accuracy(nn, test_batch, test_labels, print_acc=True)

print("Beginning Training...")
error_graph = nn.train(train_batch, train_labels, 1, 64)

calc_accuracy(nn, test_batch, test_labels, print_acc=True)
show_error_graph(error_graph, ylim=5)
