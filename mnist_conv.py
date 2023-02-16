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
# nn = NeuralNet(lr=.00001, loss=CategoricalCrossEntropy) # Gets ~62% acc w/ 5 epochs
nn = NeuralNet(lr=.000008, loss=CategoricalCrossEntropy) # Gets ~62% acc w/ 5 epochs

depth = 2
print(nn.lr, depth)
nn.layers = [
    ConvLayer((1, 28, 28), 3, depth),
    ReshapeLayer((depth, 26, 26), (1, depth * 26 * 26)),
    ActivationLayer(ReLU),

    FCLayer(depth * 26 * 26, 100),
    ActivationLayer(ReLU),
    FCLayer(100, 50),
    ActivationLayer(ReLU),
    FCLayer(50, 10),

    Softmax()
]

print("Beginning Training...")
error_graph = nn.train(train_batch, train_labels, 5)

calc_accuracy(nn, test_batch, test_labels, print_acc=True)
show_error_graph(error_graph, ylim=25)
