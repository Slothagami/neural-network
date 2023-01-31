from network.neuralnet   import *
from network.activations import *
from network.eval        import calc_accuracy

from keras.datasets import mnist
from keras.utils import np_utils

print("Loading Data...")
(train_batch, train_labels), (test_batch, test_labels) = mnist.load_data()

# Reshape Data
train_batch = train_batch.reshape(train_batch.shape[0], 1, 28*28).astype("float32")
test_batch  = test_batch .reshape(test_batch .shape[0], 1, 28*28).astype("float32")

train_batch /= 255 # normalize pixel values
test_batch  /= 255

train_labels = np_utils.to_categorical(train_labels) # reformat to vector labels
test_labels  = np_utils.to_categorical(test_labels )


# Train
nn = NeuralNet(lr=.01)
nn.config((28*28, 100, 50, 10), Tanh)
# Gets ~80% acc w/ 5 epochs

print("Beginning Training...")
nn.train(train_batch, train_labels, 5)

nsamples = 2000
calc_accuracy(nn, test_batch[:nsamples], test_labels[:nsamples], print_acc=True)
