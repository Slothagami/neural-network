from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np

def random_order(train, test):
    (train_batch, train_labels), (test_batch, test_labels) = train, test 

    train_index = np.random.permutation(train_batch.shape[0])
    test_index  = np.random.permutation(test_batch .shape[0])

    train_batch  = train_batch[train_index,  :]
    train_labels = train_labels[train_index, :]
    test_batch   = test_batch[test_index,  :]
    test_labels  = test_labels[test_index, :]

    return (train_batch, train_labels), (test_batch, test_labels)

def mnist_fc(nsamples=1000):
    # load mnist, shaped for fully connected network
    print("Loading Data...")

    (train_batch, train_labels), (test_batch, test_labels) = mnist.load_data()

    # Reshape Data
    train_batch = train_batch.reshape(train_batch.shape[0], 1, 28*28).astype("float64")[:nsamples]
    test_batch  = test_batch .reshape(test_batch .shape[0], 1, 28*28).astype("float64")[:nsamples]

    train_batch /= 255 # normalize pixel values
    test_batch  /= 255

    train_labels = np_utils.to_categorical(train_labels)[:nsamples] # reformat to vector labels
    test_labels  = np_utils.to_categorical(test_labels )[:nsamples]

    print("Data Loaded.")
    return random_order((train_batch, train_labels), (test_batch, test_labels))
