import ctypes
from ctypes import c_int, c_uint, c_double, c_void_p
import numpy as np

lib = ctypes.CDLL("./build/network.so")

class Matrix(ctypes.Structure):
    _fields_ = [
        ("data",   ctypes.POINTER(ctypes.c_double)),
        ("width",  ctypes.c_uint),
        ("height", ctypes.c_uint),
        ("size",   ctypes.c_uint)
    ]

    @staticmethod
    def from_numpy(np_array):
        if len(np_array.shape) == 2:
            height, width = np_array.shape
        elif len(np_array.shape) == 1:
            height, width = 1, np_array.shape[0]
        else:
            raise ValueError(f"Numpy array of shape {np_array.shape} not supported for conversion")

        mat = new_matrix(width, height)
        data = np_array.astype(np.float64).flatten()
        mat.contents.data = data.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        return mat
    
    @staticmethod
    def to_numpy(mat_ptr):
        mat = mat_ptr.contents
        data_ptr = ctypes.cast(mat.data, ctypes.POINTER(ctypes.c_double * mat.size))
        data = np.ctypeslib.as_array(data_ptr.contents)
        return np.reshape(data, (mat.width, mat.height))

class Layer(ctypes.Structure):
    pass

class Network(ctypes.Structure):
    pass

layer = ctypes.POINTER(Layer)
mat   = ctypes.POINTER(Matrix)
net   = ctypes.POINTER(Network)
LayerFunc     = ctypes.CFUNCTYPE(mat, layer, mat)
GradFunc      = ctypes.CFUNCTYPE(mat, layer, mat, c_double)
LossFunc      = ctypes.CFUNCTYPE(mat, mat, mat)
DispErrorFunc = ctypes.CFUNCTYPE(c_double, mat, mat)

Layer._fields_ = [
    ("forward",  c_void_p),
    ("backward", c_void_p),
    ("delta_weights", mat),
    ("delta_biases",  mat),
    ("delta_n",       c_int),
    ("weights",       mat),
    ("biases",        mat),
    ("input",         mat),
    ("output",        mat)
]
Network._fields_ = [
    ("layers", ctypes.POINTER(layer)),
    ("loss", c_void_p),
    ("num_layers", c_int),
]

make_network = lib.make_network
make_network.argtypes = [c_void_p]
make_network.restype  = net

free_network = lib.free_network
free_network.argtypes = [net]
free_network.restype  = None

net_add_layer = lib.net_add_layer
net_add_layer.argtypes = [net, layer]
net_add_layer.restype  = None

net_train = lib.net_train
net_train.argtypes = [net, c_void_p, ctypes.POINTER(mat), ctypes.POINTER(mat), c_int, c_int, c_double, c_int, c_int]
net_train.restype = None

test_acc = lib.test_acc
test_acc.argtypes = [net, ctypes.POINTER(mat), ctypes.POINTER(mat), c_int, c_void_p]
test_acc.restype = None

class Net:
    def __init__(self, loss, loss_disp):
        self.network = make_network(loss)
        self.loss_disp = loss_disp

    def add(self, layer):
        net_add_layer(self.network, layer)

    def train(self, training_data, epochs, lr, batch_size, print_interval=100):
        (batch, labels), (test_batch, test_labels) = training_data
        samples = len(batch)
        batch  = Net.batch_to_pointer(batch)
        labels = Net.batch_to_pointer(labels)

        if test_batch is not None and test_labels is not None:
            test_batch  = Net.batch_to_pointer(test_batch)
            test_labels = Net.batch_to_pointer(test_labels)

        net_train(self.network, self.loss_disp, batch, labels, samples, epochs, lr, print_interval, batch_size)
        test_acc(self.network, test_batch or batch, test_labels or labels, samples, self.loss_disp)

    @staticmethod
    def batch_to_pointer(batch):
        # Create an array of mat pointers for each sample
        samples = []
        for sample in batch:
            samples.append(Matrix.from_numpy(sample))

        ptr_arr = (mat * len(samples))()
        for i, samp in enumerate(samples): ptr_arr[i] = samp
        return ptr_arr
    
    def __del__(self):
        free_network(self.network)

# C Function Definitions
new_matrix = lib.new_matrix
new_matrix.argtypes = [c_uint, c_uint]
new_matrix.restype  = mat

FCLayer = lib.FCLayer
FCLayer.argtypes = [c_uint, c_uint]
FCLayer.restype = layer

# Define Layer Types
TanhLayer = lib.TanhLayer
TanhLayer.argtypes = []
TanhLayer.restype = layer

SoftmaxLayer = lib.SoftmaxLayer
SoftmaxLayer.argtypes = []
SoftmaxLayer.restype = layer

ReluLayer = lib.ReluLayer
ReluLayer.argtypes = []
ReluLayer.restype = layer

SigmoidLayer = lib.SigmoidLayer
SigmoidLayer.argtypes = []
SigmoidLayer.restype = layer

# Define Error Functions
mse = lib.mse
mse_grad = lib.mse_grad

cce = lib.cce
cce_grad = lib.cce_grad
