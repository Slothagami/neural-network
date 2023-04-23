import ctypes
from ctypes import c_int, c_uint, c_double
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
        width, height = np_array.shape 
        mat = new_matrix(width, height)
        data = np_array.astype(np.float64).flatten()
        mat.contents.data = data.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        return mat
    
    @staticmethod
    def to_numpy(mat_ptr):
        mat = mat_ptr.contents
        data_ptr = ctypes.cast(mat.data, ctypes.POINTER(ctypes.c_double * mat.size))
        data = np.ctypeslib.as_array(data_ptr.contents)
        return np.reshape(data, (mat.height, mat.width))

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
    ("forward",  LayerFunc),
    ("backward", GradFunc),
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
    ("loss", LossFunc),
    ("num_layers", c_int),
]

make_network = lib.make_network
make_network.argtypes = [LossFunc]
make_network.restype  = net

net_add_layer = lib.net_add_layer
net_add_layer.argtypes = [net, layer]
net_add_layer.restype  = None

net_train = lib.net_train
net_train.argtypes = [net, DispErrorFunc, ctypes.POINTER(mat), ctypes.POINTER(mat), c_int, c_int, c_double, c_int, c_int]
net_train.restype = None

class Net:
    def __init__(self, loss, loss_disp):
        self.network = make_network(loss)
        self.loss_disp = loss_disp

    def add(self, layer):
        net_add_layer(self.network, layer)

    def train(self, batch, labels, samples, epochs, lr, batch_size, print_interval=100):
        # need to convert arrays into mat** type?
        net_train(self.network, self.loss_disp, batch, labels, samples, epochs, lr, print_interval, batch_size)

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
mse.argtypes = [mat, mat]
mse.restype = c_double

mse_grad = lib.mse_grad
mse_grad.argtypes = [mat, mat]
mse_grad.restype = mat

cce = lib.cce
cce.argtypes = [mat, mat]
cce.restype = c_double

cce_grad = lib.cce_grad
cce_grad.argtypes = [mat, mat]
cce_grad.restype = mat
