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

layer = ctypes.POINTER(Layer)
mat   = ctypes.POINTER(Matrix)
LayerFunc = ctypes.CFUNCTYPE(mat, layer, mat)
GradFunc  = ctypes.CFUNCTYPE(mat, layer, mat, c_double)

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

# C Function Definitions
new_matrix = lib.new_matrix
new_matrix.argtypes = [c_uint, c_uint]
new_matrix.restype  =  mat

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
