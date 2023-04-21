import ctypes
import numpy as np

lib = ctypes.CDLL("./build/network.so")

class Matrix(ctypes.Structure):
    _fields_ = [
        ("data",   ctypes.POINTER(ctypes.c_double)),
        ("width",  ctypes.c_uint),
        ("height", ctypes.c_uint),
        ("size",   ctypes.c_uint)
    ]

new_matrix = lib.new_matrix
new_matrix.argtypes = [ctypes.c_uint, ctypes.c_uint]
new_matrix.restype  =  ctypes.POINTER(Matrix)

def mat_from_numpy(np_array):
    width, height = np_array.shape 
    mat = new_matrix(width, height)
    data = np_array.astype(np.float64).flatten()
    mat.contents.data = data.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    return mat

def mat_to_numpy(mat_ptr):
    mat = mat_ptr.contents
    data_ptr = ctypes.cast(mat.data, ctypes.POINTER(ctypes.c_double * mat.size))
    data = np.ctypeslib.as_array(data_ptr.contents)
    return np.reshape(data, (mat.height, mat.width))
