from network import Matrix
import numpy as np

arr = np.arange(9).reshape((3, 3))
mat = Matrix.from_numpy(arr)
arr = Matrix.to_numpy(mat)
print(arr)
