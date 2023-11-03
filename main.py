#!/home/carlitos/Documents/Projects/Python/gpu/venv/bin/python3

import numpy as np
from numba import cuda

# Define a function to perform matrix multiplication on the GPU
@cuda.jit
def gpu_matrix_multiply(a, b, result):
    i, j = cuda.grid(2)
    if i < result.shape[0] and j < result.shape[1]:
        result[i, j] = 0
        for k in range(a.shape[1]):
            result[i, j] += a[i, k] * b[k, j]

# Generate random matrices
matrix_size = 1000
a = np.random.rand(matrix_size, matrix_size)
b = np.random.rand(matrix_size, matrix_size)

# Allocate memory for the result
result = np.zeros((matrix_size, matrix_size))

# Configure the threads and blocks
threadsperblock = (16, 16)
blockspergrid_x = (matrix_size + threadsperblock[0] - 1) // threadsperblock[0]
blockspergrid_y = (matrix_size + threadsperblock[1] - 1) // threadsperblock[1]
blockspergrid = (blockspergrid_x, blockspergrid_y)

# Perform matrix multiplication on GPU
gpu_matrix_multiply[blockspergrid, threadsperblock](a, b, result)

print("Matrix multiplication result shape:", result.shape)

