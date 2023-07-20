'''
With numpy matmul swapping rows in a matrix is faster than just swapping
indices for matrices of NxN with N ~61/62
The operation to achieve a new matrix is E @ A, where A is the matrix we want
to swap rows and E is the identity matrix with the ones swapped in the target
rows

EXAMPLE: 
    A = [[1, 2, 3],    E = [[0, 1, 0],   E @ A = [[4, 5, 6],
         [4, 5, 6],         [1, 0, 0],            [1, 2, 3],
         [7, 8, 9]]         [0, 0, 1]]            [7, 8, 9]] 
'''


import numpy as np
import sys
import time
from tinygrad.tensor import Tensor

N = int(sys.argv[1])
R = 1
DATA_SWAP_TIME = 0
MATMUL_TIME = 0
TINY_TIME = 0

def swapRows(data, row1, row2):
    for i in range(0, N):
        data[row1*N+i], data[row2*N+i] = data[row2*N+i], data[row1*N+i]
    return data

data = np.random.randint(0, 10, size=(N, N))
start = time.time()
swapped_data = swapRows(data.flatten(), 0, R)
DATA_SWAP_TIME = time.time() - start
print(f"EXEC TIME NORMAL SWAP: {DATA_SWAP_TIME}\n")

'''
e = np.eye(N)
e = e.flatten()
e[0] = 0
e[1] = 1
e[R*N+R] = 0
e[R*N] = 1
e = e.reshape(N, N)
start = time.time()
swapped = e @ data
MATMUL_TIME = time.time() - start
print(f"EXEC TIME MATMUL: {MATMUL_TIME}\n")
#print(swapped.flatten()==swapped_data)
'''
A = Tensor(data, device="METAL")
e = np.eye(N, dtype="int")
e = e.flatten()
e[0] = 0
e[1] = 1
e[R*N+R] = 0
e[R*N] = 1
E = Tensor(e, device="METAL")
E.reshape(N, N)
start = time.time()
swapped_tiny = (E.reshape(N, 1, N)*A.permute(1, 0).reshape(1, N, N)).sum(axis=2)
TINY_TIME = time.time()-start
print(f"EXEC TIME TINYGRAD: {TINY_TIME}")
m = min([TINY_TIME, MATMUL_TIME, DATA_SWAP_TIME])
d = {TINY_TIME: "tinygrad", MATMUL_TIME: "matmul", DATA_SWAP_TIME: "data swap"}
print(d[m])
