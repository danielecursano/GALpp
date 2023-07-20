from tinygrad.tensor import Tensor
import numpy as np
import time
import sys

N = int(sys.argv[1])

NUMPY = 0
TINYGRAD = 0

A = np.random.rand(N, N)
B = np.random.rand(N, N)
start = time.time()
C = A@B
NUMPY = time.time() - start

A = Tensor.rand(N, N)
B = Tensor.rand(N, N)
start = time.time()
C = (A.reshape(N, 1, N) * B.permute(1, 0).reshape(1, N, N)).sum(axis=2)
TINYGRAD = time.time() - start

print(f"TINYGRAD EXEC TIME: {TINYGRAD}\nNUMPY EXEC TIME: {NUMPY}")
