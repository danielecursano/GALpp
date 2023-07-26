#include "tensor.h"
#include "linalg.h"

Tensor<float> linalg::solve(Tensor<float> A, Tensor<float> B) {
    assert(A.det()!=0);
    A = A.inv();
    return A.dot(B);
}