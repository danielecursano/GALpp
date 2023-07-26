#ifndef LINALG_H
#define LINALG_H

namespace linalg {
    
    void matmul(Tensor<float> A, Tensor<float> B, Tensor<float> *C);
    Tensor<float> solve(Tensor<float> A, Tensor<float> B);
    
}

#endif