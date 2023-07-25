#ifndef OPS_H
#define OPS_H

namespace linalg {

    void matmul(Tensor<float> A, Tensor<float> B, Tensor<float> *C);
    Tensor<float> solve(Tensor<float> A, Tensor<float> B);
    
}

#endif