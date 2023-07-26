#include "tensor.h"
#include "linalg.h"

namespace linalg {

    void matmul(Tensor<float> A, Tensor<float>B, Tensor<float> *C) {
        for (int i=0; i<A.shape()[0]; i++) 
            for (int j=0; j<B.shape()[1]; j++) 
                for (int k=0; k<A.shape()[1]; k++)
                    (*C)[i*A.shape()[1]+k] += A[i*B.shape()[1]+j]*B[j*A.shape()[1]+k]; 
    }

    Tensor<float> solve(Tensor<float> A, Tensor<float> B) {
        assert(A.det()!=0);
        A = A.inv();
        return A.dot(B);
    }

}