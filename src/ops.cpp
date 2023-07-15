#include "tensor.h"
#include "ops.h"

void matmul(Tensor<float> A, Tensor<float>B, Tensor<float> *C) {
    for (int i=0; i<A.shape[0]; i++) 
        for (int j=0; j<B.shape[1]; j++) 
            for (int k=0; k<A.shape[1]; k++)
                (*C)[i*A.shape[1]+k] += A[i*B.shape[1]+j]*B[j*A.shape[1]+k]; 
}