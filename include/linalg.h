#ifndef LINALG_H
#define LINALG_H

namespace linalg {
    
    Tensor<float> solve(Tensor<float> A, Tensor<float> B);

    template <class T>
    void matmul(Tensor<T> A, Tensor<T>B, Tensor<T> *C) {
        for (int i=0; i<A.shape()[0]; i++) 
            for (int j=0; j<B.shape()[1]; j++) 
                for (int k=0; k<A.shape()[1]; k++)
                    (*C)[i*A.shape()[1]+k] += A[i*B.shape()[1]+j]*B[j*A.shape()[1]+k]; 
    }
    
    template <class T>
    Tensor<T> cross(Tensor<T> A, Tensor<T> B) {
        assert(A.dims()==1 && B.dims()==1 && A.shape()[0]==3 && B.shape()[0]==3);
        Tensor<T> output({3});
        output[0] = A[1] * B[2] - A[2] * B[1];
        output[1] = A[2] * B[0] - A[0] * B[2];
        output[2] = A[0] * B[1] - A[1] * B[0];
        return output;
    }

}

#endif