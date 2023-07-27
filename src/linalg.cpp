#include "tensor.h"
#include "linalg.h"
#include "math.h"

Tensor<float> linalg::solve(Tensor<float> A, Tensor<float> B) {
    assert(A.det()!=0);
    A = A.inv();
    return A.dot(B);
}

float linalg::FourierCoeff(Tensor<float> v, Tensor<float> b) {
    assert(v.dims()==1 && v.shape()==b.shape());
    return v.dot(b)[0] / pow(b.norm(), 2);
}

Tensor<float> linalg::GSO(Tensor<float> v) {
    assert(v.dims()==2);
    Tensor<float> base(v.shape());
    for (int i=0; i<v.shape()[1]; i++) {
        Tensor<float> tmp({v.shape()[0]});
        Tensor<float> vi({v.shape()[0]});
        for (int j=0; j<v.shape()[0]; j++) {
            tmp[j] = v[j*v.shape()[1]+i];
            vi[j] = v[j*v.shape()[1]+i];
        }
        for (int j=0; j<i; j++) {
            Tensor<float> tmp_bi({v.shape()[0]});
            for (int k=0; k<v.shape()[0]; k++)
                tmp_bi[k] = base[k*v.shape()[1]+j];
            tmp_bi * linalg::FourierCoeff(vi, tmp_bi);
            tmp = tmp - tmp_bi;
        }
        tmp * (1/tmp.norm());
        for (int j=0; j<v.shape()[0]; j++)
            base[j*v.shape()[1]+i] = tmp[j];
    }
    return base;
}

Tensor<float> linalg::project(Tensor<float> orth_basis, Tensor<float> v) {
    // Given a orthonormal basis of a vector space Q, the matrix associated with the projection
    // is P = Q*Q^T
    return orth_basis.dot(orth_basis.transpose()).dot(v);
}