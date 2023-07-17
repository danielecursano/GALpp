#include "tensor.h"
#include "ops.h"
#include <iostream>
using namespace std;

void test_tensor() {
    Tensor<float> T({5, 1, 1});
    cout << T.size() << endl;
    T.reshape({1, 1, 5});
    T.rand();
    cout << T({0, 0, 1}) << endl;
    cout << T.rank() << endl;
    T*2.0;
    cout << T({0, 0, 1}) << endl;
    //T.print();  
    /*
    Tensor<int> test({1});
    cout << test.rank() << endl;
    cout << test.size() << endl;
    test.reshape({1, 1, 1});
    cout << test.rank() << endl;
    */
}

void test_inner() {
    Tensor<int> one({5});
    one.rand();
    one.print();
    Tensor<int> second({5});
    second.rand();
    second.print();
    Tensor<int> product = one.dot(second);
    product.print();
}

void test_matmul() {
    #define N 100
    Tensor<int> a({N, N});
    a.rand();
    //a.print();
    Tensor<int> b({N, N});
    b.rand();
    //b.print();
    Tensor<int> c({N, N});
    c = a.dot(b);
    //c.print();
}

void test_transpose() {
    Tensor<int> a({5});
    a.rand();
    a.print();
    Tensor<int>b({5});
    b = a.transpose();
    b.print();
    cout << endl;
    Tensor<int> A({3, 2});
    A.rand();
    A.print();
    Tensor<int> B({2, 3});
    B = A.transpose();
    B.print();
}

void test() {
    Tensor<int> A({5, 5});
    Tensor<int> B({5, 5});
    A.rand();
    B.rand();
    A.reshape({25, 5});
    B.reshape({5, 25});
    Tensor<int> C({25, 25});
    C = A.dot(B);
    C.print();
}

void test_ops() {
    Tensor<float> A({5, 5});
    Tensor<float> B({5, 5});
    Tensor<float> C({5, 5});
    A.rand();
    B.rand();
    A.print();
    B.print();
    matmul(A, B, &C);
    C.print();
}

void test_print() {
    Tensor<int> A({2, 3, 4});
    Tensor<int> B({5});
    Tensor<int> C({4, 4});
    A.rand(); B.rand(); C.rand();
    B.print();
    C.print();
    A.print();

    Tensor<int> D({});
    D.print();
    cout << "\n\n";
    Tensor<int> E({2, 2, 2, 2, 2});
    E.rand();
    E.print();
}

void test_flatten() {
    Tensor<int> A({2, 4});
    A.rand();
    A.print();
    Tensor<int> B = A.flatten();
    B.print();
}

void test_transposereshape() {
    Tensor<int> A({2, 3, 4});
    A.rand();
    A.print();
    A.reshape({4, 3, 2});
    A.print();
}

void test_reduce() {
    Tensor<float> A({3, 3});
    A.rand();
    A*10;
    A.print();
    A * 0.1;
    A.reduce();
    A.print();
}

int main() {
    srand(time(0));
    test_reduce();
}
