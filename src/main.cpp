#include "tensor.h"
#include "linalg.h"
#include "timer.h"
#include <iostream>
using namespace std;
#define R 1

int N = 4;

void test_tensor() {
    Tensor<float> T({5, 1, 1});
    cout << T.size() << endl;
    T.reshape({1, 1, 5});
    T.rand();
    cout << T({0, 0, 1}) << endl;
    cout << T.dims() << endl;
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
    Tensor<int> A = Tensor<int>::random({N, N});
    Tensor<int> B = Tensor<int>::random({N, N});
    Tensor<int> C = A.dot(B);
    //A.print(); B.print();
    C.print();
    C.empty();
    linalg::matmul(A, B, &C);
    C.print();
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
    linalg::matmul(A, B, &C);
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
    //A * 0.1;
    A.reduce();
    A.print();
    cout << A.det() << endl;
}

void test_trace() {
    Tensor<int> A({3, 3});
    A.rand();
    A.print();
    cout << A.trace() << endl;
    Tensor<int> B({2, 3});
    B.rand();
    B.print();
    cout << B.trace() << endl;
}

void test_det() {
    Tensor<float> A({4, 4});
    A.load({1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, 1});
    cout << A.det() << endl;
    A.reduce();
    A.print();
    cout << A.size() << endl;
}

void test_constructors() {
    Tensor<float> A = Tensor<float>::eye(3);
    A.print();
    Tensor<float> B = Tensor<float>::random({3, 3});
    B.print();
    Tensor<float> C = B.dot(A);
    C.print();
}

void test_swappp() {
    double start, end;
	Tensor<int> A = Tensor<int>::random({N, 1});
	Tensor<int> B = Tensor<int>::eye(N);
    //A.print();
    start = get_time();
    B({0, 0}) = 0;
    B({0, R}) = 1;
    B({R, R}) = 0;
    B({R, 0}) = 1;
    A = B.dot(A);
    end = get_time();
    cout << "EXEC TIME MATMUL: " << end-start << endl;
    //A.print();
    start = get_time();
    int tmp = A({0, 0});
    A({0, 0}) = A({R, 0});
    A({R, 0}) = tmp;
    end = get_time();
    cout << "EXEC TIME SWAP: " << end-start << endl;
    //A.print();
}

void test_outer() {
    Tensor<float> A({3});
    Tensor<float> B({3});
    //B = B.transpose();
    A.load({0, 9, 4}); B.load({4,0,2});
    //A.dot(B).print(); 
    //B = B.transpose();
    //A.reshape({5}); B.reshape({5});
    A.outer(B).transpose().print();
}

void test_inv() {
    Tensor<float> A({3, 3});
    A.load({1, 1, -1, 1, -1, 1, -1, 1, 1});
    A.inv().print();

    A.rand();
    A.print();
    A.inv().print();

}

void test_solve() {
    Tensor<float> A({3, 3});
    A.load({1, 2, -2, 2, 1, -1, 2, -1, 2});
    Tensor<float> B({3, 1});
    B.load({-1, 1, 6});
    linalg::solve(A, B).print();
    cout << "random test" << endl;
    A.rand();
    B.rand();
    A.print(); B.print();
    linalg::solve(A, B).print();
}

void test_cross() {
    Tensor<int> A = Tensor<int>::random({N});
    Tensor<int> B = Tensor<int>::random({N});
    Tensor<int> C = linalg::cross(A, B);
    A.print(); B.print(); C.print();
}

void test_OLS() {
	Tensor<float> A({3, 2});
	Tensor<float> B({3, 1});
	A.load({1, 2, 1, 1, 2, 1});
	B.load({1, 1, 1});
	(A.transpose().dot(A).inv()).dot(A.transpose()).dot(B).print();
}

void test_quadratic() {
	/* Given data representing points in R2 
	 * X : -1, 0, 1, 2
	 * Y :  0, 0, 2, 4
	 * Find the quadratic polynomial that minimizes quadratic error
	 * p = a + bx + cx^2
	 */
	Tensor<float> A({4, 3});
	A.load({1, -1, 1, 1, 0, 0, 1, 1, 1, 1, 2, 4});
	Tensor<float> Y({4, 1});
	Y.load({0,0,2,4});
	(A.transpose().dot(A).inv()).dot(A.transpose()).dot(Y).print(); // {a, b, c}
	
	A.load({1, -1.5077, 2.2731, 1, 1.1625, 1.3514, 1, -0.9541, 0.9103, 1, 0, 0});
	Y.load({-1.8484, 0.5296, 0.9886, 3});
	(A.transpose().dot(A).inv()).dot(A.transpose()).dot(Y).print();
	
}

void test_norm() {
    Tensor<int> A = Tensor<int>::random({N});
    A.print();
    cout << "Norm: " << A.norm() << endl;
    Tensor<int> B = Tensor<int>::random({3, 3});
    B.print();
    cout << B.norm() << endl;
}

void test_sum() {
    Tensor<int> A = Tensor<int>::random({N, 1});
    Tensor<int> B = Tensor<int>::random({N, 1});
    A.print(); B.print();
    Tensor<int> C = A+B;
    C.print();
    C = A-B;
    C.print();

    A = Tensor<int>::random({N, N});
    B = Tensor<int>::random({N, N});
    A.print(); B.print();
    C = A+B;
    C.print();
    C = A-B;
    C.print();
}

void test_distance() {
    Tensor<int> A({3});
    Tensor<int> B({3});
    A.load({2, -1, 3});
    B.load({3, 1, 4});
    cout << linalg::distance(A, B) << endl;
}

void test_fourier() {
    Tensor<float> a({3});
    a.load({1, 0, -1});
    Tensor<float> b({3});
    b.load({1, -1, 0});
    cout << linalg::FourierCoeff(a, b) << endl;

    Tensor<float> basis({3, 2});
    basis.load({1, 1, -1, 0, 0, -1});
    Tensor<float> ortonormal_basis = linalg::GSO(basis);
    ortonormal_basis.print();
    Tensor<float> v({3, 1}); v.load({1, 2, 3});
    ortonormal_basis.dot(ortonormal_basis.transpose()).dot(v).print();
    linalg::project(ortonormal_basis, v).print();
    cout << "P^2 = P" << endl;
    ortonormal_basis.dot(ortonormal_basis.transpose()).dot(ortonormal_basis.dot(ortonormal_basis.transpose())).print();
    ortonormal_basis.dot(ortonormal_basis.transpose()).print();
}

int main() {
    srand(time(0));
    cin >> N;
    test_fourier(); 
}
