#include "tensor.h"
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
    cout << "primo vettore: ";
    //one.print();
    for (int i=0; i<5; i++) 
        cout << one({i}) << ", ";
    cout << endl;
    Tensor<int> second({5});
    second.rand();
    cout << "secondo vettore: ";
    //second.print();
    for (int i=0; i<5; i++)
        cout << second({i}) << ", ";
    cout << endl;
    Tensor<int> product = one.dot(second);
    //cout << "risultato: ";
    //product.print();
    for (int i=0; i<product.size(); i++) {
        cout << product({i}) << endl;
    }
    cout << "\n\n";
    one.print();
}

void test_matmul() {
    Tensor<int> a({2, 3});
    a.rand();
    a.print();
    Tensor<int> b({3, 2});
    b.rand();
    b.print();
    Tensor<int> c({2, 2});
    c = a.dot(b);
    c.print();
}

int main() {
    srand(time(0));
    test_matmul();
}
