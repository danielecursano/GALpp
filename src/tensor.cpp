#include "tensor.h"
#include "assert.h"
#include <vector>
#include <iostream>
#include <cstdlib>

template<class T>
Tensor<T>::Tensor(std::vector<int>shape) : shape(shape) {
    int size = 1;
    for (int s: shape) {
        size *= s;
    }   
    data.resize(size);
}

template<class T>
bool Tensor<T>::check_index(const std::vector<int>& index) const{
    for (int i=0; i<shape.size(); i++) {
        if (index[i] >= shape[i]) 
            return false;
    } 
    return true;
}

template<class T>
int Tensor<T>::calculate_index(const std::vector<int>& index) const {
    assert(this->check_index(index));
    int output = 0;
    int dims = 1;
    for (int i = index.size() - 1; i >= 0; --i) {
        output += index[i] * dims;
        dims *= shape[i];
    }
    return output;
}

template<class T>
T& Tensor<T>::operator()(const std::vector<int>& index) {
    return data[calculate_index(index)];
}

template<class T>
const T& Tensor<T>::operator()(const std::vector<int>& index) const {
    return data[calculate_index(index)];
}

template<class T>
void Tensor<T>::operator*(const T scalar) {
    for (int i=0; i<data.size(); i++) {
        data[i] *= scalar;
    }
}

template<class T>
Tensor<T> Tensor<T>::dot(const Tensor<T> &other) {
    if ((*this).rank() == 1 && other.shape.size()==1 && data.size()==other.data.size()) {
        Tensor<T>output({1});
        T out_data = 0;
        for (int i=0; i<data.size(); i++) {
            out_data += data[i] * other({i});
        } 
        output.data[0] = out_data;
        return output; 
    }

    if ((*this).rank() == 2 && other.shape.size()==2 && shape[1]==other.shape[0]) {
        Tensor<T>output({shape[0], other.shape[1]});
        for (int i=0; i<shape[0]; i++) 
            for (int j=0; j<other.shape[1]; j++) 
                for (int k=0; k<shape[1]; k++)
                    output({i, j}) += (*this)({i, k}) * other({k, j});
        return output;
    }
    assert(0==1);
}

template<class T>
void Tensor<T>::reshape(std::vector<int> new_shape) {
    int new_size = 1;
    for (int s: new_shape) {
        new_size *= s;
    }
    assert(new_size==this->size());
    shape = new_shape;
}

template<>
void Tensor<float>::rand() {
    for (int i=0; i<data.size(); i++) 
        data[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
}

template<>
void Tensor<int>::rand() {
    for (int i=0; i<data.size(); i++) {
        float tmp = (static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX));
        data[i] = 10*tmp;
    }
}

template<class T>
int Tensor<T>::size() {
    return data.size();
}

template<class T>
int Tensor<T>::rank() {
    return shape.size();
}

template<class T>
void Tensor<T>::print() {
    if (shape.size()==1) {
        std::cout << "Vector [";
        for (T i: data)
            std::cout << i << ", ";
        std::cout << "]" << std::endl;
        return;
    }
    if (shape.size()==2) {
        std::cout << "Matrix [" << std::endl;
        for (int i=0; i<shape[0]; i++) {
            std::cout << "[";
            for (int j=0; j<shape[1]; j++) {
                std::cout << (*this)({i, j}) << ", ";
            }
            std::cout << "]" << std::endl;
        }
        std::cout << "]" << std::endl; 
    }
}

template class Tensor<int>;
template class Tensor<float>;