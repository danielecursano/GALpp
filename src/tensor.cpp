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
T& Tensor<T>::operator[](int index) {
    return data[index];
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

template<class T>
Tensor<T> Tensor<T>::transpose() {
    if (shape.size()==1) {
        return *this;
    }
    if (shape.size()==2) {
        Tensor<T>output({shape[1], shape[0]});

        for (int row=0; row<shape[0]; row++) {
            for (int col=0; col<shape[1]; col++) {
                output.data[(col*shape[0])+row] = data[row*shape[1]+col];
            }
        }
        return output;
    }
    assert(1==0);
}

template<class T>
void Tensor<T>::load(const std::vector<T>& extern_data) {
    data = extern_data;
}

template<class T>
Tensor<T> Tensor<T>::flatten() {
    Tensor<T>output({(*this).size()});
    output.load(data);
    return output;
}

template<class T>
void Tensor<T>::reduce() {
    assert(shape.size()==2);
    for (int i=0; i<shape[0]; i++) {
        T pivot = data[i*shape[0]];
        int index = 0;
        if (pivot==0) {
            bool found = 0;
            for (int k=0; k<shape[1]; k++) {
                if (data[i*shape[0]+k]!=0 && !found) {
                    found = 1;
                    pivot = data[i*shape[0]+k];
                    index = k;
                }
            }
            if (!found) {
                //swap rows;
            }
        }
        for (int j=i+1; j<shape[0]; j++) {
            T coeff = data[j*shape[0]+index] / pivot;
            for (int k=0; k<shape[1]; k++) {
                data[j*shape[0]+k] -= coeff * data[i*shape[0]+k];
            }
        }
    }
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
    std::cout << "Tensor(";
    if (shape.empty()) {
        std::cout << "[])" << std::endl;
        return;
    }  
    printRecursive(data, shape, 0, std::vector<int>(shape.size()), 0);
    std::cout << ")" << std::endl;
}

template<class T>
void Tensor<T>::printRecursive(const std::vector<T>& data, const std::vector<int>& shape, int depth, std::vector<int> indices, int indent) {
    for (int i = 0; i < indent; i++) {
        std::cout << "  ";
    }

    if (depth == shape.size() - 1) {
        std::cout << "[";

        for (int i = 0; i < shape[depth]; i++) {
            indices[depth] = i;
            std::cout << (*this)(indices);

            if (i < shape[depth] - 1) {
                std::cout << ", ";
            }
        }

        std::cout << "]";
    } else {
        std::cout << "[" << std::endl;

        for (int i = 0; i < shape[depth]; i++) {
            indices[depth] = i;
            printRecursive(data, shape, depth + 1, indices, indent + 1);

            if (i < shape[depth] - 1) {
                std::cout << ",";
            }

            std::cout << std::endl;
        }

        for (int i = 0; i < indent; i++) {
            std::cout << "  ";
        }

        std::cout << "]";
    }
}


template class Tensor<int>;
template class Tensor<float>;
