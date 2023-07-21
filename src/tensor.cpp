#include "tensor.h"
#include "assert.h"
#include <vector>
#include <iostream>
#include <cstdlib>
#include "math.h"

template<class T>
Tensor<T>::Tensor(std::vector<int>shape) : shape_(shape) {
    int size = 1;
    for (int s: shape) {
        size *= s;
    }   
    data.resize(size);
    rows_swapped = 0;
}

template<class T>
Tensor<T> Tensor<T>::eye(int dim) {
    Tensor<T> output({dim, dim});
    for (int i=0; i<dim; i++) 
        output({i, i}) = 1;
    return output;
}

template<class T>
Tensor<T> Tensor<T>::random(std::vector<int>shape) {
    Tensor<T> output(shape);
    output.rand();
    return output;
}

template<class T>
bool Tensor<T>::check_index(const std::vector<int>& index) const{
    for (int i=0; i<shape_.size(); i++) {
        if (index[i] >= shape_[i]) 
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
        dims *= shape_[i];
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
    if ((*this).dims() == 1 && other.shape_.size()==1 && data.size()==other.data.size()) {
        Tensor<T>output({1});
        T out_data = 0;
        for (int i=0; i<data.size(); i++) {
            out_data += data[i] * other({i});
        } 
        output.data[0] = out_data;
        return output; 
    }

    if ((*this).dims() == 2 && other.shape_.size()==2 && shape_[1]==other.shape_[0]) {
        Tensor<T>output({shape_[0], other.shape_[1]});
        for (int i=0; i<shape_[0]; i++) 
            for (int j=0; j<other.shape_[1]; j++) 
                for (int k=0; k<shape_[1]; k++)
                    output({i, j}) += (*this)({i, k}) * other({k, j});
        return output;
    }
    assert(0==1);
}

template<class T>
Tensor<T> Tensor<T>::outer(Tensor<T> other) {
    assert((*this).dims()==1 && other.shape_.size()==1 && data.size()==other.data.size());
    reshape({shape_[0], 1});
    Tensor<T> a = (*this);
    Tensor<T> b = other;
    a.reshape({shape_[0], 1}); b.reshape({1, shape_[0]});
    return a.dot(b);
}

template<class T>
void Tensor<T>::reshape(std::vector<int> new_shape_) {
    int new_size = 1;
    for (int s: new_shape_) {
        new_size *= s;
    }
    assert(new_size==this->size());
    shape_ = new_shape_;
}

template<class T>
Tensor<T> Tensor<T>::transpose() {
    if (shape_.size()==1) {
        return *this;
    }
    if (shape_.size()==2) {
        Tensor<T>output({shape_[1], shape_[0]});

        for (int row=0; row<shape_[0]; row++) {
            for (int col=0; col<shape_[1]; col++) {
                output.data[(col*shape_[0])+row] = data[row*shape_[1]+col];
            }
        }
        return output;
    }
    assert(1==0);
}

template<class T>
void Tensor<T>::load(const std::vector<T>& extern_data) {
    assert(data.size()==extern_data.size());
    data = extern_data;
}

template<class T>
Tensor<T> Tensor<T>::flatten() {
    Tensor<T>output({(*this).size()});
    output.load(data);
    return output;
}

template<class T>
void Tensor<T>::swapRows(int row1, int row2) {
    assert(shape_.size()==2 && row1 < shape_[0] && row2 < shape_[1]);
    if (row1==row2) 
        return ;
    rows_swapped++;
    for (int i=0; i<shape_[1]; i++) {
        const T tmp = data[row1*shape_[0]+i];
        data[row1*shape_[0]+i] = data[row2*shape_[0]+i];
        data[row2*shape_[0]+i] = tmp;
    }
}

template<class T>
int Tensor<T>::findPivot(int row) {
    assert(shape_.size() == 2 && row < shape_[0]);
    int index = 0;
    while (index < shape_[1] && data[row * shape_[1] + index] == 0) {
        index++;
    }
    if (index >= shape_[1] || data[row * shape_[1] + index] == 0) {
        index = -1;
    }
    return index;
}

template<class T>
void Tensor<T>::reduce() {
    assert(shape_.size() == 2);
    rows_swapped = 0;
    for (int i = 0; i < shape_[0]; i++) {
        T pivot = data[i * shape_[1] + i];
        int index = i;
        if (pivot == 0) {
            index = findPivot(i);
            int row = i;
            for (int j = i + 1; j < shape_[0]; j++) {
                if (findPivot(j) != -1 && findPivot(j) < index) {
                    row = j;
                    index = findPivot(j);
                }
            }
            swapRows(i, row);
            pivot = data[i * shape_[1] + i];
        }
        for (int j = i + 1; j < shape_[0]; j++) {
            T coeff = data[j * shape_[1] + index] / pivot;
            for (int k = 0; k < shape_[1]; k++) {
                data[j * shape_[1] + k] -= coeff * data[i * shape_[1] + k];
            }
        }
    }
}

template<class T>
T Tensor<T>::det() {
    assert(shape_.size()==2 && shape_[0]==shape_[1]);
    Tensor<T> copy = (*this);
    copy.reduce();
    T det = pow(-1, rows_swapped);
    for (int i=0; i<shape_[0]; i++) {
        det *= copy({i, i});
    }
    return det;
}

template<class T>
T Tensor<T>::trace() {
    assert(shape_.size()==2);
    int lower_dim = shape_[0] >= shape_[1] ? shape_[1] : shape_[0];
    T trace = 0;
    for (int i=0; i<lower_dim; i++) 
        trace += (*this)({i, i});
    return trace;
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
int Tensor<T>::dims() {
    return shape_.size();
}

template<class T>
std::vector<int> Tensor<T>::shape() {
    return shape_;
}

template<class T>
void Tensor<T>::print() {
    std::cout << "Tensor(";
    if (shape_.empty()) {
        std::cout << "[])" << std::endl;
        return;
    }  
    printRecursive(data, shape_, 0, std::vector<int>(shape_.size()), 0);
    std::cout << ")" << std::endl;
}

template<class T>
void Tensor<T>::printRecursive(const std::vector<T>& data, const std::vector<int>& shape_, int depth, std::vector<int> indices, int indent) {
    for (int i = 0; i < indent; i++) {
        std::cout << "  ";
    }

    if (depth == shape_.size() - 1) {
        std::cout << "[";

        for (int i = 0; i < shape_[depth]; i++) {
            indices[depth] = i;
            std::cout << (*this)(indices);

            if (i < shape_[depth] - 1) {
                std::cout << ", ";
            }
        }

        std::cout << "]";
    } else {
        std::cout << "[" << std::endl;

        for (int i = 0; i < shape_[depth]; i++) {
            indices[depth] = i;
            printRecursive(data, shape_, depth + 1, indices, indent + 1);

            if (i < shape_[depth] - 1) {
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