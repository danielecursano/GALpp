#ifndef TENSOR_H
#define TENSOR_H

#include <vector>

template<class T>
class Tensor {
    std::vector<T> data;
    std::vector<int> shape;

    int calculate_index(const std::vector<int>& index) const;
    bool check_index(const std::vector<int>& index) const;

    public:
        Tensor(std::vector<int> shape);
        T& operator()(const std::vector<int>& index);
        const T& operator()(const std::vector<int>& index) const;
        void operator*(T scalar);
        Tensor<T> dot(const Tensor<T> &other);
        void reshape(std::vector<int> new_shape);
        void rand();
        int size();
        int rank();
        void print();
};

#endif