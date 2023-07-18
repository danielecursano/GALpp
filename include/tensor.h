#ifndef TENSOR_H
#define TENSOR_H

#include <vector>

template<class T>
class Tensor {

    std::vector<T> data;
    int rows_swapped;

    int calculate_index(const std::vector<int>& index) const;
    bool check_index(const std::vector<int>& index) const;
    void printRecursive(const std::vector<T>& data, const std::vector<int>& shape, int depth, std::vector<int> indices, int indent);
    void swapRows(int row1, int row2);
    int findPivot(int row);

    public:

        std::vector<int> shape;
        
        Tensor(std::vector<int> shape);
        T& operator()(const std::vector<int>& index);
        const T& operator()(const std::vector<int>& index) const;
        T& operator[](int index);
        void operator*(T scalar);
        Tensor<T> dot(const Tensor<T> &other);
        void reshape(std::vector<int> new_shape);
        Tensor<T> transpose();
        void load(const std::vector<T>& extern_data);
        Tensor<T> flatten();
        void reduce();
        T det();
        T trace();
        void rand();
        int size();
        int rank();
        void print();
};

#endif