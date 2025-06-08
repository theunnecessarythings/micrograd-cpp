#ifndef TENSOR_H
#define TENSOR_H

#include "core/Value.h"
#include <vector>
#include <memory> // For std::shared_ptr
#include <stdexcept> // For std::runtime_error, std::out_of_range
#include <numeric> // For std::accumulate, std::inner_product
#include <iostream> // For printing
#include <iomanip>  // For std::setw

class Tensor {
public:
    std::vector<int> shape;
    std::vector<std::shared_ptr<Value>> data; // Flattened data

    // Constructors
    Tensor(const std::vector<int>& shape, bool requires_grad = true); // Initialize with zeros or random (TBD)
    Tensor(const std::vector<int>& shape, const std::vector<float>& initial_data, bool requires_grad = true);
    Tensor(const std::vector<int>& shape, const std::vector<std::shared_ptr<Value>>& initial_values);


    // Static creation methods
    static std::shared_ptr<Tensor> zeros(const std::vector<int>& shape, bool requires_grad = true);
    static std::shared_ptr<Tensor> ones(const std::vector<int>& shape, bool requires_grad = true);
    static std::shared_ptr<Tensor> randn(const std::vector<int>& shape, bool requires_grad = true); // Standard normal distribution
    static std::shared_ptr<Tensor> from_vector(const std::vector<float>& vec_data, const std::vector<int>& shape, bool requires_grad = true);
    static std::shared_ptr<Tensor> from_values(const std::vector<std::shared_ptr<Value>>& values, const std::vector<int>& shape);


    // Getters
    int ndim() const { return shape.size(); }
    int numel() const; // Total number of elements
    std::shared_ptr<Value> get(const std::vector<int>& indices) const;
    void set(const std::vector<int>& indices, const std::shared_ptr<Value>& val);
    void set(const std::vector<int>& indices, float val_data);


    // Basic operations (element-wise)
    std::shared_ptr<Tensor> operator+(const std::shared_ptr<Tensor>& other) const;
    std::shared_ptr<Tensor> operator-(const std::shared_ptr<Tensor>& other) const;
    std::shared_ptr<Tensor> operator*(const std::shared_ptr<Tensor>& other) const; // Element-wise multiplication
    // Division can be added if necessary, requires careful handling of Value division

    // Scalar operations
    std::shared_ptr<Tensor> operator+(float scalar) const;
    std::shared_ptr<Tensor> operator-(float scalar) const;
    std::shared_ptr<Tensor> operator*(float scalar) const;
    std::shared_ptr<Tensor> operator/(float scalar) const;


    // Matrix multiplication (dot product)
    std::shared_ptr<Tensor> matmul(const std::shared_ptr<Tensor>& other) const;

    // Activation functions (element-wise)
    std::shared_ptr<Tensor> relu() const;
    std::shared_ptr<Tensor> sigmoid() const;

    // Backward pass on all elements (e.g., for a scalar loss tensor)
    void backward(); // Calls backward on the single Value if it's a scalar tensor
    void backward(const std::shared_ptr<Tensor>& grad); // For element-wise gradient input

    // Zero gradients for all Value objects in the tensor
    void zero_grad();

    // Utility to print the tensor
    void print(const std::string& title = "") const;

    std::shared_ptr<Tensor> transpose() const; // For 2D tensors (matrices)

    // Statistical methods
    std::shared_ptr<Value> sum_all() const; // Sum of all elements in the tensor
    std::shared_ptr<Value> mean_all() const; // Mean of all elements
    std::shared_ptr<Value> var_all(bool unbiased = true) const; // Variance of all elements
    std::shared_ptr<Tensor> sqrt_elem() const; // Element-wise sqrt
    std::shared_ptr<Tensor> rsqrt_elem() const; // Element-wise reciprocal sqrt: 1/sqrt(x)
    std::shared_ptr<Tensor> exp_elem() const;  // Element-wise exp
    std::shared_ptr<Tensor> log_elem() const;  // Element-wise log
    std::shared_ptr<Tensor> log_softmax(int axis = -1) const; // Computes log_softmax over the last dimension
    std::shared_ptr<Tensor> softmax(int axis = -1) const; // Computes softmax over the specified dimension
    std::shared_ptr<Tensor> permute(const std::vector<int>& axes_order) const;

private:
    int get_flattened_index(const std::vector<int>& indices) const;
    bool check_shape_compatibility(const std::shared_ptr<Tensor>& other, bool broadcast = false) const;
};

// Free operator functions for scalar on the left
std::shared_ptr<Tensor> operator+(float scalar, const std::shared_ptr<Tensor>& tensor);
std::shared_ptr<Tensor> operator-(float scalar, const std::shared_ptr<Tensor>& tensor);
std::shared_ptr<Tensor> operator*(float scalar, const std::shared_ptr<Tensor>& tensor);

#endif // TENSOR_H
