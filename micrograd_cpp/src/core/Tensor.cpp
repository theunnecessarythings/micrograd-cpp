#include "core/Tensor.h"
#include <random> // For std::default_random_engine and std::normal_distribution
#include <algorithm> // For std::transform, std::generate
#include <numeric>   // For std::accumulate in numel, std::inner_product
#include <functional> // For std::multiplies, std::minus, etc. (though not strictly needed for current ops)

// --- Constructors ---
Tensor::Tensor(const std::vector<int>& shape, bool requires_grad) : shape(shape) {
    int total_elements = numel();
    data.reserve(total_elements);
    for (int i = 0; i < total_elements; ++i) {
        // Initialize with zero, requires_grad is implicitly handled by Value constructor if not specified
        // Pass the string "Value_from_Tensor_zeros" to identify the source of this Value object.
        data.push_back(std::make_shared<Value>(0.0f, "Value_from_Tensor_zeros"));
    }
}

Tensor::Tensor(const std::vector<int>& shape, const std::vector<float>& initial_data, bool requires_grad) : shape(shape) {
    int total_elements = numel();
    if (initial_data.size() != static_cast<size_t>(total_elements)) {
        throw std::runtime_error("Initial data size does not match tensor shape. Expected: " + std::to_string(total_elements) + ", Got: " + std::to_string(initial_data.size()));
    }
    data.reserve(total_elements);
    for (float val : initial_data) {
        data.push_back(std::make_shared<Value>(val, "Value_from_Tensor_initial_data"));
    }
}

Tensor::Tensor(const std::vector<int>& shape, const std::vector<std::shared_ptr<Value>>& initial_values) : shape(shape) {
    int total_elements = numel();
    if (initial_values.size() != static_cast<size_t>(total_elements)) {
        throw std::runtime_error("Initial values size does not match tensor shape. Expected: " + std::to_string(total_elements) + ", Got: " + std::to_string(initial_values.size()));
    }
    data = initial_values;
}


// --- Static creation methods ---
std::shared_ptr<Tensor> Tensor::zeros(const std::vector<int>& shape, bool requires_grad) {
    return std::make_shared<Tensor>(shape, requires_grad); // Default constructor inits with zeros
}

std::shared_ptr<Tensor> Tensor::ones(const std::vector<int>& shape, bool requires_grad) {
    auto tensor = std::make_shared<Tensor>(shape, requires_grad);
    for (auto& val_ptr : tensor->data) {
        val_ptr->data = 1.0f;
        val_ptr->op = "Value_from_Tensor_ones"; // Update op
    }
    return tensor;
}

std::shared_ptr<Tensor> Tensor::randn(const std::vector<int>& shape, bool requires_grad) {
    auto tensor = std::make_shared<Tensor>(shape, requires_grad); // Inits with Value(0.0f)
    std::default_random_engine generator(std::random_device{}()); // Seed with random_device
    std::normal_distribution<float> distribution(0.0, 1.0);
    for (auto& val_ptr : tensor->data) {
        val_ptr->data = distribution(generator);
        val_ptr->op = "Value_from_Tensor_randn"; // Update op
    }
    return tensor;
}

std::shared_ptr<Tensor> Tensor::from_vector(const std::vector<float>& vec_data, const std::vector<int>& shape, bool requires_grad) {
    if (shape.empty()) {
        throw std::runtime_error("Shape cannot be empty for Tensor::from_vector");
    }
    long int expected_elements = 1;
    for (int dim : shape) {
        if (dim <= 0) throw std::runtime_error("Dimension must be positive.");
        expected_elements *= dim;
    }
    if (vec_data.size() != static_cast<size_t>(expected_elements)) {
        throw std::runtime_error("Data size does not match shape for Tensor::from_vector. Expected: " + std::to_string(expected_elements) + ", Got: " + std::to_string(vec_data.size()));
    }
    return std::make_shared<Tensor>(shape, vec_data, requires_grad);
}

std::shared_ptr<Tensor> Tensor::from_values(const std::vector<std::shared_ptr<Value>>& values, const std::vector<int>& shape) {
    if (shape.empty()) {
        throw std::runtime_error("Shape cannot be empty for Tensor::from_values");
    }
    long int expected_elements = 1;
    for (int dim : shape) {
        if (dim <= 0) throw std::runtime_error("Dimension must be positive.");
        expected_elements *= dim;
    }
    if (values.size() != static_cast<size_t>(expected_elements)) {
        throw std::runtime_error("Values size does not match shape for Tensor::from_values. Expected: " + std::to_string(expected_elements) + ", Got: " + std::to_string(values.size()));
    }
    return std::make_shared<Tensor>(shape, values);
}


// --- Getters ---
int Tensor::numel() const {
    if (shape.empty()) return 0;
    // Ensure all dimensions are positive, otherwise numel is 0 or invalid
    for (int dim : shape) {
        if (dim <= 0) return 0; // Or throw an error, depending on desired strictness
    }
    return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
}

int Tensor::get_flattened_index(const std::vector<int>& indices) const {
    if (indices.size() != shape.size()) {
        throw std::out_of_range("Number of indices (" + std::to_string(indices.size()) + ") does not match tensor dimensions (" + std::to_string(shape.size()) + ").");
    }
    int index = 0;
    int current_stride = 1;
    for (int i = shape.size() - 1; i >= 0; --i) {
        if (indices[i] < 0 || indices[i] >= shape[i]) {
            std::string error_msg = "Index (";
            for(size_t k=0; k<indices.size(); ++k) error_msg += std::to_string(indices[k]) + (k == indices.size()-1 ? "" : ", ");
            error_msg += ") out of bounds for shape (";
            for(size_t k=0; k<shape.size(); ++k) error_msg += std::to_string(shape[k]) + (k == shape.size()-1 ? "" : ", ");
            error_msg += ")";
            throw std::out_of_range(error_msg);
        }
        index += indices[i] * current_stride;
        current_stride *= shape[i];
    }
    return index;
}

std::shared_ptr<Value> Tensor::get(const std::vector<int>& indices) const {
    return data[get_flattened_index(indices)];
}

void Tensor::set(const std::vector<int>& indices, const std::shared_ptr<Value>& val) {
    data[get_flattened_index(indices)] = val;
}

void Tensor::set(const std::vector<int>& indices, float val_data) {
    // Pass the string "Value_from_Tensor_set" to identify the source of this Value object.
    data[get_flattened_index(indices)] = std::make_shared<Value>(val_data, "Value_from_Tensor_set");
}

bool Tensor::check_shape_compatibility(const std::shared_ptr<Tensor>& other, bool broadcast) const {
    // For now, require exact same shape for element-wise operations.
    // Broadcasting can be implemented later.
    if (this->shape != other->shape) {
        std::string self_shape_str = "(";
        for(size_t i=0; i<shape.size(); ++i) self_shape_str += std::to_string(shape[i]) + (i == shape.size()-1 ? "" : ", ");
        self_shape_str += ")";
        std::string other_shape_str = "(";
        for(size_t i=0; i<other->shape.size(); ++i) other_shape_str += std::to_string(other->shape[i]) + (i == other->shape.size()-1 ? "" : ", ");
        other_shape_str += ")";
        // std::cout << "Shape mismatch: self=" << self_shape_str << " other=" << other_shape_str << std::endl; // Debugging
        return false;
    }
    return true;
}


// --- Basic operations (element-wise) ---
std::shared_ptr<Tensor> Tensor::operator+(const std::shared_ptr<Tensor>& other) const {
    if (!check_shape_compatibility(other)) {
        throw std::runtime_error("Tensor shapes are not compatible for addition.");
    }
    std::vector<std::shared_ptr<Value>> result_data;
    result_data.reserve(numel());
    for (size_t i = 0; i < data.size(); ++i) {
        result_data.push_back(data[i] + other->data[i]); // Uses the free operator+ for shared_ptr<Value>
    }
    return std::make_shared<Tensor>(shape, result_data);
}

std::shared_ptr<Tensor> Tensor::operator-(const std::shared_ptr<Tensor>& other) const {
    if (!check_shape_compatibility(other)) {
        throw std::runtime_error("Tensor shapes are not compatible for subtraction.");
    }
    std::vector<std::shared_ptr<Value>> result_data;
    result_data.reserve(numel());
    for (size_t i = 0; i < data.size(); ++i) {
        result_data.push_back(data[i] - other->data[i]); // Uses the free operator- for shared_ptr<Value>
    }
    return std::make_shared<Tensor>(shape, result_data);
}

std::shared_ptr<Tensor> Tensor::sigmoid() const {
    std::vector<std::shared_ptr<Value>> result_data;
    result_data.reserve(numel());
    for (const auto& val_ptr : data) {
        if (!val_ptr) throw std::runtime_error("Null Value pointer in Tensor during sigmoid operation.");
        result_data.push_back(val_ptr->sigmoid());
    }
    return std::make_shared<Tensor>(shape, result_data);
}

std::shared_ptr<Tensor> Tensor::operator*(const std::shared_ptr<Tensor>& other) const { // Element-wise
    if (!check_shape_compatibility(other)) {
        throw std::runtime_error("Tensor shapes are not compatible for element-wise multiplication.");
    }
    std::vector<std::shared_ptr<Value>> result_data;
    result_data.reserve(numel());
    for (size_t i = 0; i < data.size(); ++i) {
        result_data.push_back(data[i] * other->data[i]); // Uses the free operator* for shared_ptr<Value>
    }
    return std::make_shared<Tensor>(shape, result_data);
}

// --- Scalar operations ---
std::shared_ptr<Tensor> Tensor::operator+(float scalar) const {
    std::vector<std::shared_ptr<Value>> result_data;
    result_data.reserve(numel());
    auto scalar_val_shared = std::make_shared<Value>(scalar, "scalar_for_op");
    for (const auto& val_ptr : data) {
        result_data.push_back(val_ptr + scalar_val_shared);
    }
    return std::make_shared<Tensor>(shape, result_data);
}

std::shared_ptr<Tensor> Tensor::operator-(float scalar) const {
    std::vector<std::shared_ptr<Value>> result_data;
    result_data.reserve(numel());
    auto scalar_val_shared = std::make_shared<Value>(scalar, "scalar_for_op");
    for (const auto& val_ptr : data) {
        result_data.push_back(val_ptr - scalar_val_shared);
    }
    return std::make_shared<Tensor>(shape, result_data);
}

std::shared_ptr<Tensor> Tensor::operator*(float scalar) const {
    std::vector<std::shared_ptr<Value>> result_data;
    result_data.reserve(numel());
    auto scalar_val_shared = std::make_shared<Value>(scalar, "scalar_for_op");
    for (const auto& val_ptr : data) {
        result_data.push_back(val_ptr * scalar_val_shared);
    }
    return std::make_shared<Tensor>(shape, result_data);
}

std::shared_ptr<Tensor> Tensor::operator/(float scalar) const {
    if (scalar == 0.0f) {
        throw std::runtime_error("Division by zero scalar.");
    }
    std::vector<std::shared_ptr<Value>> result_data;
    result_data.reserve(numel());
    auto scalar_val_shared = std::make_shared<Value>(scalar, "scalar_for_op");
    for (const auto& val_ptr : data) {
        result_data.push_back(val_ptr / scalar_val_shared);
    }
    return std::make_shared<Tensor>(shape, result_data);
}


// --- Matrix multiplication ---
std::shared_ptr<Tensor> Tensor::matmul(const std::shared_ptr<Tensor>& other) const {
    if (ndim() != 2 || other->ndim() != 2) {
        throw std::runtime_error("Matrix multiplication currently only supports 2D tensors.");
    }
    if (shape[1] != other->shape[0]) {
        throw std::runtime_error("Matrix dimensions are not compatible for multiplication (A.cols (" + std::to_string(shape[1]) + ") != B.rows (" + std::to_string(other->shape[0]) + ")).");
    }

    int M = shape[0];         // Rows of self
    int K = shape[1];         // Cols of self / Rows of other
    int N = other->shape[1];  // Cols of other

    std::vector<int> result_shape = {M, N};
    // Create a tensor initialized with Value objects already holding 0.0f
    auto result_tensor = Tensor::zeros(result_shape);

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            // The Value at result_tensor->get({i,j}) is already 0.0f.
            // We need to accumulate the products into this Value object.
            std::shared_ptr<Value> sum_val = result_tensor->get({i,j}); // This is Value(0.0f)
            for (int k_idx = 0; k_idx < K; ++k_idx) {
                auto val1 = this->get({i, k_idx});
                auto val2 = other->get({k_idx, j});
                sum_val = sum_val + (val1 * val2); // This creates new Value objects in the loop.
                                                   // The final sum_val replaces the initial Value(0.0f) in result_tensor
            }
            result_tensor->set({i, j}, sum_val); // Assign the final sum.
        }
    }
    return result_tensor;
}

// --- Activation functions ---
std::shared_ptr<Tensor> Tensor::relu() const {
    std::vector<std::shared_ptr<Value>> result_data;
    result_data.reserve(numel());
    for (const auto& val_ptr : data) {
        result_data.push_back(val_ptr->relu());
    }
    return std::make_shared<Tensor>(shape, result_data);
}

// --- Backward pass ---
void Tensor::backward() {
    if (numel() != 1) {
        throw std::runtime_error("Backward on a Tensor can only be called for a scalar (single element) tensor. Numel is " + std::to_string(numel()));
    }
    if (data.empty() || !data[0]) {
         throw std::runtime_error("Tensor data is empty or null, cannot call backward.");
    }
    data[0]->backward();
}

void Tensor::backward(const std::shared_ptr<Tensor>& grad) {
    if (this->shape != grad->shape) {
        throw std::runtime_error("Gradient tensor shape must match the original tensor shape for element-wise backward.");
    }
    if (this->data.size() != grad->data.size()){
        throw std::runtime_error("Data and grad tensor internal data sizes mismatch.");
    }
    for(size_t i=0; i < data.size(); ++i) {
        if (!this->data[i] || !grad->data[i]) {
             throw std::runtime_error("Null Value pointer encountered in Tensor backward with gradient tensor.");
        }
        this->data[i]->grad += grad->data[i]->data;
    }
    // This function sets the .grad field of each Value in this Tensor.
    // It's assumed that these Values are part of a larger computation graph,
    // and a subsequent call to backward() on a final scalar loss Value will
    // propagate gradients from these points.
}


void Tensor::zero_grad() {
    for (const auto& val_ptr : data) {
        if (val_ptr) {
            val_ptr->grad = 0.0f;
        }
    }
}

// --- Utility to print the tensor ---
void Tensor::print(const std::string& title) const {
    if (!title.empty()) {
        std::cout << title << std::endl;
    }
    if (ndim() == 0 || numel() == 0) {
        std::cout << "Tensor with shape (";
        for (size_t i = 0; i < shape.size(); ++i) {
            std::cout << shape[i] << (i == shape.size() - 1 ? "" : ", ");
        }
        std::cout << ") and no data or 0 elements." << std::endl;
        return;
    }

    if (ndim() == 1) { // Vector
        std::cout << "[";
        for (size_t i = 0; i < data.size(); ++i) {
            std::cout << std::fixed << std::setprecision(4) << (data[i] ? data[i]->data : NAN) << (i == data.size() - 1 ? "" : ", ");
        }
        std::cout << "]" << std::endl;
    } else if (ndim() == 2) { // Matrix
        std::cout << "[" << std::endl;
        for (int i = 0; i < shape[0]; ++i) {
            std::cout << "  [";
            for (int j = 0; j < shape[1]; ++j) {
                 auto val_ptr = get({i,j});
                std::cout << std::fixed << std::setprecision(4) << std::setw(8) << (val_ptr ? val_ptr->data : NAN) << (j == shape[1] - 1 ? "" : ", ");
            }
            std::cout << "]" << (i == shape[0] - 1 ? "" : ",") << std::endl;
        }
        std::cout << "]" << std::endl;
    } else {
        std::cout << "Tensor with shape (";
        for (size_t i = 0; i < shape.size(); ++i) {
            std::cout << shape[i] << (i == shape.size() - 1 ? "" : ", ");
        }
        std::cout << ")" << std::endl;
        // Printing flattened data for >2D tensors can be very verbose.
        // Consider printing a slice or summary. For now, just shape.
        // Example: print first few elements
        std::cout << "  Flattened data (first few): [";
        for (size_t i = 0; i < std::min((size_t)5, data.size()); ++i) {
             std::cout << std::fixed << std::setprecision(4) << (data[i] ? data[i]->data : NAN) << (i == std::min((size_t)5, data.size()) - 1 ? "" : ", ");
        }
        if (data.size() > 5) std::cout << "...";
        std::cout << "]" << std::endl;
    }
}

// --- Free operator functions for scalar on the left ---
std::shared_ptr<Tensor> operator+(float scalar, const std::shared_ptr<Tensor>& tensor) {
    if (!tensor) throw std::runtime_error("Input tensor is null for scalar addition.");
    return tensor->operator+(scalar);
}
std::shared_ptr<Tensor> operator-(float scalar, const std::shared_ptr<Tensor>& tensor) {
    if (!tensor) throw std::runtime_error("Input tensor is null for scalar subtraction.");
    // (scalar - tensor_val) = -(tensor_val - scalar)
    auto temp_tensor = tensor->operator-(scalar); // tensor_val - scalar
    return temp_tensor->operator*(-1.0f); // -(tensor_val - scalar)
}
std::shared_ptr<Tensor> operator*(float scalar, const std::shared_ptr<Tensor>& tensor) {
    if (!tensor) throw std::runtime_error("Input tensor is null for scalar multiplication.");
    return tensor->operator*(scalar);
}

// Note: Tensor::sigmoid() is now placed after Tensor::relu() by the previous diff block.
// This search block is to ensure no duplicate Tensor::sigmoid() remains here if the previous one failed partially.
// This also means the Tensor::transpose() will be after the (now correctly placed) sigmoid.

std::shared_ptr<Tensor> Tensor::transpose() const {
    if (ndim() != 2) {
        throw std::runtime_error("Transpose is only supported for 2D tensors (matrices). Current ndim: " + std::to_string(ndim()));
    }
    int rows = shape[0];
    int cols = shape[1];
    // Create new tensor with zeros, as data will be filled.
    // Using the constructor that initializes Value objects with 0.0f.
    auto transposed_tensor = std::make_shared<Tensor>(std::vector<int>{cols, rows});
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            // get({i,j}) returns a shared_ptr<Value>. No need to create new Value.
            transposed_tensor->set({j, i}, this->get({i, j}));
        }
    }
    return transposed_tensor;
}

// --- Statistical methods & element-wise math ---
std::shared_ptr<Value> Tensor::sum_all() const {
    if (data.empty()) return std::make_shared<Value>(0.0f, "sum_empty_tensor");
    // Summing Value objects will build a computation graph.
    auto current_sum = std::make_shared<Value>(0.0f, "sum_init");
    for (const auto& val_ptr : data) {
        if (!val_ptr) throw std::runtime_error("Null Value pointer in Tensor during sum_all.");
        current_sum = current_sum + val_ptr;
    }
    current_sum->op = "sum_all";
    return current_sum;
}

std::shared_ptr<Value> Tensor::mean_all() const {
    if (data.empty()) return std::make_shared<Value>(0.0f, "mean_empty_tensor");
    int N = numel();
    if (N == 0) return std::make_shared<Value>(0.0f, "mean_zero_numel_tensor"); // Should ideally not happen if data is not empty

    auto s = this->sum_all();
    s->op = "mean_sum_part"; // Rename op for clarity in graph if s was from sum_all()
    auto result = s / std::make_shared<Value>(static_cast<float>(N), "mean_N");
    result->op = "mean_all";
    return result;
}

std::shared_ptr<Value> Tensor::var_all(bool unbiased) const {
    int N = numel();
    if (data.empty() || (unbiased && N < 2) || N == 0) {
        return std::make_shared<Value>(0.0f, "var_insufficient_data");
    }

    auto m = this->mean_all();
    m->op = "var_mean_part";

    auto sum_sq_diff = std::make_shared<Value>(0.0f, "var_sum_sq_diff_init");
    for (const auto& val_ptr : data) {
        if (!val_ptr) throw std::runtime_error("Null Value pointer in Tensor during var_all.");
        auto diff = val_ptr - m;
        diff->op = "var_diff";
        auto diff_sq = diff * diff;
        diff_sq->op = "var_diff_sq";
        sum_sq_diff = sum_sq_diff + diff_sq;
    }
    sum_sq_diff->op = "var_sum_sq_diff";

    float divisor_val = static_cast<float>(unbiased ? (N - 1) : N);
    if (divisor_val == 0) return std::make_shared<Value>(0.0f, "var_zero_divisor"); // Should be caught by N<2 for unbiased

    auto result = sum_sq_diff / std::make_shared<Value>(divisor_val, "var_N_divisor");
    result->op = "var_all";
    return result;
}

std::shared_ptr<Tensor> Tensor::sqrt_elem() const {
    std::vector<std::shared_ptr<Value>> result_data;
    result_data.reserve(numel());
    for (const auto& val_ptr : data) {
        if (!val_ptr) throw std::runtime_error("Null Value pointer in Tensor during sqrt_elem.");
        result_data.push_back(val_ptr->pow(0.5f));
    }
    return std::make_shared<Tensor>(shape, result_data);
}

std::shared_ptr<Tensor> Tensor::rsqrt_elem() const { // 1/sqrt(x)
    std::vector<std::shared_ptr<Value>> result_data;
    result_data.reserve(numel());
    // Using a small epsilon for numerical stability with rsqrt, common in LayerNorm
    // Note: PyTorch LayerNorm applies eps to variance *before* sqrt.
    // Here, this is an element-wise rsqrt, so eps might be added to each element.
    // For LayerNorm specifically, the (var + eps).pow(-0.5) approach is better.
    // This rsqrt_elem is a general utility.
    // float epsilon = 1e-9f; // A very small epsilon to avoid sqrt(0) if not handled by pow or Value directly.
                          // However, Value::pow should handle negative inputs if necessary (e.g. return NaN or throw).
                          // And 1/0 is handled by Value division.
                          // Let's assume x is non-negative for typical rsqrt usage.

    for (const auto& val_ptr : data) {
        if (!val_ptr) throw std::runtime_error("Null Value pointer in Tensor during rsqrt_elem.");
        // auto val_plus_eps = val_ptr + std::make_shared<Value>(epsilon); // if epsilon needed here
        auto sqrt_val = val_ptr->pow(0.5f);
        result_data.push_back(std::make_shared<Value>(1.0f, "rsqrt_one") / sqrt_val);
    }
    return std::make_shared<Tensor>(shape, result_data);
}
