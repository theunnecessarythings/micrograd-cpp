#include "core/Tensor.h"
#include <random> // For std::default_random_engine and std::normal_distribution
#include <algorithm> // For std::transform, std::generate
#include <numeric>   // For std::accumulate in numel, std::inner_product
#include <functional> // For std::multiplies, std::minus, etc. (though not strictly needed for current ops)

// --- Constructors ---
Tensor::Tensor(const std::vector<int>& shape, bool requires_grad) : shape(shape) {
    int total_elements = numel(); // numel() now returns 1 for shape {} due to scalar fix
    data.reserve(total_elements);
    for (int i = 0; i < total_elements; ++i) {
        data.push_back(std::make_shared<Value>(0.0f, "Value_from_Tensor_zeros"));
    }
}

Tensor::Tensor(const std::vector<int>& shape, const std::vector<float>& initial_data, bool requires_grad) : shape(shape) {
    int total_elements = numel(); // Uses updated numel
    if (initial_data.size() != static_cast<size_t>(total_elements)) {
        throw std::runtime_error("Initial data size (" + std::to_string(initial_data.size()) +
                                 ") does not match tensor shape numel (" + std::to_string(total_elements) + ").");
    }
    data.reserve(total_elements);
    for (float val : initial_data) {
        data.push_back(std::make_shared<Value>(val, "Value_from_Tensor_initial_data"));
    }
}

Tensor::Tensor(const std::vector<int>& shape, const std::vector<std::shared_ptr<Value>>& initial_values) : shape(shape) {
    int total_elements = numel(); // Uses updated numel
    if (initial_values.size() != static_cast<size_t>(total_elements)) {
        throw std::runtime_error("Initial values size (" + std::to_string(initial_values.size()) +
                                 ") does not match tensor shape numel (" + std::to_string(total_elements) + ").");
    }
    data = initial_values;
}

// --- Static creation methods ---
std::shared_ptr<Tensor> Tensor::zeros(const std::vector<int>& shape, bool requires_grad) {
    return std::make_shared<Tensor>(shape, requires_grad);
}

std::shared_ptr<Tensor> Tensor::ones(const std::vector<int>& shape, bool requires_grad) {
    auto tensor = std::make_shared<Tensor>(shape, requires_grad); // Will init with 0s
    for (auto& val_ptr : tensor->data) { // Then set to 1.0
        if(val_ptr) val_ptr->data = 1.0f;
        if(val_ptr) val_ptr->op = "Value_from_Tensor_ones";
    }
    return tensor;
}

std::shared_ptr<Tensor> Tensor::randn(const std::vector<int>& shape, bool requires_grad) {
    auto tensor = std::make_shared<Tensor>(shape, requires_grad); // Will init with 0s
    std::default_random_engine generator(std::random_device{}());
    std::normal_distribution<float> distribution(0.0, 1.0);
    for (auto& val_ptr : tensor->data) { // Then set to random
        if(val_ptr) val_ptr->data = distribution(generator);
        if(val_ptr) val_ptr->op = "Value_from_Tensor_randn";
    }
    return tensor;
}

std::shared_ptr<Tensor> Tensor::from_vector(const std::vector<float>& vec_data, const std::vector<int>& shape, bool requires_grad) {
    long int calculated_numel;
    if (shape.empty()) {
        calculated_numel = 1; // Scalar
    } else {
        calculated_numel = 1;
        for (int dim : shape) {
            if (dim <= 0) throw std::runtime_error("Dimension must be positive for non-scalar tensor.");
            calculated_numel *= dim;
        }
    }
    if (vec_data.size() != static_cast<size_t>(calculated_numel)) {
        throw std::runtime_error("Data size (" + std::to_string(vec_data.size()) +
                                 ") does not match number of elements expected by shape (" + std::to_string(calculated_numel) + ").");
    }
    return std::make_shared<Tensor>(shape, vec_data, requires_grad);
}

std::shared_ptr<Tensor> Tensor::from_values(const std::vector<std::shared_ptr<Value>>& values, const std::vector<int>& shape) {
    long int calculated_numel;
     if (shape.empty()) {
        calculated_numel = 1; // Scalar
    } else {
        calculated_numel = 1;
        for (int dim : shape) {
            if (dim <= 0) throw std::runtime_error("Dimension must be positive for non-scalar tensor.");
            calculated_numel *= dim;
        }
    }
    if (values.size() != static_cast<size_t>(calculated_numel)) {
        throw std::runtime_error("Values size (" + std::to_string(values.size()) +
                                 ") does not match number of elements expected by shape (" + std::to_string(calculated_numel) + ").");
    }
    return std::make_shared<Tensor>(shape, values);
}

// --- Getters ---
int Tensor::numel() const {
    if (shape.empty()) return 1; // Scalar tensor has 1 element
    for (int dim : shape) {
        if (dim <= 0) return 0;
    }
    return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
}

int Tensor::get_flattened_index(const std::vector<int>& indices) const {
    if (shape.empty()) {
        if (indices.empty() || (indices.size() == 1 && indices[0] == 0)) {
            return 0;
        } else {
            throw std::out_of_range("Invalid indices for scalar tensor. Expected {} or {0}. Got " + std::to_string(indices.size()) + " indices.");
        }
    }
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
    int flat_idx = get_flattened_index(indices);
    if (flat_idx < 0 || flat_idx >= static_cast<int>(data.size())) { // Should not happen if get_flattened_index is correct
        throw std::out_of_range("Calculated flat index " + std::to_string(flat_idx) + " is out of bounds for data size " + std::to_string(data.size()));
    }
    return data[flat_idx];
}

void Tensor::set(const std::vector<int>& indices, const std::shared_ptr<Value>& val) {
    int flat_idx = get_flattened_index(indices);
     if (flat_idx < 0 || flat_idx >= static_cast<int>(data.size())) {
        throw std::out_of_range("Calculated flat index " + std::to_string(flat_idx) + " is out of bounds for data size " + std::to_string(data.size()));
    }
    data[flat_idx] = val;
}

void Tensor::set(const std::vector<int>& indices, float val_data) {
    int flat_idx = get_flattened_index(indices);
    if (flat_idx < 0 || flat_idx >= static_cast<int>(data.size())) {
        throw std::out_of_range("Calculated flat index " + std::to_string(flat_idx) + " is out of bounds for data size " + std::to_string(data.size()));
    }
    data[flat_idx] = std::make_shared<Value>(val_data, "Value_from_Tensor_set");
}

bool Tensor::check_shape_compatibility(const std::shared_ptr<Tensor>& other, bool broadcast) const {
    if (!other) return false;
    if (this->shape != other->shape) {
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
        result_data.push_back(data[i] + other->data[i]);
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
        result_data.push_back(data[i] - other->data[i]);
    }
    return std::make_shared<Tensor>(shape, result_data);
}

std::shared_ptr<Tensor> Tensor::operator*(const std::shared_ptr<Tensor>& other) const {
    if (!check_shape_compatibility(other)) {
        throw std::runtime_error("Tensor shapes are not compatible for element-wise multiplication.");
    }
    std::vector<std::shared_ptr<Value>> result_data;
    result_data.reserve(numel());
    for (size_t i = 0; i < data.size(); ++i) {
        result_data.push_back(data[i] * other->data[i]);
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
    if (!other) throw std::runtime_error("Other tensor in matmul is null.");
    if (ndim() != 2 || other->ndim() != 2) {
        throw std::runtime_error("Matrix multiplication currently only supports 2D tensors.");
    }
    if (shape[1] != other->shape[0]) {
        throw std::runtime_error("Matrix dimensions are not compatible for multiplication (A.cols (" + std::to_string(shape[1]) + ") != B.rows (" + std::to_string(other->shape[0]) + ")).");
    }
    int M = shape[0]; int K = shape[1]; int N = other->shape[1];
    auto result_tensor = Tensor::zeros({M, N});
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            std::shared_ptr<Value> sum_val = result_tensor->get({i,j});
            for (int k_idx = 0; k_idx < K; ++k_idx) {
                sum_val = sum_val + (this->get({i, k_idx}) * other->get({k_idx, j}));
            }
            result_tensor->set({i, j}, sum_val);
        }
    }
    return result_tensor;
}

// --- Activation functions & element-wise math ---
std::shared_ptr<Tensor> Tensor::relu() const {
    std::vector<std::shared_ptr<Value>> result_data;
    result_data.reserve(numel());
    for (const auto& val_ptr : data) {
        if (!val_ptr) throw std::runtime_error("Null Value pointer in Tensor during relu operation.");
        result_data.push_back(val_ptr->relu());
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

std::shared_ptr<Tensor> Tensor::exp_elem() const {
    std::vector<std::shared_ptr<Value>> result_data;
    result_data.reserve(numel());
    for (const auto& val_ptr : data) {
        if (!val_ptr) throw std::runtime_error("Null Value pointer in Tensor during exp_elem operation.");
        result_data.push_back(val_ptr->exp());
    }
    return std::make_shared<Tensor>(shape, result_data);
}

std::shared_ptr<Tensor> Tensor::log_elem() const {
    std::vector<std::shared_ptr<Value>> result_data;
    result_data.reserve(numel());
    for (const auto& val_ptr : data) {
        if (!val_ptr) throw std::runtime_error("Null Value pointer in Tensor during log_elem operation.");
        result_data.push_back(val_ptr->log());
    }
    return std::make_shared<Tensor>(shape, result_data);
}

std::shared_ptr<Tensor> Tensor::tanh_elem() const {
    std::vector<std::shared_ptr<Value>> result_data;
    result_data.reserve(this->numel());
    for (const auto& val_ptr : data) {
        if (!val_ptr) throw std::runtime_error("Null Value pointer in Tensor during tanh_elem operation.");
        result_data.push_back(val_ptr->tanh());
    }
    return std::make_shared<Tensor>(this->shape, result_data);
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

std::shared_ptr<Tensor> Tensor::rsqrt_elem() const {
    std::vector<std::shared_ptr<Value>> result_data;
    result_data.reserve(numel());
    for (const auto& val_ptr : data) {
        if (!val_ptr) throw std::runtime_error("Null Value pointer in Tensor during rsqrt_elem.");
        auto sqrt_val = val_ptr->pow(0.5f);
        result_data.push_back(std::make_shared<Value>(1.0f, "rsqrt_one") / sqrt_val);
    }
    return std::make_shared<Tensor>(shape, result_data);
}

// --- More complex operations ---
std::shared_ptr<Tensor> Tensor::log_softmax(int axis) const {
    if (this->ndim() == 0 && this->numel() == 1 && shape.empty()) { // Scalar tensor
        return Tensor::zeros(this->shape); // log(softmax(x)) = log(1) = 0
    }
    if (this->ndim() == 0 && this->numel() == 0) return std::make_shared<Tensor>(this->shape, this->data); // Empty

    int actual_axis = axis;
    if (actual_axis < 0) {
        actual_axis = this->ndim() + actual_axis;
    }
    if (actual_axis < 0 || actual_axis >= this->ndim()) {
        throw std::out_of_range("LogSoftmax axis " + std::to_string(axis) + " out of range for tensor with ndim " + std::to_string(this->ndim()));
    }

    if (actual_axis != this->ndim() - 1) {
        throw std::runtime_error("log_softmax currently only supports operation along the last dimension (axis=-1 or ndim-1). Provided axis: " + std::to_string(axis) + " for ndim: " + std::to_string(this->ndim()));
    }
     if (this->numel() == 0 && !shape.empty() ) {
         return std::make_shared<Tensor>(this->shape, std::vector<std::shared_ptr<Value>>{});
    }

    std::vector<std::shared_ptr<Value>> result_flat_data;
    result_flat_data.reserve(this->numel());

    int dim_to_log_softmax = this->shape[actual_axis];
    if (dim_to_log_softmax == 0) {
        return std::make_shared<Tensor>(this->shape, std::vector<std::shared_ptr<Value>>{});
    }

    int instances = 1;
    for (int i=0; i < actual_axis; ++i) {
        instances *= this->shape[i];
    }

    for (int inst = 0; inst < instances; ++inst) {
        std::vector<int> base_indices(this->ndim());
        int temp_inst = inst;
        for(int d = actual_axis - 1; d >= 0; --d) {
            base_indices[d] = temp_inst % this->shape[d];
            temp_inst /= this->shape[d];
        }

        base_indices[actual_axis] = 0;
        std::shared_ptr<Value> max_val = this->get(base_indices);
        for (int j = 1; j < dim_to_log_softmax; ++j) {
            base_indices[actual_axis] = j;
            auto current_val = this->get(base_indices);
            if (!current_val || !max_val) throw std::runtime_error("Null value encountered during max_val computation in log_softmax.");
            if (current_val->data > max_val->data) {
                max_val = current_val;
            }
        }
        max_val->op = "logsoftmax_max";

        auto sum_exps_minus_max = std::make_shared<Value>(0.0f, "logsoftmax_sum_exps_init");
        for (int j = 0; j < dim_to_log_softmax; ++j) {
            base_indices[actual_axis] = j;
            auto val = this->get(base_indices);
            if (!val) throw std::runtime_error("Null value encountered for val in log_softmax.");
            auto x_minus_max = val - max_val;
            x_minus_max->op = "logsoftmax_xminmax";
            auto e = x_minus_max->exp();
            e->op = "logsoftmax_exp";
            sum_exps_minus_max = sum_exps_minus_max + e;
        }
        sum_exps_minus_max->op = "logsoftmax_sum_exps";

        auto log_sum_exps = sum_exps_minus_max->log();
        log_sum_exps->op = "logsoftmax_logsumexp";

        for (int j = 0; j < dim_to_log_softmax; ++j) {
            base_indices[actual_axis] = j;
            auto val = this->get(base_indices);
             if (!val) throw std::runtime_error("Null value encountered for val (second loop) in log_softmax.");
            auto x_minus_max = val - max_val;
            auto log_softmax_j = x_minus_max - log_sum_exps;
            log_softmax_j->op = "logsoftmax_final_val";
            result_flat_data.push_back(log_softmax_j);
        }
    }
    return std::make_shared<Tensor>(this->shape, result_flat_data);
}

std::shared_ptr<Tensor> Tensor::softmax(int axis) const {
    if (this->ndim() == 0 && this->numel() == 1 && shape.empty()) { // Scalar tensor
        auto one_val = std::make_shared<Value>(1.0f, "softmax_scalar");
        return std::make_shared<Tensor>(this->shape, std::vector<std::shared_ptr<Value>>{one_val});
    }
    if (this->ndim() == 0 && this->numel() == 0) return std::make_shared<Tensor>(this->shape, this->data); // Empty

    int actual_axis = axis;
    if (actual_axis < 0) {
        actual_axis = this->ndim() + actual_axis;
    }
    if (actual_axis < 0 || actual_axis >= this->ndim()) {
        throw std::out_of_range("Softmax axis " + std::to_string(axis) + " out of range for tensor with ndim " + std::to_string(this->ndim()));
    }

    if (actual_axis != this->ndim() - 1) {
        throw std::runtime_error("Softmax currently only supports operation along the last dimension (axis=-1 or ndim-1). Provided axis: " + std::to_string(axis) + " for ndim: " + std::to_string(this->ndim()));
    }
     if (this->numel() == 0 && !shape.empty() ) {
         return std::make_shared<Tensor>(this->shape, std::vector<std::shared_ptr<Value>>{});
    }

    std::vector<std::shared_ptr<Value>> result_flat_data;
    result_flat_data.reserve(this->numel());

    int dim_to_softmax = this->shape[actual_axis];
    if (dim_to_softmax == 0) {
        return std::make_shared<Tensor>(this->shape, std::vector<std::shared_ptr<Value>>{});
    }

    int instances = 1;
    for (int i=0; i < actual_axis; ++i) {
        instances *= this->shape[i];
    }

    for (int inst = 0; inst < instances; ++inst) {
        // Determine multi-dimensional index for the start of this instance's slice
        std::vector<int> base_indices(this->ndim());
        int temp_inst = inst;
        for(int d = actual_axis - 1; d >= 0; --d) {
            base_indices[d] = temp_inst % this->shape[d];
            temp_inst /= this->shape[d];
        }

        // Max trick for numerical stability
        base_indices[actual_axis] = 0;
        std::shared_ptr<Value> max_val = this->get(base_indices);
        for (int j = 1; j < dim_to_softmax; ++j) {
            base_indices[actual_axis] = j;
            auto current_val = this->get(base_indices);
            if (!current_val || !max_val) throw std::runtime_error("Null value encountered during max_val computation in softmax.");
            if (current_val->data > max_val->data) {
                max_val = current_val;
            }
        }
        max_val->op = "softmax_max";

        std::vector<std::shared_ptr<Value>> exps;
        exps.reserve(dim_to_softmax);
        std::shared_ptr<Value> sum_exps = std::make_shared<Value>(0.0f, "softmax_sum_exps_init");

        for (int j = 0; j < dim_to_softmax; ++j) {
            base_indices[actual_axis] = j;
            auto val = this->get(base_indices);
            if (!val) throw std::runtime_error("Null value encountered for val in softmax.");
            auto x_minus_max = val - max_val;
            x_minus_max->op = "softmax_xminmax";
            auto e = x_minus_max->exp();
            e->op = "softmax_exp";
            exps.push_back(e);
            sum_exps = sum_exps + e;
        }
        sum_exps->op = "softmax_sum_exps";

        for (int j = 0; j < dim_to_softmax; ++j) {
            auto softmax_val = exps[j] / sum_exps;
            softmax_val->op = "softmax_final_val";
            result_flat_data.push_back(softmax_val);
        }
    }
    return std::make_shared<Tensor>(this->shape, result_flat_data);
}

std::shared_ptr<Tensor> Tensor::permute(const std::vector<int>& axes_order) const {
    if (axes_order.size() != this->ndim()) {
        throw std::invalid_argument("Permute: axes_order size (" + std::to_string(axes_order.size()) +
                                    ") must match tensor dimensionality (" + std::to_string(this->ndim()) + ").");
    }
    if (this->ndim() == 0 && axes_order.empty()) { // Scalar tensor
        return std::make_shared<Tensor>(this->shape, this->data); // Permute of a scalar is itself
    }


    std::vector<int> new_shape(this->ndim());
    std::vector<bool> seen_dims(this->ndim(), false);
    for (size_t i = 0; i < axes_order.size(); ++i) {
        int axis = axes_order[i];
        if (axis < 0 || axis >= this->ndim()) {
            throw std::out_of_range("Permute: Invalid axis " + std::to_string(axis) +
                                    " in axes_order for tensor with ndim " + std::to_string(this->ndim()) + ".");
        }
        if (seen_dims[axis]) {
            throw std::invalid_argument("Permute: Duplicate axis " + std::to_string(axis) + " in axes_order.");
        }
        new_shape[i] = this->shape[axis];
        seen_dims[axis] = true;
    }

    // Ensure all original dimensions were used
    for(int i=0; i < this->ndim(); ++i) {
        if(!seen_dims[i]) {
            throw std::invalid_argument("Permute: axes_order must contain all original dimensions. Missing dimension " + std::to_string(i));
        }
    }

    int N = this->numel();
    std::vector<std::shared_ptr<Value>> new_data(N);

    if (N == 0) { // Handle empty tensor (e.g. shape {2,0,3})
        return std::make_shared<Tensor>(new_shape, new_data);
    }

    std::vector<int> current_new_indices(this->ndim());
    std::vector<int> original_indices(this->ndim());

    for (int i = 0; i < N; ++i) {
        // Calculate current_new_indices based on linear index 'i' and new_shape
        int temp_idx = i;
        for (int d = this->ndim() - 1; d >= 0; --d) {
            if (new_shape[d] == 0) { // Should be caught by N == 0 if any dim is 0
                 current_new_indices[d] = 0; // or continue, as this element won't exist
                 // This case should ideally be handled by N=0 check above if any new_shape[d] is 0.
            } else {
                current_new_indices[d] = temp_idx % new_shape[d];
                temp_idx /= new_shape[d];
            }
        }

        for (int j = 0; j < this->ndim(); ++j) {
            original_indices[axes_order[j]] = current_new_indices[j];
        }

        new_data[i] = this->get(original_indices);
    }

    return std::make_shared<Tensor>(new_shape, new_data);
}

std::shared_ptr<Tensor> Tensor::reshape(const std::vector<int>& new_shape) const {
    long new_calculated_numel = 1;
    if (new_shape.empty()){
        new_calculated_numel = 1; // Scalar has 1 element
    } else {
        for (int dim : new_shape) {
            if (dim <= 0) { // Note: Some frameworks allow -1 for one dim to be inferred. Not implemented here.
                throw std::invalid_argument("Reshape: Dimensions in new_shape must be positive. Got " + std::to_string(dim));
            }
            new_calculated_numel *= dim;
        }
    }

    long current_numel_val = this->numel(); // numel() should correctly return 1 for scalar this.

    if (new_calculated_numel != current_numel_val) {
        std::string current_shape_str, new_shape_str;
        // Helper to format shape for error message
        auto format_shape = [](const std::vector<int>& s) {
            std::string res = "{";
            for(size_t i=0; i<s.size(); ++i) res += std::to_string(s[i]) + (i==s.size()-1 ? "" : ",");
            res += "}";
            return res;
        };
        current_shape_str = format_shape(this->shape);
        new_shape_str = format_shape(new_shape);

        throw std::runtime_error("Reshape: Total number of elements must remain the same. Current shape " +
                                 current_shape_str + " (numel " + std::to_string(current_numel_val) +
                                 ") cannot be reshaped to " + new_shape_str + " (numel " +
                                 std::to_string(new_calculated_numel) + ")");
    }

    // Create a new tensor with the new shape but share the same Value object(s).
    // The order of Value objects in the `data` vector remains unchanged.
    // The interpretation of this flat data is what changes due to the new shape.
    // The constructor Tensor(const std::vector<int>& shape, const std::vector<std::shared_ptr<Value>>& initial_values)
    // will use the new_shape and the existing data.
    return std::make_shared<Tensor>(new_shape, this->data);
}

// --- Backward pass ---
void Tensor::backward() {
    if (numel() != 1 && !(shape.empty() && numel()==1) ) { // Allow scalar identified by empty shape and numel 1
        throw std::runtime_error("Backward on a Tensor can only be called for a scalar (single element) tensor. Numel is " + std::to_string(numel()));
    }
    if (data.empty() || !data[0]) {
         throw std::runtime_error("Tensor data is empty or null, cannot call backward.");
    }
    data[0]->backward();
}

void Tensor::backward(const std::shared_ptr<Tensor>& grad) {
    if (!grad) throw std::runtime_error("Gradient tensor cannot be null for backward(grad).");
    if (this->shape != grad->shape) {
        throw std::runtime_error("Gradient tensor shape must match the original tensor shape for element-wise backward.");
    }
    if (this->data.size() != grad->data.size()){ // Should be redundant if shapes match and numel calculation is correct
        throw std::runtime_error("Data and grad tensor internal data sizes mismatch.");
    }
    for(size_t i=0; i < data.size(); ++i) {
        if (!this->data[i] || !grad->data[i]) {
             throw std::runtime_error("Null Value pointer encountered in Tensor backward with gradient tensor.");
        }
        this->data[i]->grad += grad->data[i]->data;
    }
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

    if (shape.empty()) { // Scalar tensor
         if (numel()==1 && !data.empty() && data[0]) {
             std::cout << "Tensor scalar: [" << std::fixed << std::setprecision(4) << data[0]->data << "]" << std::endl;
         } else {
             std::cout << "Tensor scalar: (empty data)" << std::endl;
         }
         return;
    }

    if (numel() == 0) { // For shapes like {5,0}
        std::cout << "Tensor with shape (";
        for (size_t i = 0; i < shape.size(); ++i) {
            std::cout << shape[i] << (i == shape.size() - 1 ? "" : ", ");
        }
        std::cout << ") and 0 elements." << std::endl;
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
    auto temp_tensor = tensor->operator-(scalar);
    return temp_tensor->operator*(-1.0f);
}
std::shared_ptr<Tensor> operator*(float scalar, const std::shared_ptr<Tensor>& tensor) {
    if (!tensor) throw std::runtime_error("Input tensor is null for scalar multiplication.");
    return tensor->operator*(scalar);
}

// --- Transpose and Statistical methods ---
std::shared_ptr<Tensor> Tensor::transpose() const {
    if (ndim() != 2) {
        throw std::runtime_error("Transpose is only supported for 2D tensors (matrices). Current ndim: " + std::to_string(ndim()));
    }
    int rows = shape[0];
    int cols = shape[1];
    auto transposed_tensor = std::make_shared<Tensor>(std::vector<int>{cols, rows});
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            transposed_tensor->set({j, i}, this->get({i, j}));
        }
    }
    return transposed_tensor;
}

std::shared_ptr<Value> Tensor::sum_all() const {
    if (data.empty() && !(shape.empty() && numel()==1) ) return std::make_shared<Value>(0.0f, "sum_empty_tensor");
    if (shape.empty() && numel()==1) { // Scalar tensor
        if (data.empty() || !data[0]) throw std::runtime_error("Scalar tensor has no data for sum_all.");
        return data[0]; // Sum of a scalar is the scalar itself
    }

    auto current_sum = std::make_shared<Value>(0.0f, "sum_init");
    for (const auto& val_ptr : data) {
        if (!val_ptr) throw std::runtime_error("Null Value pointer in Tensor during sum_all.");
        current_sum = current_sum + val_ptr;
    }
    current_sum->op = "sum_all";
    return current_sum;
}

std::shared_ptr<Value> Tensor::mean_all() const {
    int N = numel();
    if (N == 0 && !shape.empty()) return std::make_shared<Value>(0.0f, "mean_zero_numel_tensor");
    if (shape.empty() && N==1) { // Scalar tensor
         if (data.empty() || !data[0]) throw std::runtime_error("Scalar tensor has no data for mean_all.");
        return data[0]; // Mean of a scalar is the scalar itself
    }
    if (data.empty()) return std::make_shared<Value>(0.0f, "mean_empty_data"); // Should be caught by N=0 if shape implies

    auto s = this->sum_all();
    s->op = "mean_sum_part";
    auto result = s / std::make_shared<Value>(static_cast<float>(N), "mean_N");
    result->op = "mean_all";
    return result;
}

std::shared_ptr<Value> Tensor::var_all(bool unbiased) const {
    int N = numel();
    if (shape.empty() && N==1) return std::make_shared<Value>(0.0f, "var_scalar"); // Variance of scalar is 0

    if (data.empty() || (unbiased && N < 2) || N == 0 ) {
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
    if (divisor_val == 0) return std::make_shared<Value>(0.0f, "var_zero_divisor");
    auto result = sum_sq_diff / std::make_shared<Value>(divisor_val, "var_N_divisor");
    result->op = "var_all";
    return result;
}
