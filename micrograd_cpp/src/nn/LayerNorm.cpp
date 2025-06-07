#include "nn/LayerNorm.h"
#include <numeric>   // Not strictly needed now, but good for potential future std::accumulate use
#include <stdexcept> // For std::invalid_argument, std::runtime_error
#include <vector>    // For std::vector

LayerNorm::LayerNorm(int normalized_shape_last_dim, float eps)
    : normalized_shape_last_dim(normalized_shape_last_dim), eps(eps) {
    if (normalized_shape_last_dim <= 0) {
        throw std::invalid_argument("LayerNorm normalized_shape_last_dim must be positive. Got: " + std::to_string(normalized_shape_last_dim));
    }
    // Initialize gamma (scale) to 1s and beta (shift) to 0s
    // These should have requires_grad = true by default if not specified in Tensor static methods,
    // which is correct as they are learnable parameters.
    gamma = Tensor::ones({normalized_shape_last_dim});
    beta = Tensor::zeros({normalized_shape_last_dim});
}

std::shared_ptr<Tensor> LayerNorm::forward(const std::shared_ptr<Tensor>& input) {
    if (!input) {
        throw std::runtime_error("LayerNorm::forward: Input tensor is null.");
    }
    if (input->shape.empty()) {
        throw std::runtime_error("LayerNorm::forward: Input tensor cannot be empty or scalar for LayerNorm.");
    }
    if (input->shape.back() != normalized_shape_last_dim) {
        throw std::runtime_error("LayerNorm: last dimension of input (" + std::to_string(input->shape.back()) +
                                 ") does not match normalized_shape_last_dim (" + std::to_string(normalized_shape_last_dim) + ").");
    }

    std::vector<int> output_shape = input->shape;
    std::vector<std::shared_ptr<Value>> result_values_flat;
    result_values_flat.reserve(input->numel());

    // Determine batch size (N-1 dimensions product) and feature size (last dimension)
    int num_features = normalized_shape_last_dim;
    int total_elements = input->numel();
    int num_instances = total_elements / num_features; // Number of rows/instances to normalize independently

    for (int i = 0; i < num_instances; ++i) {
        // Extract the sub-tensor for the current instance/row
        std::vector<std::shared_ptr<Value>> current_instance_values;
        current_instance_values.reserve(num_features);
        int start_idx = i * num_features;
        for (int j = 0; j < num_features; ++j) {
            current_instance_values.push_back(input->data[start_idx + j]);
        }
        // Create a temporary 1D tensor for this instance to use mean_all and var_all
        auto current_instance_tensor = Tensor::from_values(current_instance_values, {num_features});

        // Calculate mean and variance for the current instance
        auto mean = current_instance_tensor->mean_all();

        // Variance: E[(X - E[X])^2]. Using N for denominator (not N-1 for unbiased) as is common in LayerNorm.
        std::shared_ptr<Value> sum_sq_diff = std::make_shared<Value>(0.0f, "ln_sum_sq_diff_init");
        for (const auto& val_ptr : current_instance_tensor->data) {
            auto diff = val_ptr - mean;
            sum_sq_diff = sum_sq_diff + (diff * diff);
        }
        auto var = sum_sq_diff / std::make_shared<Value>(static_cast<float>(num_features), "ln_N_var");

        // Inverse standard deviation: 1 / sqrt(var + eps)
        auto var_plus_eps = var + std::make_shared<Value>(eps, "ln_eps");
        auto inv_std_dev = var_plus_eps->pow(-0.5f);

        // Normalize and scale/shift
        for (int f = 0; f < num_features; ++f) {
            // current_instance_tensor->get({f}) is same as current_instance_values[f]
            auto x_normalized = (current_instance_values[f] - mean) * inv_std_dev;
            auto y = (gamma->get({f}) * x_normalized) + beta->get({f});
            result_values_flat.push_back(y);
        }
    }
    return Tensor::from_values(result_values_flat, output_shape);
}

std::vector<std::shared_ptr<Value>> LayerNorm::parameters() const {
    std::vector<std::shared_ptr<Value>> params;
    if (gamma) { // gamma should always exist
        params.insert(params.end(), gamma->data.begin(), gamma->data.end());
    }
    if (beta) { // beta should always exist
        params.insert(params.end(), beta->data.begin(), beta->data.end());
    }
    return params;
}
