#include "nn/RMSNorm.h"
#include <cmath>     // For std::sqrt (though pow(-0.5) is used), good for context
#include <stdexcept> // For std::invalid_argument, std::runtime_error
#include <vector>    // For std::vector

RMSNorm::RMSNorm(int normalized_shape_last_dim, float eps)
    : normalized_shape_last_dim(normalized_shape_last_dim), eps(eps) {
    if (normalized_shape_last_dim <= 0) {
        throw std::invalid_argument("RMSNorm normalized_shape_last_dim must be positive. Got: " + std::to_string(normalized_shape_last_dim));
    }
    // Initialize gamma (scale) to 1s
    gamma = Tensor::ones({normalized_shape_last_dim});
}

std::shared_ptr<Tensor> RMSNorm::forward(const std::shared_ptr<Tensor>& input) {
    if (!input) {
        throw std::runtime_error("RMSNorm::forward: Input tensor is null.");
    }
    if (input->shape.empty()) {
        throw std::runtime_error("RMSNorm::forward: Input tensor cannot be empty or scalar for RMSNorm.");
    }
    if (input->shape.back() != normalized_shape_last_dim) {
        throw std::runtime_error("RMSNorm: last dimension of input (" + std::to_string(input->shape.back()) +
                                 ") does not match normalized_shape_last_dim (" + std::to_string(normalized_shape_last_dim) + ").");
    }

    std::vector<int> output_shape = input->shape;
    std::vector<std::shared_ptr<Value>> result_values_flat;
    result_values_flat.reserve(input->numel());

    int num_features = normalized_shape_last_dim;
    auto num_features_val = std::make_shared<Value>(static_cast<float>(num_features), "rms_N");
    auto eps_val = std::make_shared<Value>(eps, "rms_eps");

    int total_elements = input->numel();
    int num_instances = total_elements / num_features;

    for (int i = 0; i < num_instances; ++i) {
        std::vector<std::shared_ptr<Value>> current_instance_values;
        current_instance_values.reserve(num_features);
        int start_idx = i * num_features;
        for (int f = 0; f < num_features; ++f) {
            current_instance_values.push_back(input->data[start_idx + f]);
        }

        // Calculate Root Mean Square: rsqrt(mean(x_j^2) + eps)
        // mean_sq = sum(x_j^2) / num_features
        std::shared_ptr<Value> sum_sq = std::make_shared<Value>(0.0f, "rms_sum_sq_init");
        for (const auto& x_j : current_instance_values) {
            if (!x_j) throw std::runtime_error("Null value in input tensor data during RMSNorm forward.");
            sum_sq = sum_sq + (x_j * x_j);
        }
        sum_sq->op = "rms_sum_sq";

        auto mean_sq = sum_sq / num_features_val;
        mean_sq->op = "rms_mean_sq";

        // rsqrt_val = 1 / sqrt(mean_sq + eps) = (mean_sq + eps)^(-0.5)
        auto mean_sq_plus_eps = mean_sq + eps_val;
        mean_sq_plus_eps->op = "rms_mean_sq_eps";
        auto rsqrt_val = mean_sq_plus_eps->pow(-0.5f);
        rsqrt_val->op = "rms_rsqrt_val";

        // Normalize and scale
        for (int f = 0; f < num_features; ++f) {
            auto x_val = current_instance_values[f];
            auto normalized_x = x_val * rsqrt_val;
            normalized_x->op = "rms_x_norm";
            if (!gamma->get({f})) throw std::runtime_error("Null gamma value during RMSNorm forward.");
            auto y = gamma->get({f}) * normalized_x;
            y->op = "rms_y_scaled";
            result_values_flat.push_back(y);
        }
    }
    return Tensor::from_values(result_values_flat, output_shape);
}

std::vector<std::shared_ptr<Value>> RMSNorm::parameters() const {
    std::vector<std::shared_ptr<Value>> params;
    if (gamma) { // gamma should always exist
        params.insert(params.end(), gamma->data.begin(), gamma->data.end());
    }
    return params;
}
