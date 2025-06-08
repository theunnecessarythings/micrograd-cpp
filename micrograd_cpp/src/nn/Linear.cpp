#include "nn/Linear.h"
#include <cmath> // For std::sqrt
#include <random> // For weight initialization
#include <stdexcept> // For std::runtime_error

Linear::Linear(int input_features, int output_features, bool use_bias)
    : input_features(input_features), output_features(output_features), has_bias(use_bias) {

    weights = Tensor::randn({output_features, input_features});
    if (input_features > 0) {
        float kaiming_std = std::sqrt(2.0f / static_cast<float>(input_features));
        for(auto& val : weights->data) {
            if (val) val->data *= kaiming_std;
        }
    }

    if (has_bias) {
        bias = Tensor::zeros({output_features});
    } else {
        bias = nullptr;
    }
}

std::shared_ptr<Tensor> Linear::forward(const std::shared_ptr<Tensor>& input) {
    if (!input) {
        throw std::runtime_error("Linear::forward: Input tensor is null.");
    }
    if (!weights) {
        throw std::runtime_error("Linear::forward: Weights tensor is null.");
    }

    // Input validation
    if (input->ndim() == 0 || input->shape.back() != input_features) {
         std::string shape_str = "(";
        for(size_t i=0; i<input->shape.size(); ++i) shape_str += std::to_string(input->shape[i]) + (i == input->shape.size()-1 ? "" : ", ");
        shape_str += ")";
        throw std::runtime_error("Linear::forward: Last dimension of input tensor (" + std::to_string(input->shape.back()) +
                                 ") must match input_features (" + std::to_string(input_features) +
                                 "). Input shape: " + shape_str);
    }

    std::vector<int> original_shape = input->shape;
    bool was_1d = false;
    std::shared_ptr<Tensor> input_2d = input;

    if (input->ndim() == 1) { // [F_in]
        was_1d = true;
        input_2d = input->reshape({1, input_features}); // Reshape to [1, F_in]
    } else if (input->ndim() > 2) { // [B, S1, S2, ..., F_in]
        // Reshape to [B*S1*S2*..., F_in]
        int batch_dims_prod = 1;
        for (size_t i = 0; i < original_shape.size() - 1; ++i) {
            batch_dims_prod *= original_shape[i];
        }
        input_2d = input->reshape({batch_dims_prod, input_features});
    }
    // If input->ndim() == 2, input_2d is already [B, F_in]

    // W is [out_features, in_features]. W.T is [in_features, out_features]
    auto weights_t = weights->transpose();
    auto output_2d = input_2d->matmul(weights_t); // [B*S*.., F_in] @ [F_in, F_out] -> [B*S*.., F_out]

    if (has_bias) {
        if (!bias) {
            throw std::runtime_error("Linear::forward: Bias is enabled but bias tensor is null.");
        }
        // output_2d is [B_eff, F_out], bias is [F_out] (1D)
        // Broadcasting addition: output_ij = output_ij + bias_j
        if (output_2d->shape[1] != bias->shape[0]) {
             throw std::runtime_error("Bias shape mismatch during addition. Output cols: " + std::to_string(output_2d->shape[1]) + ", Bias size: " + std::to_string(bias->shape[0]));
        }
        std::vector<std::shared_ptr<Value>> result_data;
        result_data.reserve(output_2d->numel());
        for (int n = 0; n < output_2d->shape[0]; ++n) { // Effective batch dimension
            for (int o = 0; o < output_2d->shape[1]; ++o) { // Output features dimension
                result_data.push_back(output_2d->get({n, o}) + bias->get({o}));
            }
        }
        output_2d = Tensor::from_values(result_data, output_2d->shape);
    }

    // Reshape back to original ndim if necessary
    if (was_1d) { // Input was [F_in], output should be [F_out]
        return output_2d->reshape({output_features});
    } else if (input->ndim() > 2) { // Input was [B, S1,..., F_in], output should be [B, S1,..., F_out]
        std::vector<int> final_output_shape;
        for (size_t i = 0; i < original_shape.size() - 1; ++i) {
            final_output_shape.push_back(original_shape[i]);
        }
        final_output_shape.push_back(output_features);
        return output_2d->reshape(final_output_shape);
    }

    return output_2d; // Input was 2D, output is 2D
}

std::vector<std::shared_ptr<Value>> Linear::parameters() const {
    std::vector<std::shared_ptr<Value>> params;
    if (weights) {
        params.insert(params.end(), weights->data.begin(), weights->data.end());
    }
    if (has_bias && bias) {
        params.insert(params.end(), bias->data.begin(), bias->data.end());
    }
    return params;
}
