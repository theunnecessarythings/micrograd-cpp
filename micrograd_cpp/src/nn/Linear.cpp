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

    // Input validation (shape check)
    if (!((input->ndim() == 1 && input->shape[0] == input_features) ||
          (input->ndim() == 2 && input->shape[1] == input_features))) {
        std::string shape_str = "(";
        for(size_t i=0; i<input->shape.size(); ++i) shape_str += std::to_string(input->shape[i]) + (i == input->shape.size()-1 ? "" : ", ");
        shape_str += ")";
        throw std::runtime_error("Linear::forward: Input tensor shape mismatch. Expected [N, " + std::to_string(input_features) + "] or [" + std::to_string(input_features) + "]. Got " + std::to_string(input->ndim()) + "D tensor with shape " + shape_str);
    }

    std::shared_ptr<Tensor> input_reshaped = input;
    bool single_input_vector = false;
    if (input->ndim() == 1) {
        // Reshape [in_features] to [1, in_features]
        // Ensure data vector from input is valid before creating new tensor from it
        if(input->data.empty() && input_features > 0) {
            throw std::runtime_error("Linear::forward: Input vector is empty but input_features > 0.");
        }
        input_reshaped = Tensor::from_values(input->data, {1, input_features});
        single_input_vector = true;
    }

    // W is [out_features, in_features]. W.T is [in_features, out_features]
    auto weights_t = weights->transpose(); // Use the new Tensor method
    auto output = input_reshaped->matmul(weights_t); // [N, in] @ [in, out] -> [N, out]

    if (has_bias) {
        if (!bias) {
            throw std::runtime_error("Linear::forward: Bias is enabled but bias tensor is null.");
        }
        // output is [N, out_features], bias is [out_features] (1D)
        // We need broadcasting addition: output_ij = output_ij + bias_j
        if (output->shape[1] != bias->shape[0]) {
             throw std::runtime_error("Bias shape mismatch during addition. Output cols: " + std::to_string(output->shape[1]) + ", Bias size: " + std::to_string(bias->shape[0]));
        }
        std::vector<std::shared_ptr<Value>> result_data;
        result_data.reserve(output->numel());
        for (int n = 0; n < output->shape[0]; ++n) { // Batch dimension (N)
            for (int o = 0; o < output->shape[1]; ++o) { // Output features dimension
                result_data.push_back(output->get({n, o}) + bias->get({o}));
            }
        }
        output = Tensor::from_values(result_data, output->shape);
    }

    if (single_input_vector) {
        // Reshape output from [1, out_features] back to [out_features]
        if(output->data.empty() && output_features > 0) {
             throw std::runtime_error("Linear::forward: Output vector is empty but output_features > 0 before final reshape.");
        }
        return Tensor::from_values(output->data, {output_features});
    }
    return output;
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
