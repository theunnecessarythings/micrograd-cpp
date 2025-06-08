#include "nn/Conv1D.h"
#include <cmath>     // For std::sqrt, std::floor
#include <stdexcept> // For std::invalid_argument, std::runtime_error
#include <string>    // For std::to_string in error messages
#include <vector>    // For std::vector

Conv1D::Conv1D(int in_c, int out_c,
               int k_size, int strd, int pad, bool bias_flag)
    : in_channels(in_c), out_channels(out_c),
      kernel_size(k_size), stride(strd), padding(pad), use_bias(bias_flag) {

    if (in_channels <= 0 || out_channels <= 0 || kernel_size <= 0 ||
        stride <= 0 || padding < 0) {
        throw std::invalid_argument("Conv1D parameters (channels, kernel_size, stride) must be positive, and padding must be non-negative.");
    }
    initialize_parameters();
}

void Conv1D::initialize_parameters() {
    // Weight initialization (Kaiming He for ReLU as a common default)
    // Shape: [out_channels, in_channels, kernel_size]
    float fan_in = static_cast<float>(in_channels * kernel_size);
    float kaiming_std = 0.0f; // Default if fan_in is 0
    if (fan_in > 0) {
        kaiming_std = std::sqrt(2.0f / fan_in);
    }

    weights = Tensor::randn({out_channels, in_channels, kernel_size});
    if (weights && weights->data.size() > 0 && fan_in > 0) { // Check if weights tensor was created and fan_in is valid for scaling
        for (auto& val : weights->data) {
            if(val) { // Ensure Value pointer is not null
                val->data *= kaiming_std;
            }
        }
    }


    if (use_bias) {
        // Bias initialization (zeros)
        // Shape: [out_channels]
        biases = Tensor::zeros({out_channels});
    } else {
        biases = nullptr;
    }
}

std::shared_ptr<Tensor> Conv1D::forward(const std::shared_ptr<Tensor>& input) {
    if (!input) {
        throw std::runtime_error("Conv1D::forward: Input tensor cannot be null.");
    }
    if (input->ndim() != 3) {
        throw std::runtime_error("Conv1D::forward: Input tensor must be 3D (N, C_in, L_in). Got " +
                                 std::to_string(input->ndim()) + "D.");
    }
    if (input->shape[1] != in_channels) {
        throw std::runtime_error("Conv1D::forward: Input tensor channels (" + std::to_string(input->shape[1]) +
                                 ") mismatch layer's in_channels (" + std::to_string(in_channels) + ").");
    }
    if (!weights) {
        throw std::runtime_error("Conv1D::forward: Weights tensor is not initialized.");
    }

    int batch_size = input->shape[0];
    int input_length = input->shape[2];

    // Calculate output dimensions
    int output_length = static_cast<int>(std::floor(static_cast<float>(input_length - kernel_size + 2 * padding) / stride)) + 1;

    if (output_length <= 0) {
        throw std::runtime_error("Conv1D::forward: Calculated output length (" + std::to_string(output_length) +
                                 ") is non-positive. Check input size (" + std::to_string(input_length) +
                                 "), kernel (" + std::to_string(kernel_size) +
                                 "), padding (" + std::to_string(padding) +
                                 "), stride (" + std::to_string(stride) + ").");
    }

    std::vector<std::shared_ptr<Value>> output_data;
    output_data.reserve(batch_size * out_channels * output_length);

    for (int n = 0; n < batch_size; ++n) {             // Iterate over batch
        for (int c_out = 0; c_out < out_channels; ++c_out) { // Iterate over output channels (filters)
            for (int ol = 0; ol < output_length; ++ol) {   // Iterate over output length
                // Calculate input window start position
                int l_start = ol * stride - padding;

                std::shared_ptr<Value> sum_val = std::make_shared<Value>(0.0f, "conv1d_sum_init");

                // Perform convolution (actually cross-correlation)
                for (int c_in = 0; c_in < in_channels; ++c_in) { // Iterate over input channels
                    for (int kl_idx = 0; kl_idx < kernel_size; ++kl_idx) {   // Iterate over kernel width/length
                        int current_l = l_start + kl_idx;

                        // Check bounds (for padding)
                        if (current_l >= 0 && current_l < input_length) {
                            auto input_val = input->get({n, c_in, current_l});
                            auto weight_val = weights->get({c_out, c_in, kl_idx});
                            if (!input_val || !weight_val) {
                                throw std::runtime_error("Conv1D::forward: Null Value pointer encountered in input or weights tensor during convolution.");
                            }
                            sum_val = sum_val + (input_val * weight_val);
                        }
                        // Padded values are implicitly zero
                    }
                }
                // Add bias if enabled
                if (use_bias) {
                    if (!biases) {
                        throw std::runtime_error("Conv1D::forward: Bias is enabled but biases tensor is null.");
                    }
                    auto bias_val = biases->get({c_out});
                    if (!bias_val) {
                        throw std::runtime_error("Conv1D::forward: Null Value pointer encountered in biases tensor.");
                    }
                    sum_val = sum_val + bias_val;
                }
                output_data.push_back(sum_val);
            }
        }
    }
    return Tensor::from_values(output_data, {batch_size, out_channels, output_length});
}

std::vector<std::shared_ptr<Value>> Conv1D::parameters() const {
    std::vector<std::shared_ptr<Value>> params;
    if (weights && !weights->data.empty()) {
        params.insert(params.end(), weights->data.begin(), weights->data.end());
    }
    if (use_bias && biases && !biases->data.empty()) {
        params.insert(params.end(), biases->data.begin(), biases->data.end());
    }
    return params;
}
