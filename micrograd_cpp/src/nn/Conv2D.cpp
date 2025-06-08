#include "nn/Conv2D.h"
#include <cmath>     // For std::sqrt, std::floor
#include <stdexcept> // For std::invalid_argument, std::runtime_error
#include <string>    // For std::to_string in error messages
#include <vector>    // For std::vector

// Constructor implementation
Conv2D::Conv2D(int in_c, int out_c,
               std::pair<int, int> k_size,
               std::pair<int, int> strd,
               std::pair<int, int> pad,
               bool bias_flag)
    : in_channels(in_c), out_channels(out_c),
      kernel_size(k_size), stride(strd), padding(pad), use_bias(bias_flag) {

    if (in_channels <= 0 || out_channels <= 0 || kernel_size.first <= 0 || kernel_size.second <= 0 ||
        stride.first <= 0 || stride.second <= 0 || padding.first < 0 || padding.second < 0) {
        throw std::invalid_argument("Conv2D parameters (channels, kernel_size, stride) must be positive, and padding must be non-negative.");
    }
    initialize_parameters();
}

Conv2D::Conv2D(int in_c, int out_c,
               int kernel_size_val, int stride_val, int padding_val, bool bias_flag)
    : Conv2D(in_c, out_c,
             {kernel_size_val, kernel_size_val},
             {stride_val, stride_val},
             {padding_val, padding_val},
             bias_flag) {}


void Conv2D::initialize_parameters() {
    // Weight initialization (Kaiming He for ReLU as a common default)
    // Shape: [out_channels, in_channels, kernel_height, kernel_width]
    float fan_in = static_cast<float>(in_channels * kernel_size.first * kernel_size.second);
    float kaiming_std = 0.0f;
    if (fan_in > 0) { // Avoid division by zero if fan_in is zero (e.g. kernel_size 0, though constructor checks this)
        kaiming_std = std::sqrt(2.0f / fan_in);
    }

    weights = Tensor::randn({out_channels, in_channels, kernel_size.first, kernel_size.second});
    if (weights && weights->data.size() > 0 && fan_in > 0) { // Check if weights tensor was created and fan_in is valid
        for (auto& val : weights->data) {
            if (val) { // Ensure Value pointer is not null
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

std::shared_ptr<Tensor> Conv2D::forward(const std::shared_ptr<Tensor>& input) {
    if (!input) {
        throw std::runtime_error("Conv2D::forward: Input tensor cannot be null.");
    }
    if (input->ndim() != 4) {
        throw std::runtime_error("Conv2D::forward: Input tensor must be 4D (N, C_in, H_in, W_in). Got " +
                                 std::to_string(input->ndim()) + "D.");
    }
    if (input->shape[1] != in_channels) {
        throw std::runtime_error("Conv2D::forward: Input tensor channels (" + std::to_string(input->shape[1]) +
                                 ") mismatch layer's in_channels (" + std::to_string(in_channels) + ").");
    }
    if (!weights) {
        throw std::runtime_error("Conv2D::forward: Weights tensor is not initialized.");
    }


    int batch_size = input->shape[0];
    int input_height = input->shape[2];
    int input_width = input->shape[3];

    int kernel_h = kernel_size.first;
    int kernel_w = kernel_size.second;
    int stride_h = stride.first;
    int stride_w = stride.second;
    int padding_h = padding.first;
    int padding_w = padding.second;

    // Calculate output dimensions
    int output_height = static_cast<int>(std::floor(static_cast<float>(input_height - kernel_h + 2 * padding_h) / stride_h)) + 1;
    int output_width = static_cast<int>(std::floor(static_cast<float>(input_width - kernel_w + 2 * padding_w) / stride_w)) + 1;

    if (output_height <= 0 || output_width <= 0) {
        throw std::runtime_error("Conv2D::forward: Calculated output dimensions are non-positive (" +
                                 std::to_string(output_height) + "x" + std::to_string(output_width) +
                                 "). Check input size, kernel, padding, stride.");
    }

    std::vector<std::shared_ptr<Value>> output_data;
    output_data.reserve(batch_size * out_channels * output_height * output_width);

    for (int n = 0; n < batch_size; ++n) {             // Iterate over batch
        for (int c_out = 0; c_out < out_channels; ++c_out) { // Iterate over output channels (filters)
            for (int oh = 0; oh < output_height; ++oh) {   // Iterate over output height
                for (int ow = 0; ow < output_width; ++ow) { // Iterate over output width
                    // Calculate input window top-left corner
                    int h_start = oh * stride_h - padding_h;
                    int w_start = ow * stride_w - padding_w;

                    std::shared_ptr<Value> sum_val = std::make_shared<Value>(0.0f, "conv_sum_init");

                    // Perform convolution (actually cross-correlation)
                    for (int c_in = 0; c_in < in_channels; ++c_in) { // Iterate over input channels
                        for (int kh_idx = 0; kh_idx < kernel_h; ++kh_idx) {     // Iterate over kernel height
                            for (int kw_idx = 0; kw_idx < kernel_w; ++kw_idx) { // Iterate over kernel width
                                int current_h = h_start + kh_idx;
                                int current_w = w_start + kw_idx;

                                // Check bounds (for padding)
                                if (current_h >= 0 && current_h < input_height &&
                                    current_w >= 0 && current_w < input_width) {

                                    auto input_val = input->get({n, c_in, current_h, current_w});
                                    auto weight_val = weights->get({c_out, c_in, kh_idx, kw_idx});
                                    if (!input_val || !weight_val) {
                                        throw std::runtime_error("Conv2D::forward: Null Value pointer encountered in input or weights tensor during convolution.");
                                    }
                                    sum_val = sum_val + (input_val * weight_val);
                                }
                                // Padded values are implicitly zero, so they don't add to sum_val if outside bounds
                            }
                        }
                    }
                    // Add bias if enabled
                    if (use_bias) {
                        if (!biases) {
                             throw std::runtime_error("Conv2D::forward: Bias is enabled but biases tensor is null.");
                        }
                        auto bias_val = biases->get({c_out});
                         if (!bias_val) {
                            throw std::runtime_error("Conv2D::forward: Null Value pointer encountered in biases tensor.");
                        }
                        sum_val = sum_val + bias_val;
                    }
                    output_data.push_back(sum_val);
                }
            }
        }
    }
    return Tensor::from_values(output_data, {batch_size, out_channels, output_height, output_width});
}

std::vector<std::shared_ptr<Value>> Conv2D::parameters() const {
    std::vector<std::shared_ptr<Value>> params;

    if (weights) {
        // weights->data is std::vector<std::shared_ptr<Value>>
        // Ensure weights and its data are not null, though constructor should initialize.
        // The data vector itself might be empty if a dimension was 0, though numel would be 0.
        // The constructor already initializes weights->data with Value objects.
        if (!weights->data.empty()) {
             params.insert(params.end(), weights->data.begin(), weights->data.end());
        }
    }

    if (use_bias && biases) {
        // biases->data is std::vector<std::shared_ptr<Value>>
        if (!biases->data.empty()) {
            params.insert(params.end(), biases->data.begin(), biases->data.end());
        }
    }
    return params;
}
