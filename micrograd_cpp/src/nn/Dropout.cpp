#include "nn/Dropout.h"
#include <stdexcept> // For std::invalid_argument, std::runtime_error
#include <vector>    // For std::vector
#include <string>    // For std::to_string

Dropout::Dropout(float dropout_probability) : p(dropout_probability), is_training(true) {
    if (p < 0.0f || p > 1.0f) { // p=1 means zero out everything, p > 1 is invalid.
                                // Standard dropout p is prob of being zeroed.
                                // PyTorch's p is prob of being zeroed.
                                // If p=1, then 1-p = 0, scale_factor = 1/0 -> Inf.
                                // So, p must be < 1.0.
        throw std::invalid_argument("Dropout probability must be in [0, 1). Got: " + std::to_string(p));
    }
}

void Dropout::train_mode(bool training) {
    is_training = training;
}
void Dropout::eval_mode() {
    is_training = false;
}

std::shared_ptr<Tensor> Dropout::forward(const std::shared_ptr<Tensor>& input) {
    if (!input) {
        throw std::runtime_error("Dropout::forward: Input tensor cannot be null.");
    }

    if (!is_training || p == 0.0f) {
        return input;
    }
    if (p == 1.0f) { // If p is 1.0, zero out everything.
        std::vector<std::shared_ptr<Value>> zero_values;
        zero_values.reserve(input->numel());
        for (size_t i = 0; i < input->numel(); ++i) {
            zero_values.push_back(std::make_shared<Value>(0.0f, "dropout_zero_p1"));
        }
        return Tensor::from_values(zero_values, input->shape);
    }


    std::vector<std::shared_ptr<Value>> output_values;
    output_values.reserve(input->numel());

    // Inverted dropout: scale by 1/(1-p) during training
    float scale_factor_val = 1.0f / (1.0f - p);
    auto scale_value = std::make_shared<Value>(scale_factor_val, "dropout_scale");

    // A new generator for each forward pass to ensure different masks.
    // Seeding with random_device for less predictability.
    // For fully reproducible tests, one might pass a generator or seed.
    std::mt19937 gen(std::random_device{}());
    std::bernoulli_distribution dist(1.0 - p); // Probability of NOT being zeroed (i.e. being kept)

    for (const auto& val_ptr : input->data) {
        if (!val_ptr) {
            throw std::runtime_error("Null Value pointer in Dropout input tensor.");
        }
        if (dist(gen)) { // If true (element is kept, bernoulli with prob 1-p)
            // Scale the kept element: val * scale_factor
            output_values.push_back(val_ptr * scale_value);
        } else { // Element is dropped (set to zero)
            // Create a Value(0.0) that is properly part of the graph if needed.
            // For dropout, dropped paths contribute zero gradient.
            // A fresh Value(0.0) correctly achieves this.
            output_values.push_back(std::make_shared<Value>(0.0f, "dropout_zero"));
        }
    }
    return Tensor::from_values(output_values, input->shape);
}

std::vector<std::shared_ptr<Value>> Dropout::parameters() const {
    return {}; // Dropout has no learnable parameters
}
