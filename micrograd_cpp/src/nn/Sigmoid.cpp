#include "nn/Sigmoid.h"
#include <stdexcept> // For std::runtime_error

std::shared_ptr<Tensor> Sigmoid::forward(const std::shared_ptr<Tensor>& input) {
    if (!input) {
        throw std::runtime_error("Sigmoid::forward: Input tensor cannot be null.");
    }
    return input->sigmoid(); // Uses the new Tensor::sigmoid() method
}

std::vector<std::shared_ptr<Value>> Sigmoid::parameters() const {
    // Sigmoid layer has no learnable parameters
    return {};
}
