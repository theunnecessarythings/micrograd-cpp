#include "nn/ReLU.h"
#include <stdexcept> // For std::runtime_error

std::shared_ptr<Tensor> ReLU::forward(const std::shared_ptr<Tensor>& input) {
    if (!input) {
        throw std::runtime_error("ReLU::forward: Input tensor cannot be null.");
    }
    // The Tensor class already has a relu() method that returns a new Tensor
    // with ReLU applied element-wise.
    return input->relu();
}

std::vector<std::shared_ptr<Value>> ReLU::parameters() const {
    // ReLU layer has no learnable parameters
    return {};
}
