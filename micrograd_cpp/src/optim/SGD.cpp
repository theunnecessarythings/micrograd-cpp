#include "optim/SGD.h"
#include <stdexcept> // For std::invalid_argument

SGD::SGD(const std::vector<std::shared_ptr<Value>>& params, float lr)
    : Optimizer(params), learning_rate(lr) {
    if (lr < 0.0f) {
        throw std::invalid_argument("Learning rate must be non-negative. Got: " + std::to_string(lr));
    }
}

void SGD::step() {
    for (const auto& p : params) {
        if (p) { // Ensure the parameter pointer is valid
            // Update rule: param_new = param_old - learning_rate * gradient
            p->data -= learning_rate * p->grad;
        }
        // Note: This direct manipulation of p->data does not create a new Value object
        // in the computation graph. This is the intended behavior for optimizers,
        // as they are updating leaf nodes (parameters) based on accumulated gradients.
        // The parameters themselves should not have their _backward functions altered by the optimizer step.
    }
}
