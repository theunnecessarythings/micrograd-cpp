#ifndef LAYER_H
#define LAYER_H

#include "core/Tensor.h" // Assuming Tensor is in core/
#include "core/Value.h"  // Assuming Value is in core/
#include <vector>
#include <memory> // For std::shared_ptr
#include <string> // For std::string

class Layer {
public:
    // Virtual destructor is important for base classes with virtual methods
    virtual ~Layer() = default;

    // Forward pass: takes a tensor (or vector of tensors) and returns a tensor (or vector)
    // For simplicity, let's assume single tensor input and single tensor output for now.
    // This can be generalized later if needed.
    virtual std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input) = 0;

    // Get all learnable parameters (weights, biases) of the layer
    virtual std::vector<std::shared_ptr<Value>> parameters() const = 0;

    // Helper method to zero gradients for all parameters in this layer
    virtual void zero_grad() {
        for (const auto& param : parameters()) {
            if (param) { // Check if the pointer is valid
                param->grad = 0.0f;
            }
        }
    }

    // Optional: A method to get sub-modules (if this layer contains other layers)
    // virtual std::vector<std::shared_ptr<Layer>> children() const { return {}; }

    // Optional: A string representation of the layer
    virtual std::string name() const { return "Layer"; }
};

#endif // LAYER_H
