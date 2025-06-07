#ifndef LINEAR_H
#define LINEAR_H

#include "nn/Layer.h" // Base class
#include "core/Tensor.h"
#include <vector>
#include <memory> // For std::shared_ptr
#include <string> // For std::string

class Linear : public Layer {
public:
    std::shared_ptr<Tensor> weights; // Tensor of shape [output_features, input_features]
    std::shared_ptr<Tensor> bias;    // Tensor of shape [output_features] or nullptr

    int input_features;
    int output_features;
    bool has_bias;

    // Constructor
    Linear(int input_features, int output_features, bool use_bias = true);

    // Forward pass
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input) override;

    // Get parameters
    std::vector<std::shared_ptr<Value>> parameters() const override;

    std::string name() const override { return "Linear"; }

private:
    // Kaiming initialization is done in constructor, so this helper might not be needed
    // void initialize_weights();
};

#endif // LINEAR_H
