#ifndef RMSNORM_H
#define RMSNORM_H

#include "nn/Layer.h"
#include "core/Tensor.h"
#include <vector>
#include <memory>
#include <string> // For std::string

class RMSNorm : public Layer {
public:
    int normalized_shape_last_dim; // The feature dimension to normalize over
    std::shared_ptr<Tensor> gamma; // Scale parameter: shape [normalized_shape_last_dim]
    float eps;

    // Constructor: normalized_shape_last_dim is typically the last dimension of the input
    RMSNorm(int normalized_shape_last_dim, float eps = 1e-6f); // Common epsilon for RMSNorm

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input) override;
    std::vector<std::shared_ptr<Value>> parameters() const override;
    std::string name() const override { return "RMSNorm"; }
};

#endif // RMSNORM_H
