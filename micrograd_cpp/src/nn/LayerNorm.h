#ifndef LAYERNORM_H
#define LAYERNORM_H

#include "nn/Layer.h"
#include "core/Tensor.h"
#include <vector>
#include <memory>
#include <string> // For std::string

class LayerNorm : public Layer {
public:
    int normalized_shape_last_dim; // The feature dimension to normalize over
    std::shared_ptr<Tensor> gamma; // Scale parameter: shape [normalized_shape_last_dim]
    std::shared_ptr<Tensor> beta;  // Shift parameter: shape [normalized_shape_last_dim]
    float eps;

    // Constructor: normalized_shape_last_dim is typically the last dimension of the input (e.g., embedding_dim)
    LayerNorm(int normalized_shape_last_dim, float eps = 1e-5f);

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input) override;
    std::vector<std::shared_ptr<Value>> parameters() const override;
    std::string name() const override { return "LayerNorm"; }
};

#endif // LAYERNORM_H
