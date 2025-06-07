#ifndef RELU_H
#define RELU_H

#include "nn/Layer.h"
#include "core/Tensor.h"
#include <vector>
#include <memory>
#include <string> // For std::string

class ReLU : public Layer {
public:
    ReLU() = default; // Default constructor

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input) override;

    std::vector<std::shared_ptr<Value>> parameters() const override;

    std::string name() const override { return "ReLU"; }
};

#endif // RELU_H
