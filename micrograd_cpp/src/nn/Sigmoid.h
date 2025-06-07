#ifndef SIGMOID_H
#define SIGMOID_H

#include "nn/Layer.h"
#include "core/Tensor.h"
#include <vector>
#include <memory>
#include <string> // For std::string

class Sigmoid : public Layer {
public:
    Sigmoid() = default;

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input) override;

    std::vector<std::shared_ptr<Value>> parameters() const override;

    std::string name() const override { return "Sigmoid"; }
};

#endif // SIGMOID_H
