#ifndef GELU_H
#define GELU_H

#include "nn/Layer.h"
#include "core/Tensor.h"
#include <memory>
#include <string> // For std::string
#include <vector> // For std::vector

class GELU : public Layer {
public:
    GELU() = default;

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input) override;
    std::vector<std::shared_ptr<Value>> parameters() const override;
    std::string name() const override { return "GELU"; }
};

#endif // GELU_H
