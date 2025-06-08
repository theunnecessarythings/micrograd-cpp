#ifndef DROPOUT_H
#define DROPOUT_H

#include "nn/Layer.h"
#include "core/Tensor.h"
#include <vector>
#include <memory>
#include <random> // For random number generation
#include <string> // For std::string

class Dropout : public Layer {
public:
    float p; // Dropout probability (probability of an element to be zeroed)
    bool is_training; // Flag to indicate training or evaluation mode

    // For reproducibility of dropout masks if needed, could be seeded.
    // A single static generator might lead to correlated masks across different Dropout layers
    // if not re-seeded or used carefully. For this example, a static generator is used in .cpp.
    // std::mt19937 gen;

    // Constructor
    Dropout(float dropout_probability = 0.5f);

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input) override;
    std::vector<std::shared_ptr<Value>> parameters() const override;

    void train_mode(bool training = true); // Method to set training/eval mode
    void eval_mode();                      // Convenience method for eval

    std::string name() const override { return "Dropout"; }
};

#endif // DROPOUT_H
