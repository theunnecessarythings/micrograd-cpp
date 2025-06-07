#ifndef TRAINER_H
#define TRAINER_H

#include "core/Tensor.h"
#include "core/Value.h"
#include "nn/Layer.h"       // For model representation (as a Layer or a sequence of Layers)
#include "optim/Optimizer.h"
#include "nn/losses.h"      // For loss functions (though not directly used by Trainer.h, good for context)
#include <vector>
#include <memory>   // For std::shared_ptr
#include <iostream> // For printing progress
#include <functional> // For std::function

// A simple sequential model representation for now.
// Could be a single Layer that internally manages other layers, or a vector of layers.
// Let's assume for now `model` is a std::shared_ptr<Layer> that represents the whole network.
// If the model is a container of layers (like nn.Sequential), it should expose parameters() and forward().

// Training step function
// Returns the loss for this step
std::shared_ptr<Value> train_step(
    const std::shared_ptr<Layer>& model,
    const std::shared_ptr<Tensor>& input_batch,
    const std::shared_ptr<Tensor>& target_batch,
    // Loss function signature needs to be more generic or we use specific ones.
    // For now, let's assume MSELoss or CrossEntropyLoss static forward methods can be passed.
    // This could be improved with std::function for the loss.
    std::function<std::shared_ptr<Value>(const std::shared_ptr<Tensor>&, const std::shared_ptr<Tensor>&)> loss_fn,
    const std::shared_ptr<Optimizer>& optimizer
);

// Basic fit function
void fit(
    const std::shared_ptr<Layer>& model,
    const std::vector<std::shared_ptr<Tensor>>& X_train, // Vector of input tensors
    const std::vector<std::shared_ptr<Tensor>>& Y_train, // Vector of target tensors
    std::function<std::shared_ptr<Value>(const std::shared_ptr<Tensor>&, const std::shared_ptr<Tensor>&)> loss_fn,
    const std::shared_ptr<Optimizer>& optimizer,
    int epochs,
    int batch_size = 1 // For now, assume batch_size=1 or data is pre-batched if batch_size > X_train.size()
);

// Evaluation step function
// Returns the loss for this step, and optionally other metrics can be computed here or outside
std::shared_ptr<Value> evaluate_step(
    const std::shared_ptr<Layer>& model,
    const std::shared_ptr<Tensor>& input_batch,
    const std::shared_ptr<Tensor>& target_batch,
    std::function<std::shared_ptr<Value>(const std::shared_ptr<Tensor>&, const std::shared_ptr<Tensor>&)> loss_fn
);

// Basic evaluate function (e.g., to calculate average loss on a test set)
// For more complex metrics like accuracy, they might be calculated inside or returned separately.
// This example focuses on average loss.
float evaluate_average_loss(
    const std::shared_ptr<Layer>& model,
    const std::vector<std::shared_ptr<Tensor>>& X_test,
    const std::vector<std::shared_ptr<Tensor>>& Y_test,
    std::function<std::shared_ptr<Value>(const std::shared_ptr<Tensor>&, const std::shared_ptr<Tensor>&)> loss_fn
);

#endif // TRAINER_H
