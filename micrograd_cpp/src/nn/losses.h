#ifndef LOSSES_H
#define LOSSES_H

#include "core/Tensor.h"
#include "core/Value.h"
#include <memory> // For std::shared_ptr
#include <vector> // For std::vector in CrossEntropyLoss target
#include <string> // For error messages

// Base class for Loss functions (optional, but can be good practice)
class Loss {
public:
    virtual ~Loss() = default;
    // Most losses compute a scalar Value
    // Inputs are typically predictions and targets
    // How to define a generic forward? For now, static methods in derived classes or free functions.
};


class MSELoss : public Loss {
public:
    MSELoss() = default;
    // Computes Mean Squared Error: sum((preds_i - targets_i)^2) / N
    // predictions: Tensor of predicted values
    // targets: Tensor of true values (must have same shape as predictions)
    // Returns a scalar Value representing the loss.
    static std::shared_ptr<Value> forward(const std::shared_ptr<Tensor>& predictions,
                                          const std::shared_ptr<Tensor>& targets);
};

// Forward declaration for CrossEntropyLoss components if needed later (e.g. LogSoftmax)

class CrossEntropyLoss : public Loss {
public:
    CrossEntropyLoss() = default;
    // Computes Cross Entropy Loss: typically LogSoftmax + NLLLoss
    // logits: Tensor of raw scores from the model, shape [batch_size, num_classes] or [num_classes] for single item
    // target_indices: Tensor of ground truth class indices (long/int), shape [batch_size] or scalar for single item
    // Returns a scalar Value representing the mean loss.
    static std::shared_ptr<Value> forward(const std::shared_ptr<Tensor>& logits,
                                          const std::shared_ptr<Tensor>& target_indices);
};

#endif // LOSSES_H
