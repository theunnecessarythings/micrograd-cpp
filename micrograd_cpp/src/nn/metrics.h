#ifndef METRICS_H
#define METRICS_H

#include "core/Tensor.h"
#include <memory> // For std::shared_ptr
#include <vector>
#include <string> // For error messages
#include <limits> // For std::numeric_limits

// Calculates accuracy for classification tasks.
// predictions: Tensor of logits or probabilities, shape [batch_size, num_classes] or [num_classes]
// target_indices: Tensor of ground truth class indices, shape [batch_size] or scalar
// Returns accuracy as a float (e.g., 0.0 to 1.0).
float accuracy_score(const std::shared_ptr<Tensor>& predictions,
                     const std::shared_ptr<Tensor>& target_indices);

#endif // METRICS_H
