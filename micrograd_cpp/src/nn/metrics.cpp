#include "nn/metrics.h"
#include <stdexcept> // For std::runtime_error, std::out_of_range
#include <algorithm> // For std::max_element (not used, but common)
#include <vector>    // For std::vector
#include <limits>    // For std::numeric_limits
#include <string>    // For std::to_string
#include <cmath>     // For std::floor

float accuracy_score(const std::shared_ptr<Tensor>& predictions,
                     const std::shared_ptr<Tensor>& target_indices) {
    if (!predictions || !target_indices) {
        throw std::runtime_error("accuracy_score: Predictions and target_indices tensors cannot be null.");
    }

    // Validate shapes
    if (predictions->ndim() == 0) {
        throw std::runtime_error("accuracy_score: Predictions tensor must be at least 1D.");
    }
    if (predictions->ndim() > 2) {
        throw std::runtime_error("accuracy_score: Predictions tensor must be 1D or 2D. Got ndim: " + std::to_string(predictions->ndim()));
    }
    if (target_indices->ndim() > 1) {
        throw std::runtime_error("accuracy_score: Target indices tensor must be 0D or 1D. Got ndim: " + std::to_string(target_indices->ndim()));
    }

    int batch_size = (predictions->ndim() == 1) ? 1 : predictions->shape[0];
    int num_classes = (predictions->ndim() == 1) ? predictions->shape[0] : predictions->shape[1];

    if (predictions->ndim() == 2 && target_indices->ndim() == 1 && (target_indices->numel() !=0 && predictions->shape[0] != target_indices->shape[0])) {
        throw std::runtime_error("accuracy_score: Batch size mismatch. Predictions batch: " + std::to_string(predictions->shape[0]) + ", Targets batch: " + std::to_string(target_indices->shape[0]));
    }
    if (predictions->ndim() == 1 && target_indices->ndim() == 1 && target_indices->numel() != 1) {
        throw std::runtime_error("accuracy_score: For 1D predictions, 1D target_indices must have only one element.");
    }
     if (target_indices->ndim() == 0 && batch_size != 1) { // Scalar target for multi-item batch
        throw std::runtime_error("accuracy_score: Scalar target_indices are only allowed for 1D predictions or batch_size=1 for 2D predictions.");
    }
    if (target_indices->numel() == 0 && batch_size > 0) {
        throw std::runtime_error("accuracy_score: Target indices tensor is empty but predictions are not.");
    }
    if (batch_size == 0) { // If predictions lead to batch_size 0 or both empty
        return 0.0f; // Or 1.0f if no predictions means 100% correct on "nothing"? 0.0 is safer.
    }


    int correct_predictions = 0;

    for (int i = 0; i < batch_size; ++i) {
        float target_idx_float;
        if (target_indices->ndim() == 0) {
             target_idx_float = target_indices->data[0]->data;
        } else {
             target_idx_float = target_indices->get({i})->data;
        }

        if (target_idx_float != std::floor(target_idx_float)) {
            throw std::runtime_error("accuracy_score: Target index must be an integer value, got " + std::to_string(target_idx_float));
        }
        long target_idx_val_long = static_cast<long>(target_idx_float);

        if (target_idx_val_long < 0 || target_idx_val_long >= num_classes) {
            throw std::out_of_range("accuracy_score: Target index " + std::to_string(target_idx_val_long) +
                                    " out of range [0, " + std::to_string(num_classes - 1) + "].");
        }

        // Get predicted class (index of max logit/probability)
        int predicted_class_idx = 0;
        float max_score = -std::numeric_limits<float>::infinity();

        for (int j = 0; j < num_classes; ++j) {
            float current_score;
            if (predictions->ndim() == 1) {
                current_score = predictions->get({j})->data;
            } else {
                current_score = predictions->get({i, j})->data;
            }
            if (current_score > max_score) {
                max_score = current_score;
                predicted_class_idx = j;
            }
        }

        if (predicted_class_idx == target_idx_val_long) {
            correct_predictions++;
        }
    }
    return static_cast<float>(correct_predictions) / batch_size;
}
