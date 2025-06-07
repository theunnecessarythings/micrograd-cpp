#include "nn/losses.h"
#include <stdexcept> // For std::runtime_error
#include <string>    // For std::to_string in error messages
#include <cmath>     // For std::floor

// --- MSELoss ---
std::shared_ptr<Value> MSELoss::forward(const std::shared_ptr<Tensor>& predictions,
                                        const std::shared_ptr<Tensor>& targets) {
    if (!predictions || !targets) {
        throw std::runtime_error("MSELoss: Predictions and targets tensors cannot be null.");
    }
    if (predictions->shape != targets->shape) {
        std::string pred_shape_str = " (";
        for(size_t i=0; i<predictions->shape.size(); ++i) pred_shape_str += std::to_string(predictions->shape[i]) + (i == predictions->shape.size()-1 ? "" : ", ");
        pred_shape_str += ")";
        std::string targ_shape_str = " (";
        for(size_t i=0; i<targets->shape.size(); ++i) targ_shape_str += std::to_string(targets->shape[i]) + (i == targets->shape.size()-1 ? "" : ", ");
        targ_shape_str += ")";
        throw std::runtime_error("MSELoss: Predictions" + pred_shape_str + " and targets" + targ_shape_str + " tensors must have the same shape.");
    }
    if (predictions->data.empty()) {
        // If predictions are empty, targets must also be empty due to shape check.
        // Loss for zero elements can be considered 0.
        return std::make_shared<Value>(0.0f, "MSE_empty_input");
    }

    int num_elements = predictions->numel();
    if (num_elements == 0) { // Should be caught by data.empty() if shape implies 0 elements (e.g. {0,5})
        return std::make_shared<Value>(0.0f, "MSE_zero_numel");
    }
    auto N_val = std::make_shared<Value>(static_cast<float>(num_elements), "MSE_N");

    std::shared_ptr<Value> sum_squared_errors = std::make_shared<Value>(0.0f, "sum_sq_err_init");

    for (size_t i = 0; i < predictions->data.size(); ++i) {
        if (!predictions->data[i] || !targets->data[i]) {
            throw std::runtime_error("MSELoss: Null Value pointer in input tensors at index " + std::to_string(i));
        }
        auto diff = predictions->data[i] - targets->data[i];
        diff->op = "MSE_diff_" + std::to_string(i);
        auto diff_sq = diff * diff;
        diff_sq->op = "MSE_diff_sq_" + std::to_string(i);
        sum_squared_errors = sum_squared_errors + diff_sq;
    }
    sum_squared_errors->op = "MSE_sum_sq_err";

    auto mse = sum_squared_errors / N_val;
    mse->op = "MSE";
    return mse;
}

// --- CrossEntropyLoss ---
std::shared_ptr<Value> CrossEntropyLoss::forward(const std::shared_ptr<Tensor>& logits,
                                                 const std::shared_ptr<Tensor>& target_indices) {
    if (!logits || !target_indices) {
        throw std::runtime_error("CrossEntropyLoss: Logits and target_indices tensors cannot be null.");
    }
    if (logits->ndim() == 0) {
        throw std::runtime_error("CrossEntropyLoss: Logits tensor must be at least 1D.");
    }
    if (logits->ndim() > 2) {
        throw std::runtime_error("CrossEntropyLoss: Logits tensor must be 1D or 2D. Got ndim: " + std::to_string(logits->ndim()));
    }
    if (target_indices->ndim() > 1) {
        throw std::runtime_error("CrossEntropyLoss: Target indices tensor must be 0D or 1D. Got ndim: " + std::to_string(target_indices->ndim()));
    }

    int batch_size = (logits->ndim() == 1) ? 1 : logits->shape[0];
    int num_classes = (logits->ndim() == 1) ? logits->shape[0] : logits->shape[1];

    if (logits->ndim() == 2 && target_indices->ndim() == 1 && (target_indices->numel() != 0 && logits->shape[0] != target_indices->shape[0])) {
         // Allow empty target_indices if batch_size is 0 (covered by batch_size == 0 check later)
        throw std::runtime_error("CrossEntropyLoss: Batch size mismatch. Logits batch: " + std::to_string(logits->shape[0]) + ", Targets batch: " + std::to_string(target_indices->shape[0]));
    }
    if (logits->ndim() == 1 && target_indices->ndim() == 1 && target_indices->numel() != 1) {
        throw std::runtime_error("CrossEntropyLoss: For 1D logits, 1D target_indices must have only one element.");
    }
    if (target_indices->ndim() == 0 && batch_size != 1) { // Scalar target for multi-item batch
        throw std::runtime_error("CrossEntropyLoss: Scalar target_indices are only allowed for 1D logits or batch_size=1 for 2D logits.");
    }
    if (target_indices->numel() == 0 && batch_size > 0) {
        throw std::runtime_error("CrossEntropyLoss: Target indices tensor is empty but logits are not.");
    }
     if (batch_size == 0) { // If logits lead to batch_size 0 (e.g. shape [0, 5]) or both empty
        return std::make_shared<Value>(0.0f, "CE_empty_input");
    }


    auto log_probs = logits->log_softmax();

    std::shared_ptr<Value> total_loss = std::make_shared<Value>(0.0f, "total_nll_init");

    for (int i = 0; i < batch_size; ++i) {
        float target_idx_float;
        if (target_indices->ndim() == 0) {
             target_idx_float = target_indices->data[0]->data;
        } else {
             target_idx_float = target_indices->get({i})->data;
        }

        if (target_idx_float != std::floor(target_idx_float)) {
            throw std::runtime_error("CrossEntropyLoss: Target index must be an integer value, got " + std::to_string(target_idx_float));
        }
        long target_idx_val_long = static_cast<long>(target_idx_float);

        if (target_idx_val_long < 0 || target_idx_val_long >= num_classes) {
            throw std::out_of_range("CrossEntropyLoss: Target index " + std::to_string(target_idx_val_long) +
                                    " out of range [0, " + std::to_string(num_classes - 1) + "].");
        }

        std::shared_ptr<Value> selected_log_prob;
        if (logits->ndim() == 1) {
            selected_log_prob = log_probs->get({(int)target_idx_val_long});
        } else {
            selected_log_prob = log_probs->get({i, (int)target_idx_val_long});
        }

        total_loss = total_loss - selected_log_prob;
    }

    auto mean_loss = total_loss / std::make_shared<Value>(static_cast<float>(batch_size), "CE_batch_size");
    mean_loss->op = "CrossEntropyLoss";
    return mean_loss;
}
