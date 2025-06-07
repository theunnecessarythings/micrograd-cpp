#include "train/Trainer.h"
#include <stdexcept> // For std::runtime_error
#include <string>    // For std::to_string in error messages
#include <vector>    // For std::vector
#include <iostream>  // For std::cout, std::endl (used in fit)

std::shared_ptr<Value> train_step(
    const std::shared_ptr<Layer>& model,
    const std::shared_ptr<Tensor>& input_batch,
    const std::shared_ptr<Tensor>& target_batch,
    std::function<std::shared_ptr<Value>(const std::shared_ptr<Tensor>&, const std::shared_ptr<Tensor>&)> loss_fn,
    const std::shared_ptr<Optimizer>& optimizer
) {
    if (!model) {
        throw std::runtime_error("train_step: Model pointer cannot be null.");
    }
    if (!input_batch) {
        throw std::runtime_error("train_step: Input batch tensor cannot be null.");
    }
    if (!target_batch) {
        throw std::runtime_error("train_step: Target batch tensor cannot be null.");
    }
    if (!loss_fn) {
        throw std::runtime_error("train_step: Loss function cannot be null.");
    }
    if (!optimizer) {
        throw std::runtime_error("train_step: Optimizer pointer cannot be null.");
    }


    // 1. Forward pass
    auto predictions = model->forward(input_batch);
    if (!predictions) {
        throw std::runtime_error("train_step: Model forward pass returned nullptr for predictions.");
    }

    // 2. Calculate loss
    auto loss = loss_fn(predictions, target_batch);
    if (!loss) {
        throw std::runtime_error("train_step: Loss computation returned nullptr.");
    }

    // 3. Zero gradients
    // Optimizer internally holds the list of parameters from the model.
    optimizer->zero_grad();

    // 4. Backward pass (compute gradients)
    loss->backward();

    // 5. Update parameters
    optimizer->step();

    return loss;
}

void fit(
    const std::shared_ptr<Layer>& model,
    const std::vector<std::shared_ptr<Tensor>>& X_train,
    const std::vector<std::shared_ptr<Tensor>>& Y_train,
    std::function<std::shared_ptr<Value>(const std::shared_ptr<Tensor>&, const std::shared_ptr<Tensor>&)> loss_fn,
    const std::shared_ptr<Optimizer>& optimizer,
    int epochs,
    int batch_size // Currently conceptual, actual batching logic is simple iteration
) {
    if (!model) {
        throw std::runtime_error("fit: Model pointer cannot be null.");
    }
    if (!loss_fn) {
        throw std::runtime_error("fit: Loss function cannot be null.");
    }
    if (!optimizer) {
        throw std::runtime_error("fit: Optimizer pointer cannot be null.");
    }
    if (X_train.size() != Y_train.size()) {
        throw std::runtime_error("fit: X_train and Y_train must have the same number of samples. X_train size: " +
                                 std::to_string(X_train.size()) + ", Y_train size: " + std::to_string(Y_train.size()));
    }
    if (epochs <= 0) {
        throw std::runtime_error("fit: Number of epochs must be positive. Got: " + std::to_string(epochs));
    }

    size_t num_samples = X_train.size();
    if (num_samples == 0) {
        std::cout << "fit: Training data is empty. No training will occur." << std::endl;
        return;
    }


    for (int epoch = 0; epoch < epochs; ++epoch) {
        float epoch_loss_sum = 0.0f;
        int steps_in_epoch = 0;

        // Simple iteration, assuming each X_train[i] is a "batch"
        // More sophisticated batching (e.g. creating mini-batches from X_train) to be added later.
        for (size_t i = 0; i < num_samples; ++i) {
            if (!X_train[i]) {
                 throw std::runtime_error("fit: Null input tensor found in X_train at index " + std::to_string(i));
            }
            if (!Y_train[i]) {
                 throw std::runtime_error("fit: Null target tensor found in Y_train at index " + std::to_string(i));
            }

            auto loss_value = train_step(model, X_train[i], Y_train[i], loss_fn, optimizer);
            if (loss_value) { // Check if loss_value is not null
                epoch_loss_sum += loss_value->data;
            }
            steps_in_epoch++;
        }

        if (steps_in_epoch > 0) {
            std::cout << "Epoch " << (epoch + 1) << "/" << epochs
                      << ", Loss: " << (epoch_loss_sum / steps_in_epoch) << std::endl;
        } else {
             // This case should ideally not be reached if num_samples > 0
             std::cout << "Epoch " << (epoch + 1) << "/" << epochs
                      << ", No data processed in this epoch." << std::endl;
        }
    }
}

std::shared_ptr<Value> evaluate_step(
    const std::shared_ptr<Layer>& model,
    const std::shared_ptr<Tensor>& input_batch,
    const std::shared_ptr<Tensor>& target_batch,
    std::function<std::shared_ptr<Value>(const std::shared_ptr<Tensor>&, const std::shared_ptr<Tensor>&)> loss_fn
) {
    if (!model) {
        throw std::runtime_error("evaluate_step: Model pointer cannot be null.");
    }
    if (!input_batch) {
        throw std::runtime_error("evaluate_step: Input batch tensor cannot be null.");
    }
    if (!target_batch) {
        throw std::runtime_error("evaluate_step: Target batch tensor cannot be null.");
    }
    if (!loss_fn) {
        throw std::runtime_error("evaluate_step: Loss function cannot be null.");
    }


    // 1. Forward pass
    auto predictions = model->forward(input_batch);
    if (!predictions) {
        throw std::runtime_error("evaluate_step: Model forward pass returned nullptr for predictions.");
    }

    // 2. Calculate loss
    auto loss = loss_fn(predictions, target_batch);
    if (!loss) {
        throw std::runtime_error("evaluate_step: Loss computation returned nullptr.");
    }

    // No gradient computation or optimizer step in evaluation.
    return loss;
}

float evaluate_average_loss(
    const std::shared_ptr<Layer>& model,
    const std::vector<std::shared_ptr<Tensor>>& X_test,
    const std::vector<std::shared_ptr<Tensor>>& Y_test,
    std::function<std::shared_ptr<Value>(const std::shared_ptr<Tensor>&, const std::shared_ptr<Tensor>&)> loss_fn
) {
    if (!model) {
        throw std::runtime_error("evaluate_average_loss: Model pointer cannot be null.");
    }
    if (!loss_fn) {
        throw std::runtime_error("evaluate_average_loss: Loss function cannot be null.");
    }
    if (X_test.size() != Y_test.size()) {
        throw std::runtime_error("evaluate_average_loss: X_test and Y_test must have the same number of samples. X_test size: " +
                                 std::to_string(X_test.size()) + ", Y_test size: " + std::to_string(Y_test.size()));
    }
    if (X_test.empty()) {
        std::cout << "evaluate_average_loss: Test dataset is empty." << std::endl;
        return 0.0f; // Or throw error, or return NaN
    }

    float total_loss_sum = 0.0f;
    int steps = 0;

    for (size_t i = 0; i < X_test.size(); ++i) {
        if (!X_test[i]) {
            throw std::runtime_error("evaluate_average_loss: Null input tensor found in X_test at index " + std::to_string(i));
        }
        if (!Y_test[i]) {
            throw std::runtime_error("evaluate_average_loss: Null target tensor found in Y_test at index " + std::to_string(i));
        }
        auto loss_value = evaluate_step(model, X_test[i], Y_test[i], loss_fn);
        if (loss_value) { // Check if loss_value is not null
             total_loss_sum += loss_value->data;
        }
        steps++;
    }

    if (steps == 0) return 0.0f; // Should be caught by X_test.empty() but as safeguard
    return total_loss_sum / steps;
}
