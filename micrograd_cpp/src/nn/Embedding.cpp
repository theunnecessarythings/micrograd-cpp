#include "nn/Embedding.h"
#include <stdexcept> // For std::out_of_range, std::invalid_argument
#include <cmath>     // For std::floor

Embedding::Embedding(int num_embeddings, int embedding_dim)
    : num_embeddings(num_embeddings), embedding_dim(embedding_dim) {
    if (num_embeddings <= 0 || embedding_dim <= 0) {
        throw std::invalid_argument("Embedding num_embeddings and embedding_dim must be positive. Got num_embeddings=" + std::to_string(num_embeddings) + ", embedding_dim=" + std::to_string(embedding_dim));
    }
    // Initialize weights, typically with small random values (e.g., from a normal distribution)
    // PyTorch default: torch.normal(0.0, 1.0, size)
    weights = Tensor::randn({num_embeddings, embedding_dim});
    // It's common to scale these initial weights, e.g., weights->operator*(0.01f) if needed,
    // but randn() already gives values from N(0,1).
}

std::shared_ptr<Tensor> Embedding::forward(const std::shared_ptr<Tensor>& input_indices_tensor) {
    if (!input_indices_tensor) {
        throw std::runtime_error("Embedding::forward (Tensor input): Input indices tensor cannot be null.");
    }
    if (!weights) {
        throw std::runtime_error("Embedding::forward (Tensor input): Embedding weights tensor is null.");
    }
    // Input tensor contains integer indices. Its data should be float representations of ints.
    // Output shape will be input_shape + [embedding_dim]

    std::vector<int> output_shape = input_indices_tensor->shape; // Copy input shape
    output_shape.push_back(embedding_dim); // Append embedding_dim

    std::vector<std::shared_ptr<Value>> output_values;
    output_values.reserve(input_indices_tensor->numel() * embedding_dim);

    for (const auto& index_value_ptr : input_indices_tensor->data) {
        if (!index_value_ptr) {
             throw std::runtime_error("Null Value pointer in input_indices_tensor for Embedding layer.");
        }
        // Ensure the float data in Value is a whole number and within representable int range.
        float float_index = index_value_ptr->data;
        if (float_index != std::floor(float_index)) {
            throw std::runtime_error("Embedding index must be an integer value, got " + std::to_string(float_index));
        }
        int index = static_cast<int>(float_index);

        if (index < 0 || index >= num_embeddings) {
            throw std::out_of_range("Embedding index (" + std::to_string(index) +
                                    ") out of range [0, " + std::to_string(num_embeddings - 1) + "].");
        }
        // Get the row corresponding to the index from the weights tensor
        for (int j = 0; j < embedding_dim; ++j) {
            output_values.push_back(weights->get({index, j}));
        }
    }
    return Tensor::from_values(output_values, output_shape);
}

std::shared_ptr<Tensor> Embedding::forward(int input_index) {
    if (!weights) {
        throw std::runtime_error("Embedding::forward (int input): Embedding weights tensor is null.");
    }
    if (input_index < 0 || input_index >= num_embeddings) {
        throw std::out_of_range("Embedding index (" + std::to_string(input_index) +
                                ") out of range [0, " + std::to_string(num_embeddings - 1) + "].");
    }
    std::vector<std::shared_ptr<Value>> row_values;
    row_values.reserve(embedding_dim);
    for (int j = 0; j < embedding_dim; ++j) {
        row_values.push_back(weights->get({input_index, j}));
    }
    // Output is a 1D tensor (the embedding vector itself)
    return Tensor::from_values(row_values, {embedding_dim});
}

std::vector<std::shared_ptr<Value>> Embedding::parameters() const {
    if (!weights) return {}; // Should not happen if constructor succeeded
    return weights->data; // All Value objects in the weights tensor are parameters
}
