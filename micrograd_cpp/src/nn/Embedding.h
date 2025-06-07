#ifndef EMBEDDING_H
#define EMBEDDING_H

#include "nn/Layer.h"
#include "core/Tensor.h"
#include <vector>
#include <memory>
#include <string> // For std::string

class Embedding : public Layer {
public:
    std::shared_ptr<Tensor> weights; // Embedding table: [num_embeddings, embedding_dim]
    int num_embeddings;
    int embedding_dim;

    // Constructor
    Embedding(int num_embeddings, int embedding_dim);

    // Forward pass: input is a Tensor of integer indices
    // Input shape: [batch_size, sequence_length] or [sequence_length] or just [1] for single index
    // Output shape: [batch_size, sequence_length, embedding_dim] or [sequence_length, embedding_dim] or [embedding_dim]
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input_indices) override;

    // Overload for single index input for convenience, though Tensor input is primary
    std::shared_ptr<Tensor> forward(int input_index);


    std::vector<std::shared_ptr<Value>> parameters() const override;

    std::string name() const override { return "Embedding"; }
};

#endif // EMBEDDING_H
