#ifndef MULTIHEADATTENTION_H
#define MULTIHEADATTENTION_H

#include "nn/Layer.h"
#include "nn/Linear.h"     // For projection layers
#include "nn/attention.h"  // For scaled_dot_product_attention (though not fully used in part 1)
#include "core/Tensor.h"
#include <vector>
#include <memory> // For std::shared_ptr
#include <stdexcept> // For std::invalid_argument
#include <string>    // For std::string

class MultiHeadAttention : public Layer {
public:
    int embed_dim;
    int num_heads;
    int head_dim; // embed_dim / num_heads
    // float dropout_p; // Placeholder for future dropout

    std::shared_ptr<Linear> wq; // Query projection
    std::shared_ptr<Linear> wk; // Key projection
    std::shared_ptr<Linear> wv; // Value projection
    std::shared_ptr<Linear> wo; // Output projection

    // Constructor
    MultiHeadAttention(int embed_dim, int num_heads, bool bias = true /*, float dropout_p = 0.0f */);

    // query, key, value: [batch_size, seq_len, embed_dim]
    // mask: [batch_size, seq_len_q, seq_len_k] (or broadcastable for attention scores)
    // Returns: [batch_size, seq_len_q, embed_dim]
    // This is the primary forward method for distinct Q, K, V.
    std::shared_ptr<Tensor> forward(
        const std::shared_ptr<Tensor>& query,
        const std::shared_ptr<Tensor>& key,
        const std::shared_ptr<Tensor>& value,
        const std::shared_ptr<Tensor>& mask = nullptr);

    // This overrides Layer::forward and implements self-attention by default.
    // A mask can be passed if needed, but Layer::forward doesn't have it.
    // To handle masks with self-attention, one might call the QKV version directly: model->forward(x,x,x,mask).
    // Or, this method could be more complex if we want to pass optional mask through Layer::forward.
    // For now, self-attention via Layer::forward won't take a mask directly.
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x) override;


    std::vector<std::shared_ptr<Value>> parameters() const override;
    std::string name() const override { return "MultiHeadAttention"; }

private:
    // Helper for splitting heads: [B, S, E] -> [B, N, S, H_D] after permute
    // std::shared_ptr<Tensor> split_heads(const std::shared_ptr<Tensor>& tensor, int batch_size, int seq_len);
    // Helper for combining heads: [B, N, S, H_D] -> [B, S, E] after permute
    // std::shared_ptr<Tensor> combine_heads(const std::shared_ptr<Tensor>& tensor, int batch_size, int seq_len);
};

#endif // MULTIHEADATTENTION_H
