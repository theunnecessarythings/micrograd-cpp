#ifndef ATTENTION_H
#define ATTENTION_H

#include "core/Tensor.h"
#include <memory> // For std::shared_ptr
#include <cmath>  // For std::sqrt
#include <vector> // For std::vector in function signature (though not directly used)

namespace micrograd_nn { // Using a namespace for nn components

// Scaled Dot-Product Attention
// Q: [batch_size, seq_len_q, d_k]
// K: [batch_size, seq_len_k, d_k]
// V: [batch_size, seq_len_v, d_v] (where seq_len_k == seq_len_v)
// mask: [batch_size, seq_len_q, seq_len_k] or [1, seq_len_q, seq_len_k] for broadcasting
//       Values of true in mask indicate positions to be masked (set to large negative before softmax)
// Returns: Output Tensor [batch_size, seq_len_q, d_v]
std::shared_ptr<Tensor> scaled_dot_product_attention(
    const std::shared_ptr<Tensor>& query,
    const std::shared_ptr<Tensor>& key,
    const std::shared_ptr<Tensor>& value,
    const std::shared_ptr<Tensor>& mask = nullptr, // Optional mask
    float dropout_p = 0.0f // Dropout probability (not implemented in first pass)
);

} // namespace micrograd_nn

#endif // ATTENTION_H
