#include "nn/MultiHeadAttention.h"
#include <stdexcept> // For std::invalid_argument, std::runtime_error
#include <string>    // For std::to_string in error messages

MultiHeadAttention::MultiHeadAttention(int emb_dim, int n_heads, bool bias /*, float drop_p */)
    : embed_dim(emb_dim), num_heads(n_heads) /*, dropout_p(drop_p) */ {

    if (embed_dim <= 0 || num_heads <= 0) {
        throw std::invalid_argument("MultiHeadAttention: Embedding dimension (" + std::to_string(emb_dim) +
                                    ") and number of heads (" + std::to_string(n_heads) + ") must be positive.");
    }
    if (embed_dim % num_heads != 0) {
        throw std::invalid_argument("MultiHeadAttention: Embedding dimension (" + std::to_string(emb_dim) +
                                    ") must be divisible by the number of heads (" + std::to_string(n_heads) + ").");
    }
    head_dim = embed_dim / num_heads;

    wq = std::make_shared<Linear>(embed_dim, embed_dim, bias);
    wk = std::make_shared<Linear>(embed_dim, embed_dim, bias);
    wv = std::make_shared<Linear>(embed_dim, embed_dim, bias);
    wo = std::make_shared<Linear>(embed_dim, embed_dim, bias);
}

std::shared_ptr<Tensor> MultiHeadAttention::forward(
    const std::shared_ptr<Tensor>& query,
    const std::shared_ptr<Tensor>& key,
    const std::shared_ptr<Tensor>& value,
    const std::shared_ptr<Tensor>& mask) {

    if (!query || !key || !value) {
        throw std::invalid_argument("MultiHeadAttention::forward: Query, Key, and Value tensors must not be null.");
    }

    // Input shapes: query [B, Sq, E], key [B, Sk, E], value [B, Sv, E] (E = embed_dim)
    if (query->ndim() != 3 || key->ndim() != 3 || value->ndim() != 3) {
        throw std::invalid_argument("MultiHeadAttention::forward: Q, K, V must be 3D tensors [batch, seq_len, embed_dim]. "
                                    "Q_ndim=" + std::to_string(query->ndim()) +
                                    ", K_ndim=" + std::to_string(key->ndim()) +
                                    ", V_ndim=" + std::to_string(value->ndim()));
    }
    if (query->shape[2] != embed_dim || key->shape[2] != embed_dim || value->shape[2] != embed_dim) {
        throw std::runtime_error("MultiHeadAttention::forward: Q, K, or V embed_dim (" +
                                 std::to_string(query->shape[2]) + ", " +
                                 std::to_string(key->shape[2]) + ", " +
                                 std::to_string(value->shape[2]) +
                                 ") mismatch with layer's embed_dim (" + std::to_string(embed_dim) + ").");
    }
    if (key->shape[1] != value->shape[1]) { // seq_len_k must match seq_len_v
        throw std::runtime_error("MultiHeadAttention::forward: Sequence length of Key and Value (seq_len_kv) must match. K_seq_len: " +
                                 std::to_string(key->shape[1]) + ", V_seq_len: " + std::to_string(value->shape[1]));
    }

    // int batch_size = query->shape[0];
    // int seq_len_q = query->shape[1];
    // int seq_len_k = key->shape[1];
    // int seq_len_v = value->shape[1];

    // 1. Linear Projections
    auto q_proj = wq->forward(query); // -> [B, Sq, E]
    auto k_proj = wk->forward(key);   // -> [B, Sk, E]
    auto v_proj = wv->forward(value); // -> [B, Sv, E]

    // 2. Split Heads: Reshape & Permute
    // q_proj from [B, Sq, E] to [B, Sq, N, H_D] then permute to [B, N, Sq, H_D]
    // k_proj from [B, Sk, E] to [B, Sk, N, H_D] then permute to [B, N, Sk, H_D]
    // v_proj from [B, Sv, E] to [B, Sv, N, H_D] then permute to [B, N, Sv, H_D]
    // (N = num_heads, H_D = head_dim)

    // These operations require Tensor::reshape() and Tensor::permute() which are not yet implemented.

    // For the purpose of this subtask (Part 1), we are focusing on projections
    // and setting up the class structure. The actual multi-head logic involving
    // reshape, permute, scaled_dot_product_attention call, combine_heads, and wo->forward()
    // will be deferred until Tensor supports these ops.

    // To make this file compilable and indicate the current state:
    throw std::runtime_error("MultiHeadAttention::forward: Linear projections complete. "
                             "Reshape/permute for head splitting, attention, head combining, "
                             "and final output projection are not yet implemented (pending Tensor ops).");

    // ---- Placeholder for future implementation (Part 2) ----
    // // Example of how it might look (pseudo-code for reshape/permute):
    // auto q_split = q_proj->reshape({batch_size, seq_len_q, num_heads, head_dim});
    // auto q_final = q_split->permute({0, 2, 1, 3}); // B, N, Sq, H_D
    //
    // auto k_split = k_proj->reshape({batch_size, seq_len_k, num_heads, head_dim});
    // auto k_final = k_split->permute({0, 2, 1, 3}); // B, N, Sk, H_D
    //
    // auto v_split = v_proj->reshape({batch_size, seq_len_v, num_heads, head_dim});
    // auto v_final = v_split->permute({0, 2, 1, 3}); // B, N, Sv, H_D
    //
    // // Mask processing for multi-head (e.g., [B, Sq, Sk] -> [B, 1, Sq, Sk] or [B, N, Sq, Sk])
    // // std::shared_ptr<Tensor> processed_mask = mask; // Needs adjustment if mask is not already broadcastable
    //
    // auto attention_output = micrograd_nn::scaled_dot_product_attention(q_final, k_final, v_final, mask /*processed_mask*/);
    // // attention_output shape: [B, N, Sq, H_D]
    //
    // // Concatenate Heads: Permute & Reshape
    // auto attention_permuted = attention_output->permute({0, 2, 1, 3}); // -> [B, Sq, N, H_D]
    // auto concatenated_output = attention_permuted->reshape({batch_size, seq_len_q, embed_dim});
    //
    // return wo->forward(concatenated_output);
    // ---- End Placeholder ----
}

std::shared_ptr<Tensor> MultiHeadAttention::forward(
    const std::shared_ptr<Tensor>& x,
    const std::shared_ptr<Tensor>& mask) {
    return this->forward(x, x, x, mask); // Self-attention
}

std::vector<std::shared_ptr<Value>> MultiHeadAttention::parameters() const {
    std::vector<std::shared_ptr<Value>> params;
    if (wq) { auto p = wq->parameters(); params.insert(params.end(), p.begin(), p.end()); }
    if (wk) { auto p = wk->parameters(); params.insert(params.end(), p.begin(), p.end()); }
    if (wv) { auto p = wv->parameters(); params.insert(params.end(), p.begin(), p.end()); }
    if (wo) { auto p = wo->parameters(); params.insert(params.end(), p.begin(), p.end()); }
    return params;
}
