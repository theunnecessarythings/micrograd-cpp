#include "nn/MultiHeadAttention.h"
#include "nn/attention.h" // For micrograd_nn::scaled_dot_product_attention
#include <stdexcept> // For std::invalid_argument, std::runtime_error
#include <string>    // For std::to_string in error messages
#include <vector>    // For std::vector

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
        throw std::invalid_argument("MultiHeadAttention: Query, Key, and Value tensors must not be null.");
    }
    if (query->ndim() != 3 || key->ndim() != 3 || value->ndim() != 3) {
        throw std::invalid_argument("MultiHeadAttention: Q, K, V must be 3D tensors [batch, seq_len, embed_dim]. "
                                    "Q_ndim=" + std::to_string(query->ndim()) +
                                    ", K_ndim=" + std::to_string(key->ndim()) +
                                    ", V_ndim=" + std::to_string(value->ndim()));
    }
    if (query->shape[2] != embed_dim || key->shape[2] != embed_dim || value->shape[2] != embed_dim) {
        throw std::runtime_error("MultiHeadAttention: Q, K, or V embed_dim (" +
                                 std::to_string(query->shape[2]) + ", " +
                                 std::to_string(key->shape[2]) + ", " +
                                 std::to_string(value->shape[2]) +
                                 ") mismatch with layer's embed_dim (" + std::to_string(embed_dim) + ").");
    }
     if (key->shape[1] != value->shape[1]) {
        throw std::runtime_error("MultiHeadAttention: Sequence length of Key and Value (seq_len_kv) must match. K_seq_len: " +
                                 std::to_string(key->shape[1]) + ", V_seq_len: " + std::to_string(value->shape[1]));
    }

    int batch_size = query->shape[0];
    int seq_len_q = query->shape[1];
    int seq_len_k = key->shape[1];
    // int seq_len_v = value->shape[1]; // same as seq_len_k

    // 1. Linear Projections
    auto q_proj = wq->forward(query); // -> [B, Sq, E]
    auto k_proj = wk->forward(key);   // -> [B, Sk, E]
    auto v_proj = wv->forward(value); // -> [B, Sv, E]

    // 2. Split Heads: Reshape & Permute
    auto q_reshaped = q_proj->reshape({batch_size, seq_len_q, num_heads, head_dim});
    auto q_final    = q_reshaped->permute({0, 2, 1, 3}); // B, N, Sq, H_D

    auto k_reshaped = k_proj->reshape({batch_size, seq_len_k, num_heads, head_dim});
    auto k_final    = k_reshaped->permute({0, 2, 1, 3}); // B, N, Sk, H_D

    auto v_reshaped = v_proj->reshape({batch_size, seq_len_k, num_heads, head_dim}); // seq_len_v == seq_len_k
    auto v_final    = v_reshaped->permute({0, 2, 1, 3}); // B, N, Sv, H_D

    // Reshape Q, K, V for scaled_dot_product_attention: [B*N, S, H_D]
    auto q_sdpa_in = q_final->reshape({batch_size * num_heads, seq_len_q, head_dim});
    auto k_sdpa_in = k_final->reshape({batch_size * num_heads, seq_len_k, head_dim});
    auto v_sdpa_in = v_final->reshape({batch_size * num_heads, seq_len_k, head_dim}); // seq_len_v == seq_len_k

    // Mask processing
    std::shared_ptr<Tensor> sdpa_mask = nullptr;
    if (mask) {
        // Current SDPA expects mask [B_sdpa, Sq, Sk] or [1, Sq, Sk]
        // B_sdpa = batch_size * num_heads
        // Original mask can be [B, Sq, Sk] or [1, Sq, Sk] (broadcastable over batch)
        // or [Sq, Sk] (broadcastable over batch and heads)
        if (mask->ndim() == 3 && mask->shape[0] == batch_size && mask->shape[1] == seq_len_q && mask->shape[2] == seq_len_k) {
            // Expand mask from [B, Sq, Sk] to [B*N, Sq, Sk] by repeating each B item N times
            std::vector<std::shared_ptr<Value>> expanded_mask_data;
            expanded_mask_data.reserve(mask->numel() * num_heads);
            for (int b_idx = 0; b_idx < batch_size; ++b_idx) {
                for (int h_idx = 0; h_idx < num_heads; ++h_idx) {
                    for (int sq_idx = 0; sq_idx < seq_len_q; ++sq_idx) {
                        for (int sk_idx = 0; sk_idx < seq_len_k; ++sk_idx) {
                            expanded_mask_data.push_back(mask->get({b_idx, sq_idx, sk_idx}));
                        }
                    }
                }
            }
            sdpa_mask = Tensor::from_values(expanded_mask_data, {batch_size * num_heads, seq_len_q, seq_len_k});
        } else if (mask->ndim() == 3 && mask->shape[0] == 1 && mask->shape[1] == seq_len_q && mask->shape[2] == seq_len_k) {
            // Expand mask from [1, Sq, Sk] to [B*N, Sq, Sk]
             std::vector<std::shared_ptr<Value>> expanded_mask_data;
            expanded_mask_data.reserve(mask->numel() * batch_size * num_heads);
            for(int bn_idx = 0; bn_idx < batch_size * num_heads; ++bn_idx) {
                for (int sq_idx = 0; sq_idx < seq_len_q; ++sq_idx) {
                    for (int sk_idx = 0; sk_idx < seq_len_k; ++sk_idx) {
                        expanded_mask_data.push_back(mask->get({0, sq_idx, sk_idx}));
                    }
                }
            }
            sdpa_mask = Tensor::from_values(expanded_mask_data, {batch_size*num_heads, seq_len_q, seq_len_k});
        } else if (mask->ndim() == 2 && mask->shape[0] == seq_len_q && mask->shape[1] == seq_len_k) {
            // Expand mask from [Sq, Sk] to [B*N, Sq, Sk]
            std::vector<std::shared_ptr<Value>> expanded_mask_data;
            expanded_mask_data.reserve(mask->numel() * batch_size * num_heads);
            for (int bn_idx = 0; bn_idx < batch_size * num_heads; ++bn_idx) {
                 for (int sq_idx = 0; sq_idx < seq_len_q; ++sq_idx) {
                    for (int sk_idx = 0; sk_idx < seq_len_k; ++sk_idx) {
                        expanded_mask_data.push_back(mask->get({sq_idx, sk_idx}));
                    }
                }
            }
            sdpa_mask = Tensor::from_values(expanded_mask_data, {batch_size*num_heads, seq_len_q, seq_len_k});
        }
        else {
            throw std::runtime_error("MultiHeadAttention: Mask shape not supported for automatic expansion. Expected [B, Sq, Sk], [1, Sq, Sk], or [Sq, Sk]. Got shape with ndim " + std::to_string(mask->ndim()));
        }
    }

    // 3. Scaled Dot-Product Attention
    auto attention_output_sdpa = micrograd_nn::scaled_dot_product_attention(q_sdpa_in, k_sdpa_in, v_sdpa_in, sdpa_mask);
    // attention_output_sdpa shape: [(B*N), Sq, H_D]

    // 4. Concatenate Heads: Reshape & Permute
    auto attention_output_heads = attention_output_sdpa->reshape({batch_size, num_heads, seq_len_q, head_dim});
    auto attention_permuted = attention_output_heads->permute({0, 2, 1, 3}); // -> [B, Sq, N, H_D]
    auto concatenated_output = attention_permuted->reshape({batch_size, seq_len_q, embed_dim});

    // 5. Final Linear Projection
    return wo->forward(concatenated_output);
}

// This is the Layer::forward override
std::shared_ptr<Tensor> MultiHeadAttention::forward(const std::shared_ptr<Tensor>& x) {
    // By default, self-attention without a mask
    return this->forward(x, x, x, nullptr);
}

// The version with explicit Q, K, V and optional mask remains as the primary worker method.
// Its signature is:
// std::shared_ptr<Tensor> MultiHeadAttention::forward(
//     const std::shared_ptr<Tensor>& query,
//     const std::shared_ptr<Tensor>& key,
//     const std::shared_ptr<Tensor>& value,
//     const std::shared_ptr<Tensor>& mask)
//
// The implementation for the QKV forward is above and is correct.
// No change needed for the QKV forward method itself, only for the self-attention overload.

std::vector<std::shared_ptr<Value>> MultiHeadAttention::parameters() const {
    std::vector<std::shared_ptr<Value>> params;
    if (wq) { auto p = wq->parameters(); params.insert(params.end(), p.begin(), p.end()); }
    if (wk) { auto p = wk->parameters(); params.insert(params.end(), p.begin(), p.end()); }
    if (wv) { auto p = wv->parameters(); params.insert(params.end(), p.begin(), p.end()); }
    if (wo) { auto p = wo->parameters(); params.insert(params.end(), p.begin(), p.end()); }
    return params;
}
