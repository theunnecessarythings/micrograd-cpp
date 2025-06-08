#include "nn/attention.h"
#include <stdexcept> // For std::runtime_error, std::invalid_argument
#include <limits>    // For std::numeric_limits
#include <vector>    // For std::vector
#include <cmath>     // For std::sqrt (already in .h but good for .cpp too)


#include <vector>    // For std::vector
#include <cmath>     // For std::sqrt (already in .h but good for .cpp too)


namespace micrograd_nn {

std::shared_ptr<Tensor> scaled_dot_product_attention(
    const std::shared_ptr<Tensor>& query,
    const std::shared_ptr<Tensor>& key,
    const std::shared_ptr<Tensor>& value,
    const std::shared_ptr<Tensor>& mask,
    float dropout_p // Not used in this implementation pass
) {
    if (!query || !key || !value) {
        throw std::invalid_argument("Query, Key, and Value tensors must not be null.");
    }

    // Assuming Q, K, V are 3D: [batch_size, seq_len, features]
    if (query->ndim() != 3 || key->ndim() != 3 || value->ndim() != 3) {
        throw std::invalid_argument("Q, K, V must be 3D tensors [batch, seq_len, features]. Query: " +
            std::to_string(query->ndim()) + "D, Key: " + std::to_string(key->ndim()) + "D, Value: " + std::to_string(value->ndim()) + "D.");
    }
    if (query->shape[0] != key->shape[0] || query->shape[0] != value->shape[0]) { // Batch size
        throw std::runtime_error("Batch sizes of Q, K, V must match. Q_batch: " + std::to_string(query->shape[0]) +
                                 ", K_batch: " + std::to_string(key->shape[0]) + ", V_batch: " + std::to_string(value->shape[0]));
    }
    if (query->shape[2] != key->shape[2]) { // d_k must match
        throw std::runtime_error("Feature dimension (d_k) of Query and Key must match. Q_dk: " + std::to_string(query->shape[2]) +
                                 ", K_dk: " + std::to_string(key->shape[2]));
    }
    if (key->shape[1] != value->shape[1]) { // seq_len_k must match seq_len_v
        throw std::runtime_error("Sequence length of Key and Value (seq_len_kv) must match. K_seq_len: " + std::to_string(key->shape[1]) +
                                 ", V_seq_len: " + std::to_string(value->shape[1]));
    }

    int d_k = query->shape[2];
    if (d_k == 0) throw std::runtime_error("Feature dimension d_k cannot be zero.");
    float scale_factor_float = 1.0f / std::sqrt(static_cast<float>(d_k));
    // scale_factor does not need to be a Value object if it's not part of backprop path,
    // or if element-wise tensor * float op handles it. Tensor has operator*(float).


    // 1. Calculate Q @ K.T
    // K is [B, S_k, D_k]. We need K_T as [B, D_k, S_k]
    // Q is [B, S_q, D_k]
    // scores = Q @ K_T -> [B, S_q, S_k]

    // Manual transpose of the last two dimensions for each batch item in Key tensor
    std::vector<std::shared_ptr<Value>> kt_data_flat;
    kt_data_flat.reserve(key->numel());
    std::vector<int> kt_shape = {key->shape[0], key->shape[2], key->shape[1]}; // B, Dk, Sk

    for(int b=0; b < key->shape[0]; ++b) {
        for(int d=0; d < key->shape[2]; ++d) { // Iterate Dk first for K_T
            for(int s=0; s < key->shape[1]; ++s) { // Iterate Sk second
                kt_data_flat.push_back(key->get({b,s,d}));
            }
        }
    }
    auto key_transposed = Tensor::from_values(kt_data_flat, kt_shape);

    // Matmul requires 2D tensors. We need batched matmul or loop.
    // For now, let's implement batched matmul logic here.
    // Q: [B, Sq, Dk], K_T: [B, Dk, Sk] -> scores: [B, Sq, Sk]
    if (query->shape[0] != key_transposed->shape[0]) { // Should be caught by earlier batch check but defensive
        throw std::runtime_error("Batch size mismatch for Q and K_transposed before matmul.");
    }
    std::vector<std::shared_ptr<Value>> scores_flat_data;
    scores_flat_data.reserve(query->shape[0] * query->shape[1] * key_transposed->shape[2]);
    std::vector<int> scores_shape = {query->shape[0], query->shape[1], key_transposed->shape[2]};

    for(int b=0; b < query->shape[0]; ++b) { // For each item in batch
        // Extract 2D slices for matmul: Q[b,:,:] and K_T[b,:,:]
        std::vector<std::shared_ptr<Value>> q_slice_data;
        for(int r=0; r < query->shape[1]; ++r) for(int c=0; c < query->shape[2]; ++c) q_slice_data.push_back(query->get({b,r,c}));
        auto q_slice = Tensor::from_values(q_slice_data, {query->shape[1], query->shape[2]});

        std::vector<std::shared_ptr<Value>> kt_slice_data;
        for(int r=0; r < key_transposed->shape[1]; ++r) for(int c=0; c < key_transposed->shape[2]; ++c) kt_slice_data.push_back(key_transposed->get({b,r,c}));
        auto kt_slice = Tensor::from_values(kt_slice_data, {key_transposed->shape[1], key_transposed->shape[2]});

        auto scores_slice = q_slice->matmul(kt_slice); // Standard 2D matmul
        scores_flat_data.insert(scores_flat_data.end(), scores_slice->data.begin(), scores_slice->data.end());
    }
    auto scores = Tensor::from_values(scores_flat_data, scores_shape);


    // 2. Scale scores
    auto scaled_scores = (*scores) * scale_factor_float; // Use Tensor::operator*(float)

    // 3. Apply mask (if provided)
    std::shared_ptr<Tensor> masked_scores = scaled_scores;
    if (mask) {
        // Shape check for mask: [B, Sq, Sk] or [1, Sq, Sk]
        bool broadcast_dim0 = (mask->shape[0] == 1 && scaled_scores->shape[0] > 1);
        if (!((mask->shape[0] == scaled_scores->shape[0] || broadcast_dim0) &&
              mask->shape[1] == scaled_scores->shape[1] &&
              mask->shape[2] == scaled_scores->shape[2])) {
            throw std::runtime_error("Mask shape incompatible with scores tensor for broadcasting.");
        }

        std::vector<std::shared_ptr<Value>> temp_masked_data;
        temp_masked_data.reserve(scaled_scores->numel());
        // Using a very large negative number by creating a Value from it
        auto large_negative_fill = std::make_shared<Value>(-1e9f, "mask_fill_value");

        for(int b=0; b < scaled_scores->shape[0]; ++b) {
            for(int sq=0; sq < scaled_scores->shape[1]; ++sq) {
                for(int sk=0; sk < scaled_scores->shape[2]; ++sk) {
                    auto score_val = scaled_scores->get({b,sq,sk});
                    int mask_b_idx = broadcast_dim0 ? 0 : b;
                    // Assuming mask data is 0.0f for non-masked, 1.0f for masked (or any non-zero for masked)
                    bool is_masked = static_cast<bool>(std::round(mask->get({mask_b_idx, sq, sk})->data));
                    if (is_masked) {
                        // For autograd, adding large negative is one way.
                        // Or, if softmax can take a fill_value, that's cleaner.
                        // Current Value ops don't allow direct replacement without breaking graph.
                        // So adding a large negative number (as a Value) is a graph-compatible way.
                        temp_masked_data.push_back(score_val + large_negative_fill);
                    } else {
                        temp_masked_data.push_back(score_val);
                    }
                }
            }
        }
        masked_scores = Tensor::from_values(temp_masked_data, scaled_scores->shape);
    }

    // 4. Softmax
    auto attention_weights = masked_scores->softmax(-1); // Use the new Tensor method (axis=-1 for last dim)

    // 5. Dropout (skipped in this pass)

    // 6. Multiply by Value V
    // V: [B, S_v, D_v] (where S_v == S_k)
    // attention_weights: [B, S_q, S_k]
    // Output: [B, S_q, D_v]
    // This is another batched matmul.
    if (attention_weights->shape[0] != value->shape[0]) {
         throw std::runtime_error("Batch size mismatch for attention_weights and V before matmul.");
    }
    std::vector<std::shared_ptr<Value>> output_flat_data;
    output_flat_data.reserve(attention_weights->shape[0] * attention_weights->shape[1] * value->shape[2]);
    std::vector<int> output_final_shape = {attention_weights->shape[0], attention_weights->shape[1], value->shape[2]};

    for(int b=0; b < attention_weights->shape[0]; ++b) {
        std::vector<std::shared_ptr<Value>> aw_slice_data;
        for(int r=0; r < attention_weights->shape[1]; ++r) for(int c=0; c < attention_weights->shape[2]; ++c) aw_slice_data.push_back(attention_weights->get({b,r,c}));
        auto aw_slice = Tensor::from_values(aw_slice_data, {attention_weights->shape[1], attention_weights->shape[2]});

        std::vector<std::shared_ptr<Value>> v_slice_data;
        for(int r=0; r < value->shape[1]; ++r) for(int c=0; c < value->shape[2]; ++c) v_slice_data.push_back(value->get({b,r,c}));
        auto v_slice = Tensor::from_values(v_slice_data, {value->shape[1], value->shape[2]});

        auto output_slice = aw_slice->matmul(v_slice);
        output_flat_data.insert(output_flat_data.end(), output_slice->data.begin(), output_slice->data.end());
    }
    auto output = Tensor::from_values(output_flat_data, output_final_shape);

    return output;
}

} // namespace micrograd_nn
