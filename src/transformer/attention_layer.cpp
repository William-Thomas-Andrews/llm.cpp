#include "attention_layer.hpp"


// ---
// Attention Layer

float AttentionLayer::dot_product(float* ptr_1, float* ptr_2, int head_dim) {
    float sum = 0.0f;
    for (int i = 0; i < head_dim; i++)
        sum += ptr_1[i] * ptr_2[i];
    return sum;
}

void AttentionLayer::softmax(std::vector<float>& scores) {
    float max_val = *std::max_element(scores.begin(), scores.end());
    float summation = 0.0f;
    for (std::size_t i = 0; i < scores.size(); i++) {
        scores[i]= std::exp(scores[i] - max_val);
        summation += scores[i];
    }
    for (std::size_t j = 0; j < scores.size(); j++)
        scores[j] = scores[j] / summation;
}

// ---
// Attention Layer
Tensor AttentionLayer::forward(Tensor& X, int pos, TransformerWeights& W, KVCache& kv_cache, int layer_idx, const TransformerConfig& config) {
    int head_dim    = config.d_model / config.num_heads;  // 64
    int kv_dim      = config.num_kv_heads * head_dim;     // 256
    int kv_per_head = config.num_heads / config.num_kv_heads;  // 8 — GQA ratio
    float scale     = 1.0f / std::sqrt((float)head_dim);

    // 1. project X into Q, K, V — weights are [out, in] (HF format), so transB=true
    Tensor Q = matmul(X, W.wq[layer_idx], LIB::BLAS, true);
    Tensor K = matmul(X, W.wk[layer_idx], LIB::BLAS, true);
    Tensor V = matmul(X, W.wv[layer_idx], LIB::BLAS, true);

    // 2. apply RoPE to Q and K
    float* q_ptr = Q.data();
    float* k_ptr = K.data();
    for (int h = 0; h < config.num_heads; h++)
        rope_vector(q_ptr + h * head_dim, head_dim, pos);
    for (int h = 0; h < config.num_kv_heads; h++)
        rope_vector(k_ptr + h * head_dim, head_dim, pos);

    // 3. write K and V into cache at position pos
    float* k_cache_ptr = kv_cache.k_cache[layer_idx].data();
    float* v_cache_ptr = kv_cache.v_cache[layer_idx].data();
    memcpy(k_cache_ptr + pos * kv_dim, K.data(), kv_dim * sizeof(float));
    memcpy(v_cache_ptr + pos * kv_dim, V.data(), kv_dim * sizeof(float));

    // 4. per-head attention
    // output accumulator [1, d_model]
    std::array<int, Tensor::MAX_DIMS> out_shape = {};
    out_shape[0] = 1;
    out_shape[1] = config.d_model;
    Tensor output(out_shape, 2);  // zero initialized

    // scores buffer [pos+1] — reused each head
    std::vector<float> scores(pos + 1);

    for (int h = 0; h < config.num_heads; h++) {
        // which KV head does this belong to?
        int kv_head = h / kv_per_head;

        float* q_head = Q.data() + h * head_dim;
        float* out_head = output.data() + h * head_dim;

        // compute scores dot(q_head, each cached k for kv_h)
        for (int t = 0; t <= pos; t++) {
            float* k_t = k_cache_ptr + t * kv_dim + kv_head * head_dim;
            scores[t] = scale * dot_product(q_head, k_t, head_dim);
        }

        // softmax over scores[0..pos]
        softmax(scores);

        // weighted sum of V: out_head += scores[t] * v_cache[t]
        for (int t = 0; t <= pos; t++) {
            float* v_t = v_cache_ptr + t * kv_dim + kv_head * head_dim;
            for (int d = 0; d < head_dim; d++) {
                out_head[d] += scores[t] * v_t[d];
            }
        }
    }

    // 5. output projection — wo is [out, in] (HF format), transB=true
    Tensor result = matmul(output, W.wo[layer_idx], LIB::BLAS, true);
    return result;
}