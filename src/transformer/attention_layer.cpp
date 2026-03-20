#include "attention_layer.hpp"

// ---
// Attention Layer

float AttentionLayer::dot_product(int8_t* ptr_1, int8_t* ptr_2, Tensor& tensor_1, Tensor& tensor_2, int head_dim) {
    int32_t acc = 0;
    for (int i = 0; i < head_dim; i++)
        acc += (int32_t)ptr_1[i] * (int32_t)ptr_2[i];
    
    return acc * (tensor_1.q_scale() * tensor_2.q_scale());
}

void AttentionLayer::softmax(std::vector<float>& scores) {
    float max_val = *std::max_element(scores.begin(), scores.end());
    float summation = 0.0f;
    for (int i = 0; i < scores.size(); i++) {
        scores[i]= std::exp(scores[i] - max_val);
        summation += scores[i];
    }
    for (int j = 0; j < scores.size(); j++)
        scores[j] = scores[j] / summation;
}

Tensor AttentionLayer::forward(Tensor& X, int pos, TransformerWeights& W, KVCache& kv_cache, int layer_idx, const TransformerConfig& config) {
    int head_dim    = config.d_model / config.num_heads;
    int kv_dim      = config.num_kv_heads * head_dim;
    int kv_per_head = config.num_heads / config.num_kv_heads;
    float scale     = 1.0f / std::sqrt(head_dim); 

    // 1. project X into Q, K, V — weights are [out, in] (HF format), so transB=true
    Tensor Q = matmul(X, W.wq[layer_idx], LIB::BLOCKED, true); // implicit transpose = true
    Tensor K = matmul(X, W.wk[layer_idx], LIB::BLOCKED, true); // implicit transpose = true
    Tensor V = matmul(X, W.wv[layer_idx], LIB::BLOCKED, true); // implicit transpose = true

    // 2. apply RoPE to Q and K
    rope(Q, K, head_dim, pos);

    // 3. write K and V into cache at position pos
    int8_t* k_cache_ptr = kv_cache.k_cache[layer_idx].data();
    int8_t* v_cache_ptr = kv_cache.v_cache[layer_idx].data();
    memcpy(k_cache_ptr + pos * kv_dim, K.data(), kv_dim * sizeof(int8_t));
    memcpy(v_cache_ptr + pos * kv_dim, V.data(), kv_dim * sizeof(int8_t));

    // 4. per-head attention
    // output accumulator [1, d_model]
    std::array<int, Tensor::MAX_DIMS> out_shape = {};
    out_shape[0] = 1;
    out_shape[1] = config.d_model;
    Tensor output(out_shape, 2);  // zero initialized
    std::vector<float> out_head(output.numel()); // zero initialized just like its Tensor above

    // scores buffer [pos+1] — reused each head
    std::vector<float> scores(pos + 1);

    for (int h = 0; h < config.num_heads; h++) {
        // which KV head does this belong to?
        int kv_head = h / kv_per_head;

        int8_t* q_head = Q.data() + h * head_dim;
        // int8_t* out_head = output.data() + h * head_dim; /

        // compute scores dot(q_head, each cached k for kv_h)
        for (int t = 0; t <= pos; t++) {
            float dot = 0.0f;
            int8_t* k_t = k_cache_ptr + t * kv_dim + kv_head * head_dim;
            scores[t] = scale * dot_product(q_head, k_t, Q, kv_cache.k_cache[layer_idx], head_dim);
        }

        // softmax over scores[0..pos]
        softmax(scores);

        // weighted sum of V: out_head += scores[t] * v_cache[t]
        for (int t = 0; t <= pos; t++) {
            // int8_t* v_t = v_cache_ptr + t * kv_dim + kv_head * head_dim; /
            for (int d = 0; d < head_dim; d++) {
                float v_val = kv_cache.v_cache[layer_idx].dequantize(v_cache_ptr[d + t * kv_dim + kv_head * head_dim]);
                out_head[h * head_dim + d] += scores[t] * v_val;
            }
        }
    }

    // Compute output scale from actual output range
    float r_max = -INFINITY, r_min = +INFINITY;
    for (int i = 0; i < (int)out_head.size(); i++) {
        r_max = std::max(r_max, out_head[i]);
        r_min = std::min(r_min, out_head[i]);
    }
    float scale = std::max(std::abs(r_max), std::abs(r_min)) / q_max;
    if (scale == 0.0f) scale = 1e-8f;
    output.set_scale(scale);

    // Re-Quantize
    for (int i = 0; i < output.numel(); i++)
        output.data()[i] = output.quantize(out_head[i]);



    // 5. output projection — wo is [out, in] (HF format), transB=true
    Tensor result = matmul(output, W.wo[layer_idx], LIB::BLOCKED, true);
    return result;
}
