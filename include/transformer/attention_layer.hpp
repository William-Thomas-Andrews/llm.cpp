#pragma once

#include "tensor.hpp"
#include "config.hpp"
#include "weights.hpp"
#include "kv_cache.hpp"
#include "ops.hpp"

// ---
// Attention Layer

struct AttentionLayer {
    float dot_product(int8_t* ptr_1, int8_t* ptr_2, Tensor& tensor_1, Tensor& tensor_2, int head_dim);
    void softmax(std::vector<float>& scores);
    Tensor forward(Tensor& X, int pos,
                    TransformerWeights& W,
                    KVCache& kv_cache,
                    int layer_idx,
                    const TransformerConfig& config);
};