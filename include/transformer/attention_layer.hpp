#pragma once

#include "tensor.hpp"
#include "config.hpp"
#include "weights.hpp"
#include "kv_cache.hpp"
#include "ops.hpp"

// ---
// Attention Layer

struct AttentionLayer {
    int8_t dot_product(int8_t* ptr_1, int8_t* ptr_2, int head_dim);
    void softmax(std::vector<int8_t>& scores);
    Tensor forward(Tensor& X, int pos,
                    TransformerWeights& W,
                    KVCache& kv_cache,
                    int layer_idx,
                    const TransformerConfig& config);
};