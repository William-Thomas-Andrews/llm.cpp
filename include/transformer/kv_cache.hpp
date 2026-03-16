#pragma once

#include "config.hpp"
#include "tensor.hpp"

#include <vector>

// ---
// KV Cache — stores K and V tensors for each layer

struct KVCache {
    // one entry per layer
    std::vector<Tensor> k_cache;    // [num_layers, max_seq_len, kv_dim]
    std::vector<Tensor> v_cache;    // [num_layers, max_seq_len, kv_dim]
    int current_pos = 0;            // how many tokens have been processed
    KVCache();
    KVCache(const TransformerConfig& config);
    void clear();
};