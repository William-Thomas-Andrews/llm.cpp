#include "kv_cache.hpp"

// ---
// KV Cache — stores K and V tensors for each layer

KVCache::KVCache() {
    ;
}
KVCache::KVCache(const TransformerConfig& config) { 
    int kv_dim = config.num_kv_heads * (config.d_model / config.num_heads);  // 4 * 64 = 256
    std::array<int, Tensor::MAX_DIMS> shape = {};
    shape[0] = config.max_seq_len;
    shape[1] = kv_dim;

    k_cache.resize(config.num_layers);
    v_cache.resize(config.num_layers);

    for (int i = 0; i < config.num_layers; i++) {
        k_cache[i] = Tensor(shape, 2);
        v_cache[i] = Tensor(shape, 2);
    }
}

void KVCache::clear() {
    for (int i = 0; i < (int)k_cache.size(); i++) {
        k_cache[i].fill(0.0f);
        v_cache[i].fill(0.0f);
    }
    current_pos = 0;
}

