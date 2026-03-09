#include "transformer.hpp"


// ---
// KV Cache — stores K and V tensors for each layer
KVCache::KVCache(const TransformerConfig& config) {

}

void KVCache::clear() {
    k_cache.clear();
    v_cache.clear();
    current_pos = 0;
}

// ---
// Attention Layer
Tensor AttentionLayer::forward(const Tensor& x, int pos, 
    const TransformerWeights& w, KVCache& kv_cache, 
    int layer_idx, const TransformerConfig& config) {

}

// ---
// FFN Layer
Tensor FFNLayer::forward(const Tensor& x, const TransformerWeights& w,
                        int layer_idx, const TransformerConfig& config) {

}

// ---
// Full Model
Transformer::Transformer(const std::string& model_path) : tokenizer_(model_path + "/tokenizer.model") {
    load(model_path);
    // TODO: finish construction logic
}

// run forward pass, return logits over vocabulary
Tensor Transformer::forward(const std::vector<int>& token_ids, int pos) {

}

// greedy sample — just argmax
float Transformer::greedy_sample(Tensor& logits) {
    float* arr = logits.data();
    return *std::max_element(arr, arr + logits.numel());
}

// temperature sample
int Transformer::sample(const Tensor& logits, float temperature = 1.0f) {
    
}

const TransformerConfig& Transformer::config() const {
    return config_;
}

// Private
void Transformer::load(const std::string& model_path) {

}