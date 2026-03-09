#include "transformer.hpp"


// ---
// KV Cache — stores K and V tensors for each layer
KVCache::KVCache(const TransformerConfig& config) {

}

void KVCache::clear() {

}

// ---
// Attention Layer
Tensor AttentionLayer::forward(const Tensor& x, int pos,
                            const TransformerWeights& w,
                            KVCache& kv_cache,
                            int layer_idx,
                            const TransformerConfig& config)
{

}

// ---
// FFN Layer
Tensor FFNLayer::forward(const Tensor& x,
                        const TransformerWeights& w,
                        int layer_idx,
                        const TransformerConfig& config) 
{



}

// ---
// Full Model
Transformer::Transformer(const std::string& model_path) {

}

// run forward pass, return logits over vocabulary
Tensor Transformer::forward(const std::vector<int>& token_ids, int pos) {

}

// greedy sample — just argmax
int Transformer::greedy_sample(const Tensor& logits) {

}

// temperature sample
int Transformer::sample(const Tensor& logits, float temperature = 1.0f) {

}

const TransformerConfig& Transformer::config() const {

}

// Private
void Transformer::load(const std::string& model_path) {

}