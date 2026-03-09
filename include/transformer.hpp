// transformer.hpp
#pragma once
#include <string>
#include <vector>
#include <array>
#include "tensor.hpp"
#include "ops.hpp"

// ---
// Config — all hyperparameters loaded from config.json

struct TransformerConfig {
    int d_model;        // hidden dimension (2048 for TinyLLaMA)
    int num_heads;      // attention heads (32)
    int num_kv_heads;   // key/value heads for GQA (4 for TinyLLaMA)
    int num_layers;     // transformer layers (22)
    int ffn_dim;        // feed forward dimension (5632)
    int vocab_size;     // vocabulary size (32000)
    int max_seq_len;    // maximum sequence length (2048)
    float rope_theta;   // RoPE base frequency (10000.0)
    float eps;          // RMSNorm epsilon (1e-5)
};

// ---
// Weights — all model tensors loaded from model.bin

struct TransformerWeights {
    // embedding
    Tensor token_embedding;     // [vocab_size, d_model]

    // per layer weights (one per layer)
    std::vector<Tensor> attn_norm;      // [num_layers, d_model]
    std::vector<Tensor> ffn_norm;       // [num_layers, d_model]

    // attention projections
    std::vector<Tensor> wq;     // [num_layers, d_model, d_model]
    std::vector<Tensor> wk;     // [num_layers, d_model, kv_dim]
    std::vector<Tensor> wv;     // [num_layers, d_model, kv_dim]
    std::vector<Tensor> wo;     // [num_layers, d_model, d_model]

    // FFN projections (SwiGLU has 3 weight matrices)
    std::vector<Tensor> w_gate; // [num_layers, ffn_dim, d_model]
    std::vector<Tensor> w_up;   // [num_layers, ffn_dim, d_model]
    std::vector<Tensor> w_down; // [num_layers, d_model, ffn_dim]

    // final norm + output
    Tensor final_norm;          // [d_model]
    Tensor lm_head;             // [vocab_size, d_model]
};

// ---
// KV Cache — stores K and V tensors for each layer

struct KVCache {
    // one entry per layer
    std::vector<Tensor> k_cache;    // [num_layers, max_seq_len, kv_dim]
    std::vector<Tensor> v_cache;    // [num_layers, max_seq_len, kv_dim]
    int current_pos = 0;            // how many tokens have been processed

    KVCache(const TransformerConfig& config);
    void clear();
};

// ---
// Attention Layer

struct AttentionLayer {
    Tensor forward(const Tensor& x, int pos,
                   const TransformerWeights& w,
                   KVCache& kv_cache,
                   int layer_idx,
                   const TransformerConfig& config);
};

// ---
// FFN Layer

struct FFNLayer {
    Tensor forward(const Tensor& x,
                   const TransformerWeights& w,
                   int layer_idx,
                   const TransformerConfig& config);
};

// ---
// Full Model

class Transformer {
public:
    Transformer(const std::string& model_path);

    // run forward pass, return logits over vocabulary
    Tensor forward(const std::vector<int>& token_ids, int pos);

    // greedy sample — just argmax
    float greedy_sample(Tensor& logits);

    // temperature sample
    int sample(const Tensor& logits, float temperature = 1.0f);

    const TransformerConfig& config() const;

private:
    TransformerConfig config_;
    TransformerWeights weights_;
    KVCache kv_cache_;

    void load(const std::string& model_path);
};