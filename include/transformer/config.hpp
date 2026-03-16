#pragma once

#include <cstdint>

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
    int8_t rope_theta;   // RoPE base frequency (10000.0)
    int8_t eps;          // RMSNorm epsilon (1e-5)
};