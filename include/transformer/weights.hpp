#pragma once

#include "tensor.hpp"
#include <vector>

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