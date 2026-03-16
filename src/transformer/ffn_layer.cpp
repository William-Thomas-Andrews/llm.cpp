#include "ffn_layer.hpp"


// ---
// FFN (Feed Forward Network) Layer
Tensor FFNLayer::forward(Tensor& X, TransformerWeights& W, int layer_idx, const TransformerConfig& config) {

    // gate = silu(X @ W_gate.T) — weights are [out, in] (HF format), so transB=true
    Tensor gate = silu(matmul(X, W.w_gate[layer_idx], LIB::BLAS, true));

    // up = X @ W_up.T
    Tensor up = matmul(X, W.w_up[layer_idx], LIB::BLAS, true);

    // hidden = gate * up
    Tensor hidden = mul(gate, up); // completion of swiglu

    // out = hidden @ W_down.T
    return matmul(hidden, W.w_down[layer_idx], LIB::BLAS, true);
}