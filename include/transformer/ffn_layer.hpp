#pragma once

#include "config.hpp"
#include "weights.hpp"
#include "ops.hpp"

// ---
// FFN (Feed Forward Network) Layer

struct FFNLayer {
    Tensor forward(Tensor& X,
                    TransformerWeights& W,
                    int layer_idx,
                    const TransformerConfig& config);
};
