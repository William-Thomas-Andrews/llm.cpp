#pragma once

#include "tensor.hpp"
#include "ops.hpp"
#include "tokenizer.hpp"
#include "json.hpp"
#include "attention_layer.hpp"
#include "ffn_layer.hpp"

#include <string>
#include <vector>
#include <array>
#include <fstream>
#include <cstring>
#include <filesystem>
#include <string>
#include <random>


// ---
// Full Model

class Transformer {
public:
    Transformer(const std::string& model_path);

    // embed token
    Tensor embed(int token_id);

    // run forward pass, return logits over vocabulary
    Tensor forward(int token_id, int pos);

    // greedy sample — just argmax
    int greedy_sample(Tensor& logits);

    // temperature sample
    int sample(std::vector<float>& logits, int n, float temperature, float top_p = 0.9f,
               const std::vector<int>& past_tokens = {}, float rep_penalty = 1.3f);

    const TransformerConfig& config() const;
    const TransformerWeights& weights() const;
    const KVCache& kv_cache() const;

    const Tokenizer& tokenizer() const { return tokenizer_; }

private:
    TransformerConfig config_;
    TransformerWeights weights_;
    KVCache kv_cache_;
    Tokenizer tokenizer_;

    // void load(const std::string& model_path);
    std::array<int, Tensor::MAX_DIMS> make_shape(int* dims, int ndim); // helper
    void load_config(const std::string& model_path);
    void load_weights(const std::string& model_path);
    void load_model(const std::string& model_path);
};