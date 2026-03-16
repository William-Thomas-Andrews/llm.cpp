#include "transformer.hpp"


// ---
// Full Model

Transformer::Transformer(const std::string& model_path) : tokenizer_(model_path + "/tokenizer.model"), kv_cache_() {
    load_model(model_path);
    kv_cache_ = KVCache(config_);
}

Tensor Transformer::embed(int token_id) {
    int embed_rows = weights_.token_embedding.shape_at(0);
    if (token_id < 0 || token_id >= embed_rows)
        token_id = 0;  // clamp to UNK rather than reading out-of-bounds

    std::array<int, Tensor::MAX_DIMS> shape = {};
    shape[0] = 1;
    shape[1] = config_.d_model;

    // non-owning view into row token_id of the embedding table
    int8_t* row = weights_.token_embedding.data() + (token_id * config_.d_model);
    return Tensor(row, config_.d_model, shape, 2);
}

// run forward pass, return logits over vocabulary
Tensor Transformer::forward(int token_id, int pos) {
    Tensor X = embed(token_id);
    FFNLayer ffn;
    AttentionLayer attn;
    
    // Loops through the config_.num_layers layers (from num_layers in TransformerConfig)
    for (int layer = 0; layer < config_.num_layers; layer++) {
        Tensor attn_input = rmsnorm(X, weights_.attn_norm[layer]);
        Tensor attn_out = attn.forward(attn_input, pos, weights_, kv_cache_, layer, config_);
        X = add(X, attn_out);

        Tensor ffn_input = rmsnorm(X, weights_.ffn_norm[layer]);
        Tensor ffn_out = ffn.forward(ffn_input, weights_, layer, config_);
        X = add(X, ffn_out);
    }

    X = rmsnorm(X, weights_.final_norm);
    
    // logits: lm_head is [vocab_size, d_model]; pass directly with transB=true
    return matmul(X, weights_.lm_head, LIB::BLAS, true);
}

// greedy sample — just argmax
int Transformer::greedy_sample(Tensor& logits) {
    int8_t* arr = logits.data();
    return std::max_element(arr, arr + logits.numel()) - arr;
}

// temperature + top-p (nucleus) sample
int Transformer::sample(float* logits, int n, float temperature, float top_p,
                        const std::vector<int>& past_tokens, float rep_penalty) {
    // Apply repetition penalty: divide logit of any already-seen token by rep_penalty
    if (rep_penalty != 1.0f) {
        for (int id : past_tokens)
            logits[id] /= rep_penalty;
    }
    if (temperature != 1.0f)
        scale(logits, n, 1.0f / temperature);
    // std::cout << "Here sample" << std::endl;
    softmax(logits, n);
    // std::cout << "Here sample" << std::endl;
    float* probs = logits;

    // Build sorted index list by descending probability
    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        return probs[a] > probs[b];
    });

    // Keep only the nucleus (top tokens summing to top_p), renormalize
    float cumsum = 0.0f;
    int cutoff = n;
    for (int i = 0; i < n; i++) {
        cumsum += probs[indices[i]];
        if (cumsum >= top_p) { cutoff = i + 1; break; }
    }

    // Sample from the nucleus
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, cumsum);
    float r = dis(gen);

    float running = 0.0f;
    for (int i = 0; i < cutoff; i++) {
        running += probs[indices[i]];
        if (r < running) return indices[i];
    }
    return indices[cutoff - 1];
}

const TransformerConfig& Transformer::config() const {
    return config_;
}

const TransformerWeights& Transformer::weights() const {
    return weights_;
}

const KVCache& Transformer::kv_cache() const {
    return kv_cache_;
}

void Transformer::load_config(const std::string& model_path) {
    // std::filesystem::path currentDir = std::filesystem::current_path();
    // Convert the path to a string and print it
    // std::cout << "Current working directory: " << currentDir.string() << std::endl;
    std::ifstream f(model_path + "/config.json");
    if (!f.is_open()) throw std::runtime_error("Cannot open " + model_path + "/config.json");
    auto j = nlohmann::json::parse(f);

    config_.d_model      = j["hidden_size"];
    config_.num_heads    = j["num_attention_heads"];
    config_.num_kv_heads = j["num_key_value_heads"];
    config_.num_layers   = j["num_hidden_layers"];
    config_.ffn_dim      = j["intermediate_size"];
    config_.vocab_size   = j["vocab_size"];
    config_.max_seq_len  = j["max_position_embeddings"];
    config_.rope_theta   = j["rope_theta"];
    config_.eps          = j["rms_norm_eps"];
}


// helper — make a 1D tensor shape array
std::array<int, Tensor::MAX_DIMS> Transformer::make_shape(int* dims, int ndim) {
    std::array<int, Tensor::MAX_DIMS> shape = {};
    for (int i = 0; i < ndim; i++) shape[i] = dims[i];
    return shape;
}

void Transformer::load_weights(const std::string& model_path) {
    std::string bin_path = model_path + "/model.bin";
    FILE* f = fopen(bin_path.c_str(), "rb");
    if (!f) throw std::runtime_error("Cannot open model.bin");

    // read number of tensors
    int num_tensors;
    fread(&num_tensors, sizeof(int), 1, f);
    printf("loading %d tensors...\n", num_tensors);

    // pre-size per-layer vectors
    int L = config_.num_layers;
    weights_.attn_norm.resize(L);
    weights_.ffn_norm.resize(L);
    weights_.wq.resize(L);
    weights_.wk.resize(L);
    weights_.wv.resize(L);
    weights_.wo.resize(L);
    weights_.w_gate.resize(L);
    weights_.w_up.resize(L);
    weights_.w_down.resize(L);

    for (int t = 0; t < num_tensors; t++) {
        // read name
        char name[64];
        fread(name, 1, 64, f);

        // read ndim
        int ndim;
        fread(&ndim, sizeof(int), 1, f);

        // read shape (always 8 ints in file)
        int dims[8] = {};
        fread(dims, sizeof(int), 8, f);

        // compute numel
        size_t numel = 1;
        for (int i = 0; i < ndim; i++) numel *= dims[i];

        // read scale factor
        float scale = 1.0f;
        fread(&scale, sizeof(float), 1, f);

        // allocate tensor and read data
        auto shape = make_shape(dims, ndim);
        Tensor tensor(shape, ndim);
        fread(tensor.data(), sizeof(int8_t), numel, f);
        tensor.set_scale(scale);

        // route tensor to the right field by name
        std::string n(name);

        if (n == "model.embed_tokens.weight") {
            weights_.token_embedding = std::move(tensor);
        } else if (n == "model.norm.weight") {
            weights_.final_norm = std::move(tensor);
        } else if (n == "lm_head.weight") {
            weights_.lm_head = std::move(tensor);
        } else {
            // parse layer index from "model.layers.N.xxx"
            int layer = -1;
            sscanf(name, "model.layers.%d.", &layer);
            if (layer < 0 || layer >= L) {
                printf("  warning: unrecognized tensor %s, skipping\n", name);
                continue;
            }

            if      (strstr(name, "input_layernorm.weight"))        weights_.attn_norm[layer] = std::move(tensor);
            else if (strstr(name, "post_attention_layernorm.weight"))weights_.ffn_norm[layer]  = std::move(tensor);
            else if (strstr(name, "self_attn.q_proj.weight"))        weights_.wq[layer]        = std::move(tensor);
            else if (strstr(name, "self_attn.k_proj.weight"))        weights_.wk[layer]        = std::move(tensor);
            else if (strstr(name, "self_attn.v_proj.weight"))        weights_.wv[layer]        = std::move(tensor);
            else if (strstr(name, "self_attn.o_proj.weight"))        weights_.wo[layer]        = std::move(tensor);
            else if (strstr(name, "mlp.gate_proj.weight"))           weights_.w_gate[layer]    = std::move(tensor);
            else if (strstr(name, "mlp.up_proj.weight"))             weights_.w_up[layer]      = std::move(tensor);
            else if (strstr(name, "mlp.down_proj.weight"))           weights_.w_down[layer]    = std::move(tensor);
            else printf("  warning: unrecognized tensor %s\n", name);
        }
    }

    fclose(f);
    printf("weights loaded.\n");
}


// Private
void Transformer::load_model(const std::string& model_path) {
    load_config(model_path);
    load_weights(model_path);
}