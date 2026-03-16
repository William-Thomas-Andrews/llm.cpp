#include "transformer.hpp"


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

float AttentionLayer::dot_product(float* ptr_1, float* ptr_2, int head_dim) {
    float sum = 0.0f;
    for (int i = 0; i < head_dim; i++)
        sum += ptr_1[i] * ptr_2[i];
    return sum;
}

void AttentionLayer::softmax(std::vector<float>& scores) {
    float max_val = *std::max_element(scores.begin(), scores.end());
    float summation = 0.0f;
    for (int i = 0; i < scores.size(); i++) {
        scores[i]= std::exp(scores[i] - max_val);
        summation += scores[i];
    }
    for (int j = 0; j < scores.size(); j++)
        scores[j] = scores[j] / summation;
}

// ---
// Attention Layer
Tensor AttentionLayer::forward(Tensor& X, int pos, TransformerWeights& W, KVCache& kv_cache, int layer_idx, const TransformerConfig& config) {
    int head_dim    = config.d_model / config.num_heads;  // 64
    int kv_dim      = config.num_kv_heads * head_dim;     // 256
    int kv_per_head = config.num_heads / config.num_kv_heads;  // 8 — GQA ratio
    float scale     = 1.0f / std::sqrt((float)head_dim);
    // Tensor W_q_T = W.wq[layer_idx].transpose();
    // Tensor W_k_T = W.wk[layer_idx].transpose();
    // Tensor W_v_T = W.wv[layer_idx].transpose();    

    // 1. project X into Q, K, V — weights are [out, in] (HF format), so transB=true
    Tensor Q = matmul(X, W.wq[layer_idx], LIB::BLAS, true);
    Tensor K = matmul(X, W.wk[layer_idx], LIB::BLAS, true);
    Tensor V = matmul(X, W.wv[layer_idx], LIB::BLAS, true);

    // 2. apply RoPE to Q and K
    float* q_ptr = Q.data();
    float* k_ptr = K.data();
    for (int h = 0; h < config.num_heads; h++)
        rope_vector(q_ptr + h * head_dim, head_dim, pos);
    for (int h = 0; h < config.num_kv_heads; h++)
        rope_vector(k_ptr + h * head_dim, head_dim, pos);

    // 3. write K and V into cache at position pos
    float* k_cache_ptr = kv_cache.k_cache[layer_idx].data();
    float* v_cache_ptr = kv_cache.v_cache[layer_idx].data();
    memcpy(k_cache_ptr + pos * kv_dim, K.data(), kv_dim * sizeof(float));
    memcpy(v_cache_ptr + pos * kv_dim, V.data(), kv_dim * sizeof(float));

    // 4. per-head attention
    // output accumulator [1, d_model]
    std::array<int, Tensor::MAX_DIMS> out_shape = {};
    out_shape[0] = 1;
    out_shape[1] = config.d_model;
    Tensor output(out_shape, 2);  // zero initialized

    // scores buffer [pos+1] — reused each head
    std::vector<float> scores(pos + 1);

    for (int h = 0; h < config.num_heads; h++) {
        // which KV head does this belong to?
        int kv_head = h / kv_per_head;

        float* q_head = Q.data() + h * head_dim;
        float* out_head = output.data() + h * head_dim;

        // compute scores dot(q_head, each cached k for kv_h)
        for (int t = 0; t <= pos; t++) {
            float dot = 0.0f;
            float* k_t = k_cache_ptr + t * kv_dim + kv_head * head_dim;
            scores[t] = scale * dot_product(q_head, k_t, head_dim);
        }

        // softmax over scores[0..pos]
        softmax(scores);

        // weighted sum of V: out_head += scores[t] * v_cache[t]
        for (int t = 0; t <= pos; t++) {
            float* v_t = v_cache_ptr + t * kv_dim + kv_head * head_dim;
            for (int d = 0; d < head_dim; d++) {
                out_head[d] += scores[t] * v_t[d];
            }
        }
    }


    // 5. output projection — wo is [out, in] (HF format), transB=true
    Tensor result = matmul(output, W.wo[layer_idx], LIB::BLAS, true);
    return result;
}

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
    float* row = weights_.token_embedding.data() + (token_id * config_.d_model);
    return Tensor(row, config_.d_model, shape, 2);
}

// run forward pass, return logits over vocabulary
Tensor Transformer::forward(int token_id, int pos) {
    Tensor X = embed(token_id);
    FFNLayer ffn;
    AttentionLayer attn;
    
    // Loops through the config_.num_layers (22) layers (from num_layers in TransformerConfig)
    for (int layer = 0; layer < config_.num_layers; layer++) {
        Tensor attn_input = rmsnorm(X, weights_.attn_norm[layer]);
        Tensor attn_out = attn.forward(attn_input, pos, weights_, kv_cache_, layer, config_);
        X = add(X, attn_out);

        Tensor ffn_input = rmsnorm(X, weights_.ffn_norm[layer]);
        Tensor ffn_out = ffn.forward(ffn_input, weights_, layer, config_);
        X = add(X, ffn_out);
    }

    X = rmsnorm(X, weights_.final_norm);
    
    // Tensor lm_head_T = weights_.lm_head.transpose();
    // logits
    // return matmul(X, lm_head_T, LIB::BLAS, true);
    // lm_head is [vocab_size, d_model]; pass directly with transB=true
    return matmul(X, weights_.lm_head, LIB::BLAS, true);
}

// greedy sample — just argmax
int Transformer::greedy_sample(Tensor& logits) {
    float* arr = logits.data();
    return std::max_element(arr, arr + logits.numel()) - arr;
}

// temperature + top-p (nucleus) sample
int Transformer::sample(Tensor& logits, float temperature, float top_p,
                        const std::vector<int>& past_tokens, float rep_penalty) {
    // Apply repetition penalty: divide logit of any already-seen token by rep_penalty
    if (rep_penalty != 1.0f) {
        float* raw = logits.data();
        for (int id : past_tokens)
            raw[id] /= rep_penalty;
    }

    logits.scale(1.0f / temperature);
    logits.softmax();
    float* probs = logits.data();
    int n = logits.numel();

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

        // allocate tensor and read data
        auto shape = make_shape(dims, ndim);
        Tensor tensor(shape, ndim);
        fread(tensor.data(), sizeof(float), numel, f);

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