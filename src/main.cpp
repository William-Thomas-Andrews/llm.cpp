#include "transformer/transformer.hpp"
#include <iostream>
#include <string>

#define MODEL_PATH "models/tinyllama-chat"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: jarvis <model_path> [prompt] [max_tokens]\n";
        return 1;
    }
    std::string input = "The most popular language in the world is\nAnswer:";
    std::string model_path  = argv[1];
    std::string prompt      = argc > 2 ? argv[2] : input;
    int max_tokens          = argc > 3 ? std::stoi(argv[3]) : 100;

    // load model
    Transformer transformer(model_path); 

    auto append = [&](std::vector<int>& ids, const std::string& text) {
        auto pieces = transformer.tokenizer().encode(text);
        // Strip leading BOS that encode() auto-prepends
        if (!pieces.empty() && pieces[0] == transformer.tokenizer().bos_id())
            pieces.erase(pieces.begin());
        // Strip standalone leading ▁ (space token, decode = " ") — added by SentencePiece
        // at string boundaries. Keep ▁< (decode = " <") which is correct for <|system|>.
        if (!pieces.empty() && transformer.tokenizer().decode(pieces[0]) == " ")
            pieces.erase(pieces.begin());
        ids.insert(ids.end(), pieces.begin(), pieces.end());
    };

    std::vector<int> input_ids;
    input_ids.push_back(transformer.tokenizer().bos_id());
    // append(input_ids, "<|system|>\nYou are a helpful assistent.\n");
    // input_ids.push_back(transformer.tokenizer().eos_id());
    // append(input_ids, "\n<|user|>\n" + prompt);
    // input_ids.push_back(transformer.tokenizer().eos_id());
    // append(input_ids, "\n<|assistant|>\n");
    append(input_ids, prompt);

    std::cout << prompt;
    std::cout.flush();

    int pos = 0;
    int next_id = -1;
    std::vector<int> generated;

    // process prompt tokens first
    for (size_t i = 0; i < input_ids.size() - 1; i++)
        transformer.forward(input_ids[i], pos++);

    // generate
    next_id = input_ids.back(); // get last element
    for (int i = 0; i < max_tokens; i++) {
        Tensor logits = transformer.forward(next_id, pos++);
        next_id = transformer.sample(logits, 0.8f, 0.9f, generated);
if (next_id == transformer.tokenizer().eos_id()) break;
        if (next_id >= transformer.tokenizer().vocab_size()) break;  // chat template tokens (<|assistant|> etc.)
        generated.push_back(next_id);
        std::cout << transformer.tokenizer().decode(next_id);
        std::cout.flush();
    }

    std::cout << "\n";

    // printf("=== [Debug TransformerConfig] ===\n");
    // printf("d_model:      %d\n", transformer.config().d_model);
    // printf("num_heads:    %d\n", transformer.config().num_heads);
    // printf("num_kv_heads: %d\n", transformer.config().num_kv_heads);
    // printf("num_layers:   %d\n", transformer.config().num_layers);
    // printf("ffn_dim:      %d\n", transformer.config().ffn_dim);
    // printf("max_seq_len:  %d\n", transformer.config().max_seq_len);
    // printf("rope_theta:   %.1f\n", transformer.config().rope_theta);
    // printf("eps:          %e\n", transformer.config().eps);

    // printf("[Debug KVCache] Initialized: %d layers, k_cache[0] shape=(%d, %d)\n",
    // (int)transformer.kv_cache().k_cache.size(),
    //     transformer.kv_cache().k_cache[0].shape_at(0),
    //     transformer.kv_cache().k_cache[0].shape_at(1));

    return 0;
}