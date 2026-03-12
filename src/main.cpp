#include "transformer.hpp"

int main() {
    Transformer transformer = Transformer("models/tinyllama"); // load model
    printf("=== TransformerConfig ===\n");
    printf("d_model:      %d\n", transformer.config().d_model);
    printf("num_heads:    %d\n", transformer.config().num_heads);
    printf("num_kv_heads: %d\n", transformer.config().num_kv_heads);
    printf("num_layers:   %d\n", transformer.config().num_layers);
    printf("ffn_dim:      %d\n", transformer.config().ffn_dim);
    printf("vocab_size:   %d\n", transformer.config().vocab_size);
    printf("max_seq_len:  %d\n", transformer.config().max_seq_len);
    printf("rope_theta:   %.1f\n", transformer.config().rope_theta);
    printf("eps:          %e\n", transformer.config().eps);

    printf("KVCache initialized: %d layers, k_cache[0] shape=(%d, %d)\n",
    (int)transformer.kv_cache().k_cache.size(),
        transformer.kv_cache().k_cache[0].shape_at(0),
        transformer.kv_cache().k_cache[0].shape_at(1));
}