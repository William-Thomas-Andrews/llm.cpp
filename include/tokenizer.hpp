#pragma once
#include <string>
#include <vector>
#include "token_vocab_map.hpp"
#include "token_id_map.hpp"
#include <sentencepiece_processor.h>

// TODO: experiment with implementing custom concurrent hashmap into here

class Tokenizer {
public:
    Tokenizer(const std::string& model_path);

    // encode a string to token ids
    std::vector<int> encode(const std::string& input_text) const;

    // decode a single token id to string
    std::string decode(int token_id) const;

    // decode a sequence of token ids to string
    std::string decode(const std::vector<int>& token_ids) const;

    // batch encode multiple prompts in parallel (OpenMP + your hashmap)
    std::vector<std::vector<int>> batch_encode(const std::vector<std::string>& texts) const;

    int vocab_size() const;
    int bos_id() const;   // beginning of sequence
    int eos_id() const;   // end of sequence
    int pad_id() const;   // padding

private:
    static constexpr int MAP_INITIAL_SIZE = 65536;
    sentencepiece::SentencePieceProcessor processor_;
    VocabHashMap vocab_;       // token string -> id
    IdHashMap id_to_token_;    // id -> token string
    int vocab_size_;
    int bos_id_;
    int eos_id_;
    int pad_id_;

    void load(const std::string& model_path);
};