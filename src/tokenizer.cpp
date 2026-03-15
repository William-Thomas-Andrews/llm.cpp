#include "tokenizer.hpp"

// TODO: experiment with implementing custom concurrent hashmap into here

Tokenizer::Tokenizer(const std::string& model_path) {
    // : vocab_(MAP_INITIAL_SIZE), id_to_token_(MAP_INITIAL_SIZE) {
    load(model_path);
    bos_id_ = processor_.bos_id();
    eos_id_ = processor_.eos_id();
    pad_id_ = processor_.pad_id();
    vocab_size_ = processor_.GetPieceSize();
    // for (int i = 0; i < vocab_size_; i++) { // Input data to custom map 
    //     std::string piece = processor_.IdToPiece(i);
    //     vocab_.insert(piece, i);
    //     id_to_token_.insert(i, piece);
    // }
}

// encode a string to token ids
std::vector<int> Tokenizer::encode(const std::string& input_text) const {
    std::vector<int> ids;
    processor_.Encode(input_text, &ids);
    return ids;
}

// decode a single token id to string
std::string Tokenizer::decode(int token_id) const {
    std::string piece = processor_.IdToPiece(token_id);

    // Handle <0xNN> byte tokens (e.g. <0x0A> = newline, <0x09> = tab)
    if (piece.size() == 6 && piece[0] == '<' && piece[1] == '0' && piece[2] == 'x' && piece[5] == '>') {
        unsigned char byte_val = (unsigned char)std::stoi(piece.substr(3, 2), nullptr, 16);
        return std::string(1, (char)byte_val);
    }

    // Replace all ▁ (U+2581, UTF-8: e2 96 81) with spaces
    std::string result;
    for (size_t i = 0; i < piece.size(); ) {
        if (i + 2 < piece.size() && (unsigned char)piece[i]   == 0xe2
                                 && (unsigned char)piece[i+1] == 0x96
                                 && (unsigned char)piece[i+2] == 0x81) {
            result += ' ';
            i += 3;
        } else {
            result += piece[i++];
        }
    }
    return result;
}

// decode a sequence of token ids to string
std::string Tokenizer::decode(const std::vector<int>& token_ids) const {
    std::string text;
    processor_.Decode(token_ids, &text);
    return text;
}

// batch encode multiple prompts in parallel
std::vector<std::vector<int>> Tokenizer::batch_encode(const std::vector<std::string>& texts) const {
    std::vector<std::vector<int>> results(texts.size());
    #pragma omp parallel for
    for (int i = 0; i < texts.size(); i++)
        processor_.Encode(texts[i], &results[i]);
    return results;
}

int Tokenizer::vocab_size() const {
    return vocab_size_;
}
int Tokenizer::bos_id() const {   // beginning of sequence
    return bos_id_;
}
int Tokenizer::eos_id() const {   // end of sequence
    return eos_id_;
}
int Tokenizer::pad_id() const {   // padding
    return pad_id_;
}


// PRIVATE: Load model path
void Tokenizer::load(const std::string& model_path) {
    const auto status = processor_.Load(model_path);
    if (!status.ok()) std::cerr << status.ToString() << std::endl; // error
}
