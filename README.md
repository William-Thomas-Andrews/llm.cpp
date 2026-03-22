# llm.cpp

A from-scratch C++ LLM inference engine. Loads transformer weights, runs tokenization via SentencePiece, and generates text using greedy/top-p sampling.

The default model is **TinyLlama-1.1B-Chat**, a compact chat-tuned LLaMA model that fits easily in CPU RAM.

## Dependencies

- g++ with C++23 support
- CMake >= 3.22
- OpenMP
- BLAS (e.g. OpenBLAS)
- [SentencePiece](https://github.com/google/sentencepiece) (`libsentencepiece` installed to `~/.local`)



## Model Setup

### 1. Create the model directory

```bash
mkdir -p models/tinyllama-chat
```

### 2. Download the model files from Hugging Face

```bash
wget -P models/tinyllama-chat https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/config.json
wget -P models/tinyllama-chat https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/tokenizer.model
wget -P models/tinyllama-chat https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/model.safetensors
```

### 3. Convert weights to the binary format

The engine uses a custom flat binary format (`model.bin`) instead of safetensors directly. Run the conversion script:

```bash
pip install -r tools/requirements.txt
python3 tools/convert_weights.py models/tinyllama-chat models/tinyllama-chat/model.bin
```

This converts the bfloat16 safetensors weights to float32 and writes them to `model.bin`.

## 
```bash
mkdir build   # make a build directory for the CMakeLists to identify
```

## Build

```bash
./run_build.sh   # runs 'cmake .. && make' inside the 'build/' directory
```

## Run

```bash
./execute.sh     # runs the executable
```

Arguments:
1. `model_path` — path to the model directory (required)
2. `prompt` — input text (optional, defaults to a built-in prompt)
3. `max_tokens` — maximum tokens to generate (optional, defaults to 100)
