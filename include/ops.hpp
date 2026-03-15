#pragma once

#include "tensor.hpp"
#include <cmath>

// ---
// Backend selector
// Extensible: add CUDA, OpenCL, and other backends later

enum class Backend {
    NAIVE,
    CPU,
    CUDA,   // future
};

enum class LIB {
    NAIVE,
    BLAS,
};


// ---
// Matrix multiplication
// C = A @ B
// A: [M, K], B: [K, N], C: [M, N]
Tensor matmul(Tensor& A, Tensor& B, LIB mult, bool transB = false);

// ---
// Naive Matrix multiplication
// C = A @ B
// A: [M, K], B: [K, N], C: [M, N]
Tensor matmul_naive(Tensor& A, LIB& B, int M, int K, int N);

// ---
// Accelerated Matrix multiplication (OpenBlas)
// C = A @ B
// A: [M, K], B: [K, N], C: [M, N]
Tensor matmul_blas(Tensor& A, Tensor& B, int M, int K, int N, bool transB = false);

// ---
// Normalization

// RMS Norm
Tensor rmsnorm(Tensor& X, Tensor& weight, float eps = 1e-6f);

// ---
// Attention
Tensor softmax(Tensor& X, int dim);

// Apply rotary positional embeddings to a single vector
void rope_vector(float* vec, int head_dim, int position);

// Apply rotary positional embeddings to Q and K
void rope(Tensor& Q, Tensor& K, int head_dim, int position);

// ---
// Activations
Tensor silu(const Tensor& X);

// SwiGLU: silu(gate) * x — used in LLaMA FFN
Tensor swiglu(const Tensor& gate, const Tensor& X);

// ---
// Elementwise
Tensor add(Tensor& A, Tensor& B);
Tensor mul(Tensor& A, Tensor& B);