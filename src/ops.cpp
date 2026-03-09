#include "ops.hpp"


// ---
// Matrix Multiplication

// A: [M, K], B: [K, N], C: [M, N]
Tensor matmul_naive(Tensor& A, Tensor& B, int M, int K, int N) {
    std::array<int, Tensor::MAX_DIMS> shape = {};
    shape[0] = M;
    shape[1] = N;
    Tensor C(shape, 2);  // owning, zero initialized

    float* a = A.data();
    float* b = B.data();
    float* c = C.data();

    // Naive Matrix Multiply Operation (ikj)
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            // float r = A[i][k]
            float r = a[i*K + k];
            for (int j = 0; j < N; j++) {
                // C[i][j] += r * B[k][j]
                c[i*N + j] += r * b[k*N + j];
            }
        }
    }

    return C;
}

Tensor matmul_blas(Tensor& A, Tensor& B, int M, int K, int N) {
    std::array<int, Tensor::MAX_DIMS> shape = {};
    shape[0] = M;
    shape[1] = N;
    Tensor C(shape, 2);  // owning, zero initialized

    // OpenBlas Matrix Multiply Operation
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0, A.data(), K, B.data(), N, 0.0, C.data(), N);

    return C;
}

// A: [M, K], B: [K, N], C: [M, N]
Tensor matmul(Tensor& A, Tensor& B, Multiplication mult) {
    // A must be [M, K], B must be [K, N]
    if (A.ndim() != 2 || B.ndim() != 2) throw std::runtime_error("Error: dimensions incorrect.");
    if (A.shape_at(1) != B.shape_at(0)) throw std::runtime_error("Error: columns of A do not match rows of B.");
    if (!A.is_contiguous() || !B.is_contiguous()) throw std::runtime_error("Error: data not contiguous.");

    int M = A.shape_at(0);
    int K = A.shape_at(1);
    int N = B.shape_at(1);

    switch(mult) {
        case Multiplication::NAIVE:     return matmul_naive(A, B, M, K, N);
        case Multiplication::BLAS:      return matmul_blas(A, B, M, K, N);
        default:                        throw std::runtime_error("Unsupported multiplication.");
    }
}

//
// ---



// ---
// Normalization

Tensor rmsnorm(const Tensor& X, Tensor& weight, float eps = 1e-6f) {
    int n = X.shape_at(X.ndim() - 1);  // last dim (number of elements per vector)
    int num_vectors = X.numel() / n;

    Tensor Y = X;
    float* y = Y.data();
    float* w = weight.data();

    for (int i = 0; i < num_vectors; i++) {
        float* vec = y + i*n;

        // compute mean of squares
        float ms = 0.0f;
        for (int j = 0; j < n; j++) 
            ms = ms + vec[j]*vec[j];
        ms = ms / n;

        // normalize and scale
        ms = 1.0f /std::sqrt(ms + eps);
        for (int j = 0; j < n; j++)
            vec[j] = vec[j] * ms * w[j];
    }

    return Y;
}

//
// ---



// ---
// Attention
Tensor softmax(const Tensor& X, int dim) {
    if (!X.is_contiguous()) throw std::runtime_error("softmax requires contiguous tensor");
    
    int num_vectors = X.numel() / X.shape_at(dim);
    int vector_len = X.shape_at(dim);

    Tensor Y = X;
    float* y = Y.data();

    for (int i = 0; i < num_vectors; i++) {
        float* ptr = y + i * vector_len;
        float max_val = *std::max_element(ptr, ptr + vector_len);
        float summation = 0.0f;
        for (int j = 0; j < vector_len; j++) {
            ptr[j]= std::exp(ptr[j] - max_val);
            summation += ptr[j];
        }
        for (int k = 0; k < vector_len; k++)
            ptr[k] = ptr[k] / summation;
    }
    return Y;
}

//
// ---



// Apply rotary positional embeddings to Q and K
void rope(Tensor& q, Tensor& k, int head_dim, int pos);

//
// ---



// ---
// Activations

// Sigmoid Linear Unit
Tensor silu(const Tensor& x) {
    Tensor Y = x;
    float* y = Y.data();

    for (int i = 0; i < Y.numel(); i++)
        y[i] = y[i] * (1.0f / (1.0f + std::exp(-y[i])));

    return Y;
}

// SwiGLU: silu(gate) * x — used in LLaMA FFN
Tensor swiglu(Tensor& gate, Tensor& X) {
    if (gate.numel() != X.numel()) throw std::runtime_error("swiglu: size mismatch");
    Tensor Y = silu(gate);
    float* y = Y.data();
    float* x = X.data();

    for (int i = 0; i < Y.numel(); i++) 
        y[i] = y[i] * x[i];

    return Y;
}

//
// ---



// ---
// Elementwise arithmetic

// Elementwise add
Tensor add(Tensor& A, Tensor& B) {
    if (A.numel() != B.numel()) throw std::runtime_error("Error: Tensor number of elements each do not match for elementwise addition.");
    size_t n = A.numel();
    // Allocate memory
    size_t aligned_nbytes = (A.nbytes() + A.BYTE_ALIGNMENT - 1) & ~(A.BYTE_ALIGNMENT - 1);
    void* raw_ptr = std::aligned_alloc(A.BYTE_ALIGNMENT, aligned_nbytes);
    if (raw_ptr == nullptr) throw std::domain_error("Data array allocation failed.");
    // Set raw pointer in data_
    float* c = static_cast<float*>(raw_ptr);
    float* a = A.data();
    float* b = B.data();
    // 1. Copy a to c: c = a
    cblas_scopy(n, b, 1, c, 1);
    // 2. Add b to c: c = 1.0 * b + c (which is now a)
    cblas_saxpy(n, 1, b, 1, c, 1);
    Tensor view(c, n, A.shape_array(), A.ndim());
    return view;
}

// Elementwise multiply
Tensor mul(Tensor& A, Tensor& B) {
    if (A.numel() != B.numel()) throw std::runtime_error("Error: Tensor number of elements each do not match for elementwise addition.");
    size_t n = A.numel();
    // Allocate memory
    size_t aligned_nbytes = (A.nbytes() + A.BYTE_ALIGNMENT - 1) & ~(A.BYTE_ALIGNMENT - 1);
    void* raw_ptr = std::aligned_alloc(A.BYTE_ALIGNMENT, aligned_nbytes);
    if (raw_ptr == nullptr) throw std::domain_error("Data array allocation failed.");
    // Set raw pointer in data_
    float* c = static_cast<float*>(raw_ptr);
    float* a = A.data();
    float* b = B.data();
    // Standard loop, compiler auto optimized
    for (int i = 0; i < n; i++)
        c[i] = a[i] * b[i];
    Tensor view(c, n, A.shape_array(), A.ndim());
    return view;
}

//
// ---