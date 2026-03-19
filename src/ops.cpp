#include "ops.hpp"

// ---
// Matrix Multiplication

// A: [M, K], B: [K, N], C: [M, N]
Tensor matmul_naive(Tensor& A, Tensor& B, int M, int K, int N) {
    std::array<int, Tensor::MAX_DIMS> shape = {};
    shape[0] = M;
    shape[1] = N;
    Tensor C(shape, 2);  // owning, zero initialized

    // quantized values
    int8_t* aq = A.data(); int8_t* bq = B.data();

    // dequantized values
    std::vector<float> ad(A.numel()), bd(B.numel()), cd(C.numel());

    // De-Quantize arrays for matrix multiplication
    for (int i = 0; i < A.numel(); i++)
        ad[i] = A.dequantize(aq[i]);
    for (int i = 0; i < B.numel(); i++)
        bd[i] = B.dequantize(bq[i]);

    // Naive Matrix Multiply Operation (ikj)
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            // float r = A[i][k]
            float r = ad[i*K + k];
            for (int j = 0; j < N; j++) {
                // C[i][j] += r * B[k][j]
                cd[i*N + j] += r * bd[k*N + j];
            }
        }
    }

    // compute scale for C from actual output range
    float r_max = *std::max_element(cd.begin(), cd.end()); // max value for (unquantized) float number line
    float r_min = *std::min_element(cd.begin(), cd.end()); // min value for (unquantized) float number line
    float scale = std::max(std::abs(r_max), std::abs(r_min)) / q_max;
    C.set_scale(scale); // set that scale

    // We are defining the mapping:
    // for quantization:       q = clip(round(r / scale), -127.0f, 127.0f)
    // for dequantization:     r ~= q * scale

    // Quantize to return back C
    int8_t* c = C.data();
    for (int i = 0; i < C.numel(); i++)
        c[i] = C.quantize(cd[i]);

    return C;
}

Tensor matmul_blas(Tensor& A, Tensor& B, int M, int K, int N, bool transB) {
    std::array<int, Tensor::MAX_DIMS> shape = {};
    shape[0] = M;
    shape[1] = N;
    Tensor C(shape, 2);  // owning, zero initialized

    // quantized values
    int8_t* aq = A.data(); int8_t* bq = B.data();

    // dequantized values
    std::vector<float> ad(A.numel()), bd(B.numel()), cd(C.numel());

    // De-Quantize arrays for matrix multiplication
    for (int i = 0; i < A.numel(); i++)
        ad[i] = A.dequantize(aq[i]);
    for (int i = 0; i < B.numel(); i++)
        bd[i] = B.dequantize(bq[i]);

    // OpenBlas Matrix Multiply Operation
    if (transB)
        // B is [N, K] stored row-major; ldb = K (columns of B as stored)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0, ad.data(), K, bd.data(), K, 0.0, cd.data(), N);
    else
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0, ad.data(), K, bd.data(), N, 0.0, cd.data(), N);

    // compute scale for C from actual output range
    float r_max = *std::max_element(cd.begin(), cd.end()); // max value for (unquantized) float number line
    float r_min = *std::min_element(cd.begin(), cd.end()); // min value for (unquantized) float number line
    float scale = std::max(std::abs(r_max), std::abs(r_min)) / q_max; // (r_max / q_max)
    C.set_scale(scale); // set that scale

    // We are defining the mapping:
    // for quantization:       q = clip(round(r / scale), -127.0f, 127.0f)
    // for dequantization:     r ~= q * scale

    // Quantize to return back C
    int8_t* c = C.data();
    for (int i = 0; i < C.numel(); i++)
        c[i] = C.quantize(cd[i]);

    return C;
}

// A: [M, K], B: [K, N], C: [M, N]
Tensor matmul(Tensor& A, Tensor& B, LIB mult, bool transB) {
    // A must be [M, K], B must be [K, N]
    if (A.ndim() != 2 || B.ndim() != 2) {
        std::cout << A.ndim() << " and " << B.ndim() << std::endl;
        throw std::runtime_error("matmul() Error: dimensions incorrect.");
    }
    // When transB, B is [N, K]; inner dim K is at B.shape_at(1). Otherwise B is [K, N].
    if (!transB && A.shape_at(1) != B.shape_at(0)) throw std::runtime_error("matmul() Error: columns of A do not match rows of B.");
    if ( transB && A.shape_at(1) != B.shape_at(1)) throw std::runtime_error("matmul() Error: columns of A do not match columns of B.");
    if (!A.is_contiguous() || !B.is_contiguous()) throw std::runtime_error("matmul() Error: data not contiguous.");

    int M = A.shape_at(0);
    int K = A.shape_at(1);
    int N = transB ? B.shape_at(0) : B.shape_at(1);

    switch(mult) {
        case LIB::NAIVE:     return matmul_naive(A, B, M, K, N);
        case LIB::BLAS:      return matmul_blas(A, B, M, K, N, transB);
        default:                        throw std::runtime_error("Unsupported multiplication.");
    }
}



// ---
// Normalization

Tensor rmsnorm(Tensor& X, Tensor& weight, int8_t eps) {
    int n = X.shape_at(X.ndim() - 1);  // last dim (number of elements per vector)
    int num_vectors = X.numel() / n;

    Tensor Y = X;
    int8_t* y = Y.data();
    int8_t* w = weight.data();

    for (int i = 0; i < num_vectors; i++) {
        int8_t* vec = y + i*n;

        // compute mean of squares
        int8_t ms = 0.0f;
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
    int8_t* y = Y.data();

    for (int i = 0; i < num_vectors; i++) {
        int8_t* ptr = y + i * vector_len;
        int8_t max_val = *std::max_element(ptr, ptr + vector_len);
        int8_t summation = 0.0f;
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

void rope_vector(int8_t* vec, int head_dim, int position) {
    for (int i = 0; i < head_dim / 2; i++) {
        int8_t freq = 1.0f / std::pow(10000.0f, (2.0f * i) / head_dim);
        int8_t theta = position * freq;
        int8_t cos_val = std::cos(theta);
        int8_t sin_val = std::sin(theta);

        int8_t x = vec[2*i];        // first of the pair
        int8_t y = vec[2*i + 1];    // second of the pair

        vec[2*i]     = x * cos_val - y * sin_val;
        vec[2*i + 1] = x * sin_val + y * cos_val;
    }
}

// Apply rotary positional embeddings to Q and K
void rope(Tensor& Q, Tensor& K, int position) {
    int num_heads  = Q.shape_at(0);
    int head_dim   = Q.shape_at(1);

    int8_t* q = Q.data();
    int8_t* k = K.data();

    for (int h = 0; h < num_heads; h++) {
        int8_t* q_head = q + h * head_dim;
        int8_t* k_head = k + h * head_dim;
        rope_vector(q_head, head_dim, position);
        rope_vector(k_head, head_dim, position);
    }
}

//
// ---



// ---
// Activations

// Sigmoid Linear Unit
Tensor silu(const Tensor& x) {
    Tensor Y = x;
    int8_t* y = Y.data();

    for (size_t i = 0; i < Y.numel(); i++)
        y[i] = y[i] * (1.0f / (1.0f + std::exp(-y[i])));

    return Y;
}

// SwiGLU: silu(gate) * x — used in LLaMA FFN
Tensor swiglu(Tensor& gate, Tensor& X) {
    if (gate.numel() != X.numel()) throw std::runtime_error("swiglu: size mismatch");
    Tensor Y = silu(gate);
    int8_t* y = Y.data();
    int8_t* x = X.data();

    for (size_t i = 0; i < Y.numel(); i++) 
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
    Tensor C(A.shape_array(), A.ndim()); // owning, zero initialized
    int8_t* a = A.data();
    int8_t* b = B.data();
    int8_t* c = C.data();
    // cblas_scopy(A.numel(), a, 1, c, 1); // c = a
    // cblas_saxpy(A.numel(), 1.0f, b, 1, c, 1); // c = 1.0*b + c
    return C;
}

// Elementwise multiply
Tensor mul(Tensor& A, Tensor& B) {
    if (A.numel() != B.numel()) throw std::runtime_error("Error: Tensor number of elements each do not match for elementwise addition.");
    Tensor C(A.shape_array(), A.ndim()); // owning, zero initialized
    int8_t* a = A.data();
    int8_t* b = B.data();
    int8_t* c = C.data();
    for (size_t i = 0; i < C.numel(); i++) 
        c[i] = a[i] * b[i];
    return C;
}

//
// ---


void scale(float* array, int n, float scalar) {
    for (int i = 0; i < n; i++)
        array[i] *= scalar;
}

// void softmax(float* array, int n) {

// }