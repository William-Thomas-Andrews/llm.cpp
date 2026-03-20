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
    std::vector<float> cd(C.numel());

    std::vector<int32_t> acc(C.numel(), 0);

    // Naive Matrix Multiply Operation (ikj)
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            // float r = A[i][k]
            int8_t r = aq[i*K + k];
            for (int j = 0; j < N; j++) {
                // C[i][j] += r * B[k][j]
                acc[i*N + j] += r * bq[k*N + j];
            }
        }
    }
    float scaling_val = A.q_scale() * B.q_scale();
    for (int i = 0; i < cd.size(); i++)
        cd[i] = acc[i] * scaling_val;

    // compute scale for C from actual output range
    float r_max = -INFINITY; 
    float r_min = +INFINITY;
    for (int i = 0; i < cd.size(); i++) {
        r_max = std::max(r_max, cd[i]); // finding the max value for (unquantized) float number line
        r_min = std::min(r_min, cd[i]); // finding the min value for (unquantized) float number line
    }
    float scale = std::max(std::abs(r_max), std::abs(r_min)) / q_max; // (r_max / q_max)
    if (scale == 0.0f) scale = 1e-8f;
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

// Optimized Matrix Multiply Operation
// A: [M, K], B: [K, N], C: [M, N]
Tensor matmul_microkernel(Tensor& A, Tensor& B, int M, int K, int N) {
    std::array<int, Tensor::MAX_DIMS> shape = {};
    shape[0] = M;
    shape[1] = N;
    Tensor C(shape, 2);  // owning, zero initialized

    // quantized values
    int8_t* aq = A.data(); int8_t* bq = B.data();

    // dequantized values
    std::vector<float> cd(C.numel());

    std::vector<int32_t> acc(C.numel(), 0);

    int block_size = (M > 64 && N > 64 && K > 64) ? 32 : 16;
    int BM = (M / block_size) * block_size; // largest multiple of block_size <= M
    int BN = (N / block_size) * block_size; // largest multiple of block_size <= N
    int BK = (K / block_size) * block_size; // largest multiple of block_size <= K

    // manipulate blocks
    for (int bi = 0; bi < BM; bi += block_size) {
        for (int bk = 0; bk < BK; bk += block_size) {
            for (int bj = 0; bj < BN; bj += block_size) {
                // within the blocks
                    for (int i = bi; i < bi + block_size; i += 4) {
                        for (int j = bj; j < bj + block_size; j += 16) {

                            // 4 accumulators (each handles 16 columns)
                            __m256i acc0_lo = _mm256_loadu_si256((__m256i*)&acc[(i+0)*N + j]);
                            __m256i acc0_hi = _mm256_loadu_si256((__m256i*)&acc[(i+0)*N + j + 8]);

                            __m256i acc1_lo = _mm256_loadu_si256((__m256i*)&acc[(i+1)*N + j]);
                            __m256i acc1_hi = _mm256_loadu_si256((__m256i*)&acc[(i+1)*N + j + 8]);

                            __m256i acc2_lo = _mm256_loadu_si256((__m256i*)&acc[(i+2)*N + j]);
                            __m256i acc2_hi = _mm256_loadu_si256((__m256i*)&acc[(i+2)*N + j + 8]);

                            __m256i acc3_lo = _mm256_loadu_si256((__m256i*)&acc[(i+3)*N + j]);
                            __m256i acc3_hi = _mm256_loadu_si256((__m256i*)&acc[(i+3)*N + j + 8]);

                            // Process 2 k values per iteration using madd_epi16:
                            // interleave two B rows so madd computes a[k]*b[k][j] + a[k+1]*b[k+1][j]
                            // in one instruction instead of two mullo_epi32 chains.
                            // block_size is always 16 or 32, so k always steps evenly.
                            for (int k = bk; k < bk + block_size; k += 2) {

                                // Load two consecutive B rows and interleave pairs:
                                // b_lo16 = [b[k][j0], b[k+1][j0], b[k][j1], b[k+1][j1], ... x8]
                                // b_hi16 = [b[k][j8], b[k+1][j8], ..., b[k][j15], b[k+1][j15]]
                                __m128i b8_k0 = _mm_loadu_si128((__m128i*)&bq[(k+0)*N + j]);
                                __m128i b8_k1 = _mm_loadu_si128((__m128i*)&bq[(k+1)*N + j]);
                                __m256i b_lo16 = _mm256_cvtepi8_epi16(_mm_unpacklo_epi8(b8_k0, b8_k1));
                                __m256i b_hi16 = _mm256_cvtepi8_epi16(_mm_unpackhi_epi8(b8_k0, b8_k1));

                                // Pack A pair as int32: low 16 bits = a[k], high 16 bits = a[k+1].
                                // set1_epi32 broadcasts this across the register as int16 pairs
                                // [a[k], a[k+1], a[k], a[k+1], ...] for madd to consume.
                                #define MADD_ROW(row, alo, ahi)                                          \
                                {                                                                         \
                                    __m256i av = _mm256_set1_epi32(                                      \
                                        (int32_t)(uint16_t)(int16_t)aq[(i+(row))*K + k  ] |             \
                                       ((int32_t)           (int16_t)aq[(i+(row))*K + k+1] << 16));     \
                                    (alo) = _mm256_add_epi32((alo), _mm256_madd_epi16(av, b_lo16));      \
                                    (ahi) = _mm256_add_epi32((ahi), _mm256_madd_epi16(av, b_hi16));      \
                                }

                                MADD_ROW(0, acc0_lo, acc0_hi)
                                MADD_ROW(1, acc1_lo, acc1_hi)
                                MADD_ROW(2, acc2_lo, acc2_hi)
                                MADD_ROW(3, acc3_lo, acc3_hi)
                                #undef MADD_ROW
                            }

                            // store back
                            _mm256_storeu_si256((__m256i*)&acc[(i+0)*N + j], acc0_lo);
                            _mm256_storeu_si256((__m256i*)&acc[(i+0)*N + j + 8], acc0_hi);

                            _mm256_storeu_si256((__m256i*)&acc[(i+1)*N + j], acc1_lo);
                            _mm256_storeu_si256((__m256i*)&acc[(i+1)*N + j + 8], acc1_hi);

                            _mm256_storeu_si256((__m256i*)&acc[(i+2)*N + j], acc2_lo);
                            _mm256_storeu_si256((__m256i*)&acc[(i+2)*N + j + 8], acc2_hi);

                            _mm256_storeu_si256((__m256i*)&acc[(i+3)*N + j], acc3_lo);
                            _mm256_storeu_si256((__m256i*)&acc[(i+3)*N + j + 8], acc3_hi);
                        }
                    }
            }
        }
    }

    // leftover rows
    for (int i = BM; i < M; i++) {
        for (int k = 0; k < K; k++) {
            for (int j = 0; j < N; j++) {
                acc[i*N + j] += aq[i*K + k] * bq[k*N + j];
            }
        }
    }

    // leftover K: rows [0..BM), k [BK..K), all j
    for (int i = 0; i < BM; i++) {
        for (int k = BK; k < K; k++) {
            for (int j = 0; j < N; j++) {
                acc[i*N + j] += aq[i*K + k] * bq[k*N + j];
            }
        }
    }

    // leftover N: rows [0..BM), k [0..BK), cols [BN..N)
    for (int i = 0; i < BM; i++) {
        for (int k = 0; k < BK; k++) {
            for (int j = BN; j < N; j++) {
                acc[i*N + j] += aq[i*K + k] * bq[k*N + j];
            }
        }
    }

    float scaling_val = A.q_scale() * B.q_scale();
    for (int i = 0; i < cd.size(); i++)
        cd[i] = acc[i] * scaling_val;

    // compute scale for C from actual output range
    float r_max = -INFINITY; 
    float r_min = +INFINITY;
    for (int i = 0; i < cd.size(); i++) {
        r_max = std::max(r_max, cd[i]); // finding the max value for (unquantized) float number line
        r_min = std::min(r_min, cd[i]); // finding the min value for (unquantized) float number line
    }
    float scale = std::max(std::abs(r_max), std::abs(r_min)) / q_max; // (r_max / q_max)
    if (scale == 0.0f) scale = 1e-8f;
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
Tensor matmul_blocked(Tensor& A, Tensor& B, int M, int K, int N) {
    std::array<int, Tensor::MAX_DIMS> shape = {};
    shape[0] = M;
    shape[1] = N;
    Tensor C(shape, 2);  // owning, zero initialized

    // quantized values
    int8_t* aq = A.data(); int8_t* bq = B.data();

    // dequantized values
    std::vector<float> cd(C.numel());

    std::vector<int32_t> acc(C.numel(), 0);

    int block_size = (M > 64 && N > 64 && K > 64) ? 32 : 16;
    int BM = (M / block_size) * block_size; // largest multiple of block_size <= M
    int BN = (N / block_size) * block_size; // largest multiple of block_size <= N
    int BK = (K / block_size) * block_size; // largest multiple of block_size <= K

    // manipulate blocks
    for (int bi = 0; bi < BM; bi += block_size) {
        for (int bk = 0; bk < BK; bk += block_size) {
            for (int bj = 0; bj < BN; bj += block_size) {
                // within the blocks
                for (int i = bi; i < bi + block_size; i++) {
                    for (int k = bk; k < bk + block_size; k++) {
                        int8_t a_val = aq[i*K + k];
                        for (int j = bj; j < bj + block_size; j++) {
                            acc[i*N + j] += a_val * bq[k*N + j];
                        }
                    }
                }
            }
        }
    }

    // leftover rows
    for (int i = BM; i < M; i++) {
        for (int k = 0; k < K; k++) {
            for (int j = 0; j < N; j++) {
                acc[i*N + j] += aq[i*K + k] * bq[k*N + j];
            }
        }
    }

    // leftover K: rows [0..BM), k [BK..K), all j
    for (int i = 0; i < BM; i++) {
        for (int k = BK; k < K; k++) {
            for (int j = 0; j < N; j++) {
                acc[i*N + j] += aq[i*K + k] * bq[k*N + j];
            }
        }
    }

    // leftover N: rows [0..BM), k [0..BK), cols [BN..N)
    for (int i = 0; i < BM; i++) {
        for (int k = 0; k < BK; k++) {
            for (int j = BN; j < N; j++) {
                acc[i*N + j] += aq[i*K + k] * bq[k*N + j];
            }
        }
    }

    // scale array
    float scale_val = A.q_scale() * B.q_scale();
    for (int i = 0; i < cd.size(); i++) 
        cd[i] = acc[i] * scale_val;

    // compute scale for C from actual output range
    float r_max = -INFINITY; 
    float r_min = +INFINITY;
    for (int i = 0; i < cd.size(); i++) {
        r_max = std::max(r_max, cd[i]); // finding the max value for (unquantized) float number line
        r_min = std::min(r_min, cd[i]); // finding the min value for (unquantized) float number line
    }
    float scale = std::max(std::abs(r_max), std::abs(r_min)) / q_max; // (r_max / q_max)
    if (scale == 0.0f) scale = 1e-8f;
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
    float r_max = -INFINITY; 
    float r_min = +INFINITY;
    for (int i = 0; i < cd.size(); i++) {
        r_max = std::max(r_max, cd[i]); // finding the max value for (unquantized) float number line
        r_min = std::min(r_min, cd[i]); // finding the min value for (unquantized) float number line
    }
    float scale = std::max(std::abs(r_max), std::abs(r_min)) / q_max; // (r_max / q_max)
    if (scale == 0.0f) scale = 1e-8f;
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
        case LIB::NAIVE:       return matmul_naive(A, B, M, K, N);
        case LIB::BLOCKED:     return matmul_blocked(A, B, M, K, N);
        case LIB::MICROKERNEL: return matmul_microkernel(A, B, M, K, N);
        case LIB::BLAS:        return matmul_blas(A, B, M, K, N, transB);
        default:               throw std::runtime_error("Unsupported multiplication.");
    }
}



// ---
// Normalization

Tensor rmsnorm(Tensor& X, Tensor& weight, float eps) {
    int n = X.shape_at(X.ndim() - 1);  // last dim (number of elements per vector)
    int num_vectors = X.numel() / n;

    Tensor Y = X;
    int8_t* y = Y.data();
    int8_t* w = weight.data();

    std::vector<float> yd(Y.numel()), wd(weight.numel());
    for (int i = 0; i < Y.numel(); i++)
        yd[i] = Y.dequantize(y[i]);
    for (int i = 0; i < weight.numel(); i++)
        wd[i] = weight.dequantize(w[i]);

    for (int i = 0; i < num_vectors; i++) {
        // compute mean of squares
        float ms = 0.0f;
        for (int j = 0; j < n; j++) 
            ms = ms + yd[i*n + j]*yd[i*n + j];
        ms = ms / n;

        // normalize and scale
        ms = 1.0f /std::sqrt(ms + eps);
        for (int j = 0; j < n; j++)
            yd[i*n + j] = yd[i*n + j] * ms * wd[j];
    }

    // Compute output scale from actual output range
    float r_max = -INFINITY, r_min = +INFINITY;
    for (int i = 0; i < (int)yd.size(); i++) {
        r_max = std::max(r_max, yd[i]);
        r_min = std::min(r_min, yd[i]);
    }
    float scale = std::max(std::abs(r_max), std::abs(r_min)) / q_max;
    if (scale == 0.0f) scale = 1e-8f;
    Y.set_scale(scale);

    // Re-Quantize
    for (int i = 0; i < Y.numel(); i++)
        y[i] = Y.quantize(yd[i]);

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
    std::vector<float> yd(Y.numel());
    for (int i = 0; i < Y.numel(); i++)
        yd[i] = Y.dequantize(y[i]);

    for (int i = 0; i < num_vectors; i++) {
        float max_val = *std::max_element((yd.begin() + i*vector_len), (yd.begin() + i*vector_len) + vector_len);
        float summation = 0.0f;
        for (int j = 0; j < vector_len; j++) {
            yd[i*vector_len + j] = std::exp(yd[i*vector_len + j] - max_val);
            summation += yd[i*vector_len + j];
        }
        summation = 1.0f / summation;
        for (int k = 0; k < vector_len; k++)
            yd[i*vector_len + k] = yd[i*vector_len + k] * summation;
    }

    // Compute output scale from actual output range
    float r_max = -INFINITY, r_min = +INFINITY;
    for (int i = 0; i < (int)yd.size(); i++) {
        r_max = std::max(r_max, yd[i]);
        r_min = std::min(r_min, yd[i]);
    }
    float scale = std::max(std::abs(r_max), std::abs(r_min)) / q_max;
    if (scale == 0.0f) scale = 1e-8f;
    Y.set_scale(scale);

    // Re-Quantize
    for (int i = 0; i < Y.numel(); i++)
        y[i] = Y.quantize(yd[i]);

    return Y;
}

//
// ---

void rope_vector(std::vector<float>& vec, int h, int head_dim, const std::vector<float>& cos_vals, const std::vector<float>& sin_vals) {
    int start_index = h * head_dim;
    for (int i = 0; i < head_dim / 2; i++) {

        float x = vec[start_index + 2*i];        // first of the pair
        float y = vec[start_index + 2*i + 1];    // second of the pair

        vec[start_index + 2*i]     = x * cos_vals[i] - y * sin_vals[i];
        vec[start_index + 2*i + 1] = x * sin_vals[i] + y * cos_vals[i];
    }
}

// Apply rotary positional embeddings to Q and K
void rope(Tensor& Q, Tensor& K, int position) {
    int num_heads  = Q.shape_at(0);
    int head_dim   = Q.shape_at(1);

    int8_t* q = Q.data();
    int8_t* k = K.data();

    std::vector<float> qd(Q.numel()), kd(K.numel());
    for (int i = 0; i < Q.numel(); i++)
        qd[i] = Q.dequantize(q[i]);
    for (int i = 0; i < K.numel(); i++)
        kd[i] = K.dequantize(k[i]);

    
    std::vector<float> cos_vals(head_dim / 2);
    std::vector<float> sin_vals(head_dim / 2);
    for (int i = 0; i < head_dim / 2; i++) {
        float freq = 1.0f / std::pow(10000.0f, (2.0f * i) / head_dim);
        float theta = position * freq;
        cos_vals[i] = std::cos(theta);
        sin_vals[i] = std::sin(theta);
    }

    for (int h = 0; h < num_heads; h++) {
        rope_vector(qd, h, head_dim, cos_vals, sin_vals);
        rope_vector(kd, h, head_dim, cos_vals, sin_vals);
    }

    // Compute output scale from actual output range
    float r_max = -INFINITY, r_min = +INFINITY;
    for (int i = 0; i < (int)qd.size(); i++) {
        r_max = std::max(r_max, qd[i]);
        r_min = std::min(r_min, qd[i]);
    }
    float scale = std::max(std::abs(r_max), std::abs(r_min)) / q_max;
    if (scale == 0.0f) scale = 1e-8f;
    Q.set_scale(scale);

    // Re-Quantize
    for (int i = 0; i < Q.numel(); i++)
        q[i] = Q.quantize(qd[i]);

    // Compute output scale from actual output range
    r_max = -INFINITY; r_min = +INFINITY;
    for (int i = 0; i < (int)kd.size(); i++) {
        r_max = std::max(r_max, kd[i]);
        r_min = std::min(r_min, kd[i]);
    }
    scale = std::max(std::abs(r_max), std::abs(r_min)) / q_max;
    if (scale == 0.0f) scale = 1e-8f;
    K.set_scale(scale);

    // Re-Quantize
    for (int i = 0; i < K.numel(); i++)
        k[i] = K.quantize(kd[i]);
}

//
// ---



// ---
// Activations

// Sigmoid Linear Unit
Tensor silu(Tensor& X) {
    Tensor Y = X;
    int8_t* y = Y.data();

    std::vector<float> yd(Y.numel());

    for (size_t i = 0; i < Y.numel(); i++) {
        float xd = X.dequantize(y[i]);
        yd[i] = xd * (1.0f / (1.0f + std::exp(-xd)));
    }

    // Compute output scale from actual output range
    float r_max = -INFINITY, r_min = +INFINITY;
    for (int i = 0; i < (int)yd.size(); i++) {
        r_max = std::max(r_max, yd[i]);
        r_min = std::min(r_min, yd[i]);
    }
    float scale = std::max(std::abs(r_max), std::abs(r_min)) / q_max;
    if (scale == 0.0f) scale = 1e-8f;
    Y.set_scale(scale);

    // Re-Quantize
    for (int i = 0; i < Y.numel(); i++)
        y[i] = Y.quantize(yd[i]);

    return Y;
}

// SwiGLU: silu(gate) * x — used in LLaMA FFN
Tensor swiglu(Tensor& gate, Tensor& X) {
    if (gate.numel() != X.numel()) throw std::runtime_error("swiglu: size mismatch");
    Tensor Y = silu(gate);
    int8_t* y = Y.data();
    int8_t* x = X.data();

    std::vector<float> yd(Y.numel());

    for (size_t i = 0; i < Y.numel(); i++) 
        yd[i] = Y.dequantize(y[i]) * X.dequantize(x[i]);

    // Compute output scale from actual output range
    float r_max = -INFINITY, r_min = +INFINITY;
    for (int i = 0; i < (int)yd.size(); i++) {
        r_max = std::max(r_max, yd[i]);
        r_min = std::min(r_min, yd[i]);
    }
    float scale = std::max(std::abs(r_max), std::abs(r_min)) / q_max;
    if (scale == 0.0f) scale = 1e-8f;
    Y.set_scale(scale);

    // Re-Quantize
    for (int i = 0; i < Y.numel(); i++)
        y[i] = Y.quantize(yd[i]);
        
    return Y;
}

//
// ---



// ---
// Elementwise arithmetic

// Elementwise add
Tensor add(Tensor& A, Tensor& B) {
    if (A.numel() != B.numel()) throw std::runtime_error("[elementwise add] Error: Tensor size mismatch.");

    Tensor C(A.shape_array(), A.ndim());

    int8_t* a = A.data();
    int8_t* b = B.data();
    int8_t* c = C.data();

    std::vector<int32_t> acc(C.numel());

    // Step 1: int32 accumulation
    for (size_t i = 0; i < C.numel(); i++)
        acc[i] = (int32_t)a[i] * (int32_t)b[i];

    // Step 2: find max absolute value
    int32_t max_abs = 0;
    for (auto v : acc) {
        int32_t abs_v = (v == INT32_MIN) ? INT32_MAX : std::abs(v);
        max_abs = std::max(max_abs, abs_v);
    }

    float ab_scale = A.q_scale() * B.q_scale();

    float scale = max_abs * ab_scale / q_max;
    if (scale == 0.0f) scale = 1e-8f;

    C.set_scale(scale);

    // Step 3: requantize
    float combined_scale = ab_scale / scale;

    // Step 4: add operation
    for (size_t i = 0; i < C.numel(); i++) {
        float val = acc[i] + combined_scale;
        c[i] = C.quantize(val);
    }

    // cblas_scopy(A.numel(), a, 1, c, 1); // c = a
    // cblas_saxpy(A.numel(), 1.0f, b, 1, c, 1); // c = 1.0*b + c

    return C;
}

// Elementwise multiply
Tensor mul(Tensor& A, Tensor& B) {
    if (A.numel() != B.numel())
        throw std::runtime_error("[elementwise multiplication] Error: Tensor size mismatch.");

    Tensor C(A.shape_array(), A.ndim());

    int8_t* a = A.data();
    int8_t* b = B.data();
    int8_t* c = C.data();

    std::vector<int32_t> acc(C.numel());

    // Step 1: int32 accumulation
    for (size_t i = 0; i < C.numel(); i++)
        acc[i] = (int32_t)a[i] * (int32_t)b[i];

    // Step 2: find max absolute value
    int32_t max_abs = 0;
    for (auto v : acc) {
        int32_t abs_v = (v == INT32_MIN) ? INT32_MAX : std::abs(v);
        max_abs = std::max(max_abs, abs_v);
    }

    float ab_scale = A.q_scale() * B.q_scale();

    float scale = max_abs * ab_scale / q_max;
    if (scale == 0.0f) scale = 1e-8f;

    C.set_scale(scale);

    // Step 3: requantize
    float combined_scale = ab_scale / scale;

    // Step 4: multiplication operation
    for (size_t i = 0; i < C.numel(); i++) {
        float val = acc[i] * combined_scale;
        c[i] = C.quantize(val);
    }

    return C;
}

//
// ---


void scale(float* array, int n, float scalar) {
    for (int i = 0; i < n; i++)
        array[i] *= scalar;
}