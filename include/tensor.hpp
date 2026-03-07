#include <stddef.h>
#include <array>
// #include <iostream>

// #define MAX_DIMS 3


class Tensor {
    float* data_;
    size_t nbytes_;
    bool owns_data_;        
    static constexpr int MAX_DIMS = 8;      // How many dims we can possibly hold (transformers never need more than this)
    int ndim_;                              // How many dims are actually active
    std::array<int, MAX_DIMS> shape_;
    std::array<int, MAX_DIMS> strides_;
};