#pragma once

#include <stddef.h>
#include <array>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <exception>
#include <cstring>
#include <cblas.h>
#include <cmath>
#include <vector>

class Tensor {

    public:
        static constexpr int BYTES_PER_ELEMENT = 1;     // Bytes per int8_t
        static constexpr int BYTE_ALIGNMENT = 8;       // Bits per int8_t
        static constexpr int MAX_DIMS = 8;              // How many dims we can possibly hold (transformers never need more than this)
        
        // Constructors
        Tensor();
        Tensor(std::array<int, MAX_DIMS> shape, int ndim); // Default constructor
        Tensor(int8_t* data, size_t n, std::array<int, MAX_DIMS> shape, int ndim); // Default constructor (view constructor)
        Tensor(const Tensor& other); // Copy constructor
        Tensor(Tensor&& other) noexcept; // Move constructor
        Tensor& operator=(const Tensor& other); // Copy assignment operator
        Tensor& operator=(Tensor&& other) noexcept; // Move operator
        ~Tensor() = default;

        // Metadata queries
        int shape_at(int dim) const;
        std::string shape() const;
        std::array<int, MAX_DIMS> shape_array() const;
        int ndim() const;
        size_t numel() const;
        size_t nbytes() const;
        int dtype_size();
        bool is_contiguous() const;

        // Data access
        int8_t* data();
        // const int8_t* data() const;
        int8_t& at(std::array<int, MAX_DIMS> indices);
        const int8_t& at(std::array<int, MAX_DIMS> indices) const;

        // View operations
        Tensor reshape(std::array<int, MAX_DIMS> new_shape, int new_ndim) const;
        Tensor slice(int dim, int start, int end) const;
        Tensor transpose(int dim_a, int dim_b) const;
        Tensor transpose() const;
        void scale(int8_t scalar);
        void softmax();

        // Utility
        void fill(int8_t value);
        void print() const;

        // ---
        // Quantization
        float scale() const { return scale_; }
        void set_scale(float s) { scale_ = s; }
        int8_t quantize(float x);
        float dequantize(int8_t x);

    private:
        std::unique_ptr<int8_t[]> owned_data_;
        int8_t* data_;
        size_t nelements_;
        size_t nbytes_;
        int ndim_;
        std::array<int, MAX_DIMS> shape_;
        std::array<size_t, MAX_DIMS> strides_;
        float scale_ = 1.0f;
};