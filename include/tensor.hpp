#include <stddef.h>
#include <array>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <exception>
#include <cstring>



class Tensor {
    private:
        static constexpr int BYTES_PER_ELEMENT = 32;
        static constexpr int BYTE_ALIGNMENT = 32;
        static constexpr int MAX_DIMS = 8;                       // How many dims we can possibly hold (transformers never need more than this)

        std::unique_ptr<float[]> owned_data_;
        float* data_;
        size_t nbytes_;
        int ndim_;
        std::array<int, MAX_DIMS> shape_;
        std::array<size_t, MAX_DIMS> strides_;

    public:
        // Constructors
        Tensor(std::array<int, MAX_DIMS> shape, int ndim);
        Tensor(float* data, size_t n, std::array<int, MAX_DIMS> shape, int ndim);
        Tensor(const Tensor& other);
        Tensor(Tensor&& other) noexcept;
        Tensor& operator=(const Tensor& other);
        Tensor& operator=(Tensor&& other) noexcept;
        ~Tensor() = default;

        // Metadata queries
        int shape(int dim) const;
        int ndim() const;
        size_t numel() const;
        size_t nbytes() const;
        int dtype_size();
        bool is_contiguous() const;

        // Data access
        float* data();
        const float* data() const;
        float& at(std::array<int, MAX_DIMS> indices);
        const float& at(std::array<int, MAX_DIMS> indices) const;

        // View operations
        Tensor reshape(std::array<int, MAX_DIMS> new_shape, int new_ndim) const;
        Tensor slice(int dim, int start, int end) const;
        Tensor transpose(int dim_a, int dim_b) const;

        // Utility
        void fill(float value);
        void print() const;
};