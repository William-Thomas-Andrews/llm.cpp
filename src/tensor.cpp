#include "tensor.hpp"


// ---
// Constructors

Tensor::Tensor(std::array<int, MAX_DIMS> shape, int ndim) : ndim_(ndim) {
    shape_ = shape;
    size_t nelements = 1;
    // Find number of bytes
    for (int i = 0; i < ndim_; i++) 
        nelements *= shape[i];
    // Multiply to get the number of bytes
    nbytes_ = nelements * BYTE_ALIGNMENT;
    // Allocate memory
    void* raw_ptr = std::aligned_alloc(BYTE_ALIGNMENT, nbytes_);
    if (raw_ptr == nullptr) 
        throw std::domain_error("Data array allocation failed.");
    // Set raw pointer in data_
    data_ = static_cast<float*>(raw_ptr);
    // Set all values to 0.0
    std::memset(data_, 0.0, nbytes_);
    // Set ownership to unique ptr
    owned_data_= std::unique_ptr<float[]>(data_);
    // Create strides
    strides_[ndim_ - 1] = BYTES_PER_ELEMENT;
    for (int i = ndim_ - 2; i >= 0; i--)
        strides_[i] = strides_[i + 1] * shape_[i + 1];
}

Tensor::Tensor(float *data, size_t n, std::array<int, MAX_DIMS> shape, int ndim) {
    shape_ = shape;
    size_t nelements = 1;
    // Find number of bytes
    for (int i = 0; i < ndim_; i++) 
        nelements *= shape[i];
    // Error check if array size is different
    if (nelements != n) 
        throw std::length_error("Error: array with size n does not have correct size.");
    // Multiply to get the number of bytes
    nbytes_ = nelements * BYTE_ALIGNMENT;
    // Set pointer in data_
    data_ = data;
    // Set ownership to unique ptr in owned_data
    owned_data_= std::unique_ptr<float[]>(data_);
    // Create strides
    strides_[ndim_ - 1] = BYTES_PER_ELEMENT;
    for (int i = ndim_ - 2; i >= 0; i--)
        strides_[i] = strides_[i + 1] * shape_[i + 1];
}

/// @brief
/// @param other
Tensor::Tensor(const Tensor &other) {

}

Tensor::Tensor(Tensor &&other) noexcept {

}

Tensor& Tensor::operator=(const Tensor& other) {

}

Tensor& Tensor::operator=(Tensor&& other) noexcept {

}

//
// ---



// ---
// Metadata queries

int Tensor::shape(int dim) const {
    return shape_[dim];
}

int Tensor::ndim() const {
    return ndim_;
}

/**
 * @brief Returns the total number of elements (product of shape).
 * 
*/
size_t Tensor::numel() const {
    size_t n = 1;
    for (int i = 0; i < ndim_; i++)
        n *= shape_[i];
    return n;
}

size_t Tensor::nbytes() const {
    return nbytes_;
}

int Tensor::dtype_size() {
    return 4; // float is 4 bytes
}

bool Tensor::is_contiguous() const {
    size_t expected = 4;  // sizeof the element (float is 4 bytes)
    for (int i = ndim_ - 1; i >= 0; i--) {
        if (strides_[i] != expected) return false;
        expected *= shape_[i];
    }
    return true;
}

//
// ---



// ---
// Data access

float* Tensor::data() {
    return data_;
}

const float* Tensor::data() const {
    return data_;
}

float& Tensor::at(std::array<int, MAX_DIMS> indices) {
    int index = 0;
    for (int i = 0; i < MAX_DIMS; i++)
        index += strides_[i] * indices[i];
    return data_[index];
}

const float& Tensor::at(std::array<int, MAX_DIMS> indices) const {
    int index = 0;
    for (int i = 0; i < MAX_DIMS; i++)
        index += strides_[i] * indices[i];
    return data_[index];
}

//
// ---



// ---
// View operations

Tensor Tensor::reshape(std::array<int, MAX_DIMS> new_shape, int new_dim) const {

}

Tensor Tensor::slice(int dim, int start, int end) const {

}

Tensor Tensor::transpose(int dim_a, int dim_b) const {

}

//
// ---



// ---
// Utility

void Tensor::fill(float value) {

}

void Tensor::print() const {

}

//
// ---