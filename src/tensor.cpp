#include "tensor.hpp"


// ---
// Constructors

Tensor::Tensor() {
    owned_data_ = nullptr;
    data_ = nullptr;
}

Tensor::Tensor(std::array<int, MAX_DIMS> shape, int ndim) : ndim_(ndim) {
    shape_ = shape;
    size_t nelements = 1;
    // Find number of elements
    for (int i = 0; i < ndim_; i++) 
        nelements *= shape[i];
    nelements_ = nelements;
    // Multiply to get the number of bytes
    nbytes_ = nelements * BYTES_PER_ELEMENT;
    // Allocate memory
    size_t aligned_nbytes = (nbytes_ + BYTE_ALIGNMENT - 1) & ~(BYTE_ALIGNMENT - 1);
    void* raw_ptr = std::aligned_alloc(BYTE_ALIGNMENT, aligned_nbytes);
    if (raw_ptr == nullptr) throw std::domain_error("Data array allocation failed.");
    // Set raw pointer in data_
    data_ = static_cast<int8_t*>(raw_ptr);
    // Set all values to 0.0
    std::memset(data_, 0, nbytes_);
    // Set ownership to unique ptr
    owned_data_= std::unique_ptr<int8_t[]>(data_);
    // Create strides
    strides_[ndim_ - 1] = BYTES_PER_ELEMENT;
    for (int i = ndim_ - 2; i >= 0; i--)
        strides_[i] = strides_[i + 1] * shape_[i + 1];
}

Tensor::Tensor(int8_t *data, size_t n, std::array<int, MAX_DIMS> shape, int ndim) : ndim_(ndim) {
    shape_ = shape;
    size_t nelements = 1;
    // Find number of elements
    for (int i = 0; i < ndim_; i++) 
        nelements *= shape_[i];
    nelements_ = nelements;
    // Error check if array size is different
    if (nelements != n) throw std::length_error("Error: array with size n does not have correct size.");
    // Multiply to get the number of bytes
    nbytes_ = nelements * BYTES_PER_ELEMENT;
    // Set pointer in data_
    data_ = data;
    // Set ownership to unique ptr in owned_data
    // owned_data_= std::unique_ptr<int8_t[]>(data_);
    // Create strides
    strides_[ndim_ - 1] = BYTES_PER_ELEMENT;
    for (int i = ndim_ - 2; i >= 0; i--)
        strides_[i] = strides_[i + 1] * shape_[i + 1];
}

/// @brief Copy constructor
/// @param other another tensor
Tensor::Tensor(const Tensor &other) {
    if (this == &other) return;

    shape_ = other.shape_;
    nelements_ = other.nelements_;
    nbytes_ = other.nbytes_;
    ndim_ = other.ndim_;
    // Allocate dst memory
    size_t aligned_nbytes = (nbytes_ + BYTE_ALIGNMENT - 1) & ~(BYTE_ALIGNMENT - 1);
    void* raw_ptr = std::aligned_alloc(BYTE_ALIGNMENT, aligned_nbytes);
    if (raw_ptr == nullptr) throw std::domain_error("Data array allocation failed.");
    // Set raw pointer in data_
    data_ = static_cast<int8_t*>(raw_ptr);
    // Copy data from src to dst
    memcpy(data_, other.data_, nbytes_);
    // Set ownership to unique ptr in owned_data
    owned_data_= std::unique_ptr<int8_t[]>(data_);
    // Create strides
    strides_[ndim_ - 1] = BYTES_PER_ELEMENT;
    for (int i = ndim_ - 2; i >= 0; i--)
        strides_[i] = strides_[i + 1] * shape_[i + 1];
}

/// @brief Move constructor
/// @param other another tensor
Tensor::Tensor(Tensor &&other) noexcept {
    if (this == &other) return;

    ndim_ = other.ndim_;
    nelements_ = other.nelements_;
    nbytes_ = other.nbytes_;
    shape_ = other.shape_;
    strides_ = other.strides_;
    data_ = other.data_;
    owned_data_ = std::move(other.owned_data_);

    other.ndim_ = 0;
    other.nbytes_ = 0;
    other.shape_ = {};
    other.strides_ = {};
    other.data_ = nullptr;
}

/// @brief Copy assignment operator
/// @param other another tensor
/// @return this tensor
Tensor& Tensor::operator=(const Tensor& other) {
    if (this == &other) return *this;

    shape_ = other.shape_;
    nelements_ = other.nelements_;
    nbytes_ = other.nbytes_;
    ndim_ = other.ndim_;
    // Allocate dst memory
    size_t aligned_nbytes = (nbytes_ + BYTE_ALIGNMENT - 1) & ~(BYTE_ALIGNMENT - 1);
    void* raw_ptr = std::aligned_alloc(BYTE_ALIGNMENT, aligned_nbytes);
    if (raw_ptr == nullptr) throw std::domain_error("Data array allocation failed.");
    // Set raw pointer in data_
    data_ = static_cast<int8_t*>(raw_ptr);
    // Copy data from src to dst
    memcpy(data_, other.data_, nbytes_);
    // Set ownership to unique ptr in owned_data
    owned_data_= std::unique_ptr<int8_t[]>(data_);
    // Create strides
    strides_[ndim_ - 1] = BYTES_PER_ELEMENT;
    for (int i = ndim_ - 2; i >= 0; i--)
        strides_[i] = strides_[i + 1] * shape_[i + 1];
    return *this;
}

/// @brief Move assignment operator
/// @param other another tensor
/// @return this tensor
Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this == &other) return *this;

    ndim_ = other.ndim_;
    nelements_ = other.nelements_;
    nbytes_ = other.nbytes_;
    shape_ = other.shape_;
    strides_ = other.strides_;
    data_ = other.data_;
    owned_data_ = std::move(other.owned_data_);

    other.ndim_ = 0;
    other.nbytes_ = 0;
    other.shape_ = {};
    other.strides_ = {};
    other.data_ = nullptr;
    
    return *this;
}

//
// ---



// ---
// Metadata queries

int Tensor::shape_at(int dim) const {
    return shape_[dim];
}

std::string Tensor::shape() const {
    if (ndim_ == 0) return "()";
    std::string str = "(";
    int i = 0;
    for (; i < ndim_-1; i++)
        str += std::to_string(shape_[i]) + ", ";
    return str + std::to_string(shape_[i]) + ")";
}

std::array<int, Tensor::MAX_DIMS> Tensor::shape_array() const {
    return shape_;
}

int Tensor::ndim() const {
    return ndim_;
}

/**
 * @brief Returns the total number of elements (product of shape).
 * 
*/
size_t Tensor::numel() const {
    return nelements_;
}

size_t Tensor::nbytes() const {
    return nbytes_;
}

int Tensor::dtype_size() {
    return BYTES_PER_ELEMENT; // int8_t is 2 bytes
}

bool Tensor::is_contiguous() const {
    size_t expected = BYTES_PER_ELEMENT;  // sizeof the element (float is 2 bytes)
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

int8_t* Tensor::data() {
    return data_;
}

// const int8_t* Tensor::data() const {
//     return data_;
// }

int8_t& Tensor::at(std::array<int, MAX_DIMS> indices) {
    char* ptr = reinterpret_cast<char*>(data_);
    for (int i = 0; i < ndim_; i++)
        ptr += strides_[i] * indices[i];
    return *reinterpret_cast<int8_t*>(ptr);
}

const int8_t& Tensor::at(std::array<int, MAX_DIMS> indices) const {
    char* ptr = reinterpret_cast<char*>(data_);
    for (int i = 0; i < ndim_; i++)
        ptr += strides_[i] * indices[i];
    return *reinterpret_cast<int8_t*>(ptr);
}

//
// ---



// ---
// View operations

Tensor Tensor::reshape(std::array<int, MAX_DIMS> new_shape, int new_ndim) const {
    if (!is_contiguous()) throw std::runtime_error("Error: data not contiguous before reshape.");

    // make sure the total elements match
    size_t new_numel = 1;
    for (int i = 0; i < new_ndim; i++) new_numel *= new_shape[i];
    if (new_numel != numel()) throw std::runtime_error("reshape must preserve total number of elements");

    Tensor view(data_, nelements_, new_shape, new_ndim);  // non-owning view
    return view;
}

Tensor Tensor::slice(int dim, int start, int end) const {
    if (dim < 0 || dim >= ndim_) throw std::out_of_range("Error: dimension out of range.");
    if (!(start >= 0 && end <= shape_[dim] && start < end)) throw std::out_of_range("Error: start and end indices not correct.");
    
    Tensor view(data_, nelements_, shape_, ndim_);

    // advance the data pointer to the start index
    char* ptr = reinterpret_cast<char*>(view.data_);
    ptr += start * strides_[dim];
    view.data_ = reinterpret_cast<int8_t*>(ptr);

    // shrink the shape along the sliced dimension
    view.shape_[dim] = end - start;
    view.nelements_ = (nelements_ / shape_[dim]) * (end - start);

    return view;
}

Tensor Tensor::transpose(int dim_a, int dim_b) const {
    if (dim_a < 0 || dim_a >= ndim_ || dim_b < 0 || dim_b >= ndim_) throw std::out_of_range("Error: dimension out of range.");
    if (dim_a == dim_b) return *this;
    Tensor view(data_, nelements_, shape_, ndim_);  // non-owning view
    std::swap(view.strides_[dim_a], view.strides_[dim_b]);
    std::swap(view.shape_[dim_a], view.shape_[dim_b]);
    return view;
}

Tensor Tensor::transpose() const {
    int dim_a = 0; int dim_b = 1;
    Tensor view(data_, nelements_, shape_, ndim_);  // non-owning view
    std::swap(view.strides_[dim_a], view.strides_[dim_b]);
    std::swap(view.shape_[dim_a], view.shape_[dim_b]);
    return view;
}

void Tensor::scale(int8_t scalar) {
    for (size_t i = 0; i < nelements_; i++) 
        data_[i] *= scalar;
}

int8_t Tensor::quantize(float r) {
    if (q_scale_ == 0.0f) return 0;
    int32_t qi = static_cast<int32_t>(std::round(r / q_scale_)); // aligns with int32 accumulation pipeline
    qi = std::clamp(qi, -127, 127); // restrict codomain
    return static_cast<int8_t>(qi); // cast back to int8
}

float Tensor::dequantize(int8_t q) {
    return q_scale_ * q;
}

void Tensor::softmax() {
    if (!is_contiguous()) throw std::runtime_error("softmax requires contiguous tensor");
    int8_t max_val = *std::max_element(data_, data_ + nelements_);
    float summation = 0.0f;
    std::vector<float> buffer(nelements_);
    std::cout << "HI!" << std::endl;
    for (int i = 0; i < nelements_; i++) {
        buffer[i] = std::exp(static_cast<float>(data_[i]) - static_cast<float>(max_val));
        summation += buffer[i];
    }
    summation = 1.0f / summation;
    for (int j = 0; j < nelements_; j++)
        data_[j] = static_cast<int8_t>(buffer[j] * summation * 127.0f);
}

//
// ---



// ---
// Utility

void Tensor::fill(int8_t value) {
    std::fill(data_, data_ + nelements_, value);
}

void Tensor::print() const {
    std::cout << 
    "Tensor[F32, shape=" << shape() << ", contiguous=" << is_contiguous() << "]" 
    << std::endl;
}

//
// ---