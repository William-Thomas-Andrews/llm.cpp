#include "token_vocab_map.hpp"



// ------------------------------------------------------------------
// 'vocab_node' implementation

void print_vocab_node(const std::unique_ptr<vocab_node> input) {
    if (input == nullptr) std::cout << "||-||" << std::endl;
    else {
        std::cout << "||" << input->key << ":" << input->value << "|| ";
        vocab_node* ptr = input->next.get();
        while (ptr != nullptr) {
            if (ptr->next == nullptr)  std::cout << "||" << ptr->key << ":" << ptr->value << "||"; // if last iteration
            else std::cout << "||" << ptr->key << ":" << ptr->value << "||  ";
            ptr = ptr->next.get();
        }
        std::cout << std::endl;
    }
}

std::ostream& operator<<(std::ostream& out, const vocab_node& input) {
    out << "||" << input.key << ":" << input.value << "||";
    return out;
}

// ------------------------------------------------------------------





// ------------------------------------------------------------------
// 'vocab_bucket' implementation

// Normal Constructor
vocab_bucket::vocab_bucket(std::unique_ptr<vocab_node> other) {
    head = std::move(other);
}

void vocab_bucket::print_vocab_bucket() const {
    print_chain(head.get());
}

void vocab_bucket::free_vocab_bucket() {
    free_chain(std::move(head)); // automatically frees everything from ownership transfer and going out of scope
}

bool vocab_bucket::empty() const noexcept {
    return head == nullptr;
}

int vocab_bucket::vocab_bucket_insert(vocab_node& input) {
    if (head == nullptr) {
        head = std::make_unique<vocab_node>(input);
        return 1; // signifies a new entry was added
    }
    if (head->key == input.key) throw std::runtime_error("[insert] Error: duplicate key found: " + input.key);
    vocab_node* ptr = head.get();
    while (ptr->next != nullptr) {
        if (ptr->key == input.key) throw std::runtime_error("[insert] Error: duplicate key found: " + input.key);
        ptr = ptr->next.get();
    }
    if (ptr->key == input.key) throw std::runtime_error("[insert] Error: duplicate key found: " + input.key);
    ptr->next = std::make_unique<vocab_node>(input);
    return 0; // signifies no new entry was added
}

vocab_node vocab_bucket::pop() {
    vocab_node ret = *head.get();
    head = std::move(head->next);
    return ret;
}

// ------------------------------------------------------------------





// ------------------------------------------------------------------
// 'VocabHashMap' implementation

// 0. Default Constructor
VocabHashMap::VocabHashMap() : num_entries_(0), num_items_(0), capacity_(16) {
    array.reserve(capacity_);
    for (int i = 0; i < capacity_; i++)
        array.push_back(std::make_unique<vocab_bucket>());
}

// 1. Normal Constructor
VocabHashMap::VocabHashMap(int cap) : num_entries_(0), num_items_(0), capacity_(cap) {
    array.reserve(capacity_);
    for (int i = 0; i < capacity_; i++)
        array.push_back(std::make_unique<vocab_bucket>());
}

void VocabHashMap::free_table() {
    array.clear();
    num_entries_ = 0;
    num_items_ = 0;
}

unsigned int VocabHashMap::hash(const std::string& key) const {
    std::size_t h = hasher_(key);
    return h % capacity_;
}

vocab_node& VocabHashMap::find_vocab_node(const std::string& key) {
    std::shared_lock<std::shared_mutex> st_lock(table_lock);
    if (num_entries_ == 0) throw std::runtime_error("[find_vocab_node] Error: key: " + key + " not found. Errno 1");
    unsigned index = hash(key);
    std::shared_lock<std::shared_mutex> b_lock(array[index]->lock);
    vocab_node* ptr = array[index]->head.get();
    if (ptr == nullptr) throw std::runtime_error("[find_vocab_node] Error: key: " + key + " not found. Errno 2");
    else if (ptr->key == key) return *ptr;
    else if (ptr->next == nullptr) throw std::runtime_error("[find_vocab_node] Error: key: " + key + " not found. Errno 3");
    else if (ptr->next != nullptr) {
        ptr = ptr->next.get();
        while (ptr != nullptr) {
            if (ptr->key == key)
                return *ptr;
            ptr = ptr->next.get();
        }
    }
    throw std::runtime_error("[find_vocab_node] Error: key: " + key + " not found. Errno 4");
}

bool VocabHashMap::in_table(const std::string& key) {
    std::shared_lock<std::shared_mutex> st_lock(table_lock);
    if (num_entries_ == 0) return false;
    unsigned index = hash(key);
    std::shared_lock<std::shared_mutex> s_lock(array[index]->lock);
    vocab_node* ptr = array[index]->head.get();
    if (ptr == nullptr)  return false;
    else if (ptr->key == key) return true;
    else if (ptr->next == nullptr) return false; // if this item not the key but the only item in the list
    else if (ptr->next != nullptr) {
        ptr = ptr->next.get();
        while (ptr != nullptr) {
            if (ptr->key == key)
                return true;
            ptr = ptr->next.get();
        }
    }
    return false;
}

bool VocabHashMap::in_table(const std::string& key, int index) {
    std::shared_lock<std::shared_mutex> st_lock(table_lock);
    if (num_entries_ == 0) return false;
    std::shared_lock<std::shared_mutex> s_lock(array[index]->lock);
    vocab_node* ptr = array[index]->head.get();
    if (ptr == nullptr)  return false;
    else if (ptr->key == key) return true;
    else if (ptr->next == nullptr) return false; // if this item not the key but the only item in the list
    else if (ptr->next != nullptr) {
        ptr = ptr->next.get();
        while (ptr != nullptr) {
            if (ptr->key == key)
                return true;
            ptr = ptr->next.get();
        }
    }
    return false;
}

int VocabHashMap::find_val(const std::string& key) {
    return find_vocab_node(key).value;
}

void VocabHashMap::rehash_to(VocabHashMap& table) {
    vocab_node* ptr;
    for (int i = 0; i < capacity_; i++) {
        ptr = array[i]->head.get();
        while (ptr != nullptr) {
            table.insert(*ptr);
            ptr = ptr->next.get();
        }
    }
}

void VocabHashMap::expand() {
    std::vector<std::unique_lock<std::shared_mutex>> locks;
    for (int i = 0; i < capacity_; i++) // Locks down the entire table now that expansion has been called
        locks.push_back(std::unique_lock<std::shared_mutex>(array[i]->lock));
    for (int i = capacity_; i < capacity_*2; i++)
        array.push_back(std::make_unique<vocab_bucket>());
    capacity_ *= 2;
    VocabHashMap temp = VocabHashMap(capacity_);
    for (int i = 0; i < capacity_; i++)
        while (!array[i]->empty())
            temp.expansion_insert(array[i]->pop());
    for (int i = 0; i < capacity_; i++)
        while (!temp.array[i]->empty())
            expansion_insert(temp.array[i]->pop());
    // (lock goes out of scope) unlocks table 
}

void VocabHashMap::insert(const std::string& key, const int val) {
    vocab_node n(key, val);
    insert(n);
}

void VocabHashMap::insert(vocab_node& input) {
    { std::unique_lock<std::shared_mutex> ut_lock(table_lock); // locks table for possible expansion
    if (num_entries_ == capacity_)
        expand(); // increases hash table size atomically
    } // unlocks expansion lock regardless of if condition was met
    std::shared_lock<std::shared_mutex> s_lock(table_lock);
    unsigned index = hash(input.key);
    std::unique_lock<std::shared_mutex> b_lock(array[index]->lock);
    if (array[index]->vocab_bucket_insert(input) == 1) num_entries_++;
    num_items_++;
    // lock automatically unlocks when going out of scope
}

void VocabHashMap::insert(vocab_node&& input) {
    { std::unique_lock<std::shared_mutex> ut_lock(table_lock); // locks table for possible expansion
    if (num_entries_ == capacity_)
        expand(); // increases hash table size atomically
    } // unlocks expansion lock regardless of if condition was met
    std::shared_lock<std::shared_mutex> s_lock(table_lock);
    unsigned index = hash(input.key);
    std::unique_lock<std::shared_mutex> b_lock(array[index]->lock);
    if (array[index]->vocab_bucket_insert(input) == 1) num_entries_++;
    num_items_++;
    // lock automatically unlocks when going out of scope
}

// No lock involved
void VocabHashMap::expansion_insert(vocab_node& input) {
    unsigned index = hash(input.key);
    if (array[index]->vocab_bucket_insert(input) == 1) num_entries_++;
    num_items_++;
}

// No lock involved
void VocabHashMap::expansion_insert(vocab_node&& input) {
    unsigned index = hash(input.key);
    if (array[index]->vocab_bucket_insert(input) == 1) num_entries_++;
    num_items_++;
}

void VocabHashMap::remove(const std::string& key) {
    std::shared_lock<std::shared_mutex> st_lock(table_lock);
    int index = hash(key);
    std::unique_lock<std::shared_mutex> b_lock(array[index]->lock);
    vocab_node* ptr = array[index]->head.get();
    vocab_node* prev;
    if (ptr == nullptr) 
        throw std::runtime_error("[remove] Error: key: " + key + " not found. [DEBUG] Errno 1");
    else if (ptr->key == key) {
        if (ptr->next == nullptr) num_entries_--;
        num_items_--;
        array[index]->head = std::move(ptr->next);
        return;
    }
    else if (ptr->next == nullptr)  // if this item not the key but the only item in the list
        throw std::runtime_error("[remove] Error: key: " + key + " not found. [DEBUG] Errno 2");
    else if (ptr->next != nullptr) {
        prev = ptr;
        ptr = ptr->next.get();
        while (ptr != nullptr) {
            if (ptr->key == key) {
                prev->next = std::move(ptr->next);
                num_items_--;
                return;
            }
            prev = ptr;
            ptr = ptr->next.get();
        }
    }
    throw std::runtime_error("[remove] Error: key: " + key + " not found. [DEBUG] Errno 3");
}

void VocabHashMap::print_table() {
    std::unique_lock<std::shared_mutex> ut_lock(table_lock);
    std::cout << "VocabHashMap (capacity_ = " << capacity_
              << ", num_entries_ = " << num_entries_ << ", num_items_ = " << num_items_ << ")\n";
    std::cout << "--------------------------------------------------\n";

    for (int i = 0; i < capacity_; ++i) {
        std::cout << "[ ";
        if (i < 10) std::cout << " ";   // alignment for single-digit indices
        std::cout << i << " ] : ";
        array[i]->print_vocab_bucket();
        std::cout << '\n';
    }
    ut_lock.unlock();
    std::cout << "--------------------------------------------------\n";
}

int* VocabHashMap::num_entries() {
    return &num_entries_;
}

int* VocabHashMap::num_items() {
    return &num_items_;
}

int* VocabHashMap::capacity() {
    return &capacity_;
}

std::vector<std::unique_ptr<vocab_bucket>>& VocabHashMap::get_array() {
    return array;
}



// --------------------------------------------------
// Utilities

void print_chain(vocab_node* head) {
    if (head == nullptr) {
        std::cout << "(empty)";
        return;
    }
    vocab_node* ptr = head;
    while (ptr != nullptr) {
        std::cout << ptr->key << "(" << ptr->value << ")";
        if (ptr->next != nullptr) {
            std::cout << " -> ";
        }
        ptr = ptr->next.get();
    }
}

// Call std::move(head) as the argument like so: 
// free_chain(std::move(head));
void free_chain(std::unique_ptr<vocab_node> base) {
    // automatically frees
}