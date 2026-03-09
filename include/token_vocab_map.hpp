#include <iostream>
#include <cstdint>
#include <vector>
#include <shared_mutex>
#include <algorithm>
#include <memory>
#include <mutex>


#pragma once


struct vocab_node {
    std::string key;
    int value;
    std::unique_ptr<vocab_node> next;
    vocab_node() = default; // 0. Default Constructor
    vocab_node(std::string k, int v) : key(k), value(v), next(nullptr) {} // 1. Normal Constructor
    vocab_node(const vocab_node& other) : key(other.key), value(other.value), next(nullptr) {} // 2. Copy Constructor
};

struct vocab_bucket {
    std::unique_ptr<vocab_node> head;
    std::shared_mutex lock;
    vocab_bucket()                              = default;   // 0. Default Constructor
    vocab_bucket(std::unique_ptr<vocab_node> other);               // 1. Normal Constructor
    ~vocab_bucket()                              = default;  // 2. Destructor
    vocab_bucket(const vocab_bucket& other)            = delete;   // 3. Copy Constructor
    vocab_bucket& operator=(const vocab_bucket& other) = delete;   // 4. Copy Assignment Operator
    vocab_bucket(vocab_bucket&& other)                 = delete;   // 5. Move Constructor
    vocab_bucket& operator=(vocab_bucket&& other)      = delete;   // 6. Move Assignment Operator
    void print_vocab_bucket() const;
    void free_vocab_bucket();
    bool empty() const noexcept;
    int vocab_bucket_insert(vocab_node& input);
    vocab_node pop();
};


std::ostream& operator<<(std::ostream& out, const vocab_node& input);
void print_vocab_node(const vocab_node* input);

class VocabHashMap {
    private:
        std::vector<std::unique_ptr<vocab_bucket>> array;
        int num_entries_;
        int num_items_;
        int capacity_;
        std::shared_mutex table_lock;
        std::hash<std::string> hasher_;
        
    public:
        VocabHashMap();                                                 // 0. Default Constructor
        VocabHashMap(int capacity_);                                     // 1. Normal Constructor
        ~VocabHashMap()                                     = default;  // 2. Destructor
        VocabHashMap(const VocabHashMap& other)                = delete;   // 3. Copy Constructor
        VocabHashMap& operator=(const VocabHashMap& other)     = delete;   // 4. Copy Assignment Operator
        VocabHashMap(VocabHashMap&& other) noexcept            = delete;  // 5. Move Constructor
        VocabHashMap& operator=(VocabHashMap&& other) noexcept = delete;  // 6. Move Assignment Operator
        void free_table();
        unsigned int hash(const std::string& key) const;
        vocab_node& find_vocab_node(const std::string& key);
        bool in_table(const std::string& key);
        bool in_table(const std::string& key, int index);
        int find_val(const std::string& key);
        void rehash_to(VocabHashMap& table);
        void expand();
        void insert(const std::string& key, const int val);
        void insert(vocab_node& input);
        void insert(vocab_node&& input);
        void expansion_insert(vocab_node& input);
        void expansion_insert(vocab_node&& input);
        void remove(const std::string& key);
        void print_table();
        int* num_entries();
        int* num_items();
        int* capacity();
        std::vector<std::unique_ptr<vocab_bucket>>& get_array();
};

void print_chain(vocab_node* head);
void free_chain(std::unique_ptr<vocab_node> base);