#include <iostream>
#include <cstdint>
#include <vector>
#include <shared_mutex>
#include <algorithm>


#pragma once

struct state {
    int i, j, k;
    state() = default;
    state(int i, int j, int k) : i(i), j(j), k(k) {}
};

std::string get_string(state& state);
std::string get_string(const state& state);
std::ostream& operator<<(std::ostream& out, const state& input);
void print_state(state& s);
void print_state(const state& s);

struct node {
    state key;
    int value;
    std::unique_ptr<node> next;
    node() = default; // 0. Default Constructor
    node(state k, int v) : key(k), value(v), next(nullptr) {} // 1. Normal Constructor
    node(const node& other) : key(other.key), value(other.value), next(nullptr) {} // 2. Copy Constructor
};

struct Bucket {
    std::unique_ptr<node> head;
    std::shared_mutex lock;
    Bucket()                              = default;   // 0. Default Constructor
    Bucket(std::unique_ptr<node> other);               // 1. Normal Constructor
    ~Bucket()                              = default;  // 2. Destructor
    Bucket(const Bucket& other)            = delete;   // 3. Copy Constructor
    Bucket& operator=(const Bucket& other) = delete;   // 4. Copy Assignment Operator
    Bucket(Bucket&& other)                 = delete;   // 5. Move Constructor
    Bucket& operator=(Bucket&& other)      = delete;   // 6. Move Assignment Operator
    void print_bucket() const;
    void free_bucket();
    bool empty() const noexcept;
    int bucket_insert(node& input);
    node pop();
};

std::string get_string(node& input);
std::string get_string(const node& input);
bool operator==(const state& op1, const state& op2);
bool operator!=(const state& op1, const state& op2);
std::ostream& operator<<(std::ostream& out, const node& input);
void print_node(const node* input);

class HashTable {
    private:
        std::vector<std::unique_ptr<Bucket>> array;
        int num_entries;
        int num_items;
        int capacity;
        std::shared_mutex table_lock;
        
    public:
        HashTable();                                                 // 0. Default Constructor
        HashTable(int capacity);                                     // 1. Normal Constructor
        ~HashTable()                                     = default;  // 2. Destructor
        HashTable(const HashTable& other)                = delete;   // 3. Copy Constructor
        HashTable& operator=(const HashTable& other)     = delete;   // 4. Copy Assignment Operator
        HashTable(HashTable&& other) noexcept            = delete;  // 5. Move Constructor
        HashTable& operator=(HashTable&& other) noexcept = delete;  // 6. Move Assignment Operator
        void free_table();
        unsigned int hash(const state& key) const;
        node& find_node(const state& key);
        bool in_table(const state& key);
        bool in_table(const state& key, int index);
        int find_val(const state& key);
        void rehash_to(HashTable& table);
        void expand();
        void insert(node& input);
        void insert(node&& input);
        void expansion_insert(node& input);
        void expansion_insert(node&& input);
        void remove(const state& key);
        void print_table();
        int* get_num_entries();
        int* get_num_items();
        int* get_capacity();
        std::vector<std::unique_ptr<Bucket>>& get_array();
};

void print_chain(node* head);
void free_chain(std::unique_ptr<node> base);