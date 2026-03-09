#include <iostream>
#include <cstdint>
#include <vector>
#include <shared_mutex>
#include <algorithm>
#include <memory>
#include <mutex>


#pragma once


struct id_node {
    int key;
    std::string value;
    std::unique_ptr<id_node> next;
    id_node() = default; // 0. Default Constructor
    id_node(int k, std::string v) : key(k), value(v), next(nullptr) {} // 1. Normal Constructor
    id_node(const id_node& other) : key(other.key), value(other.value), next(nullptr) {} // 2. Copy Constructor
};

struct id_bucket {
    std::unique_ptr<id_node> head;
    std::shared_mutex lock;
    id_bucket()                              = default;   // 0. Default Constructor
    id_bucket(std::unique_ptr<id_node> other);               // 1. Normal Constructor
    ~id_bucket()                              = default;  // 2. Destructor
    id_bucket(const id_bucket& other)            = delete;   // 3. Copy Constructor
    id_bucket& operator=(const id_bucket& other) = delete;   // 4. Copy Assignment Operator
    id_bucket(id_bucket&& other)                 = delete;   // 5. Move Constructor
    id_bucket& operator=(id_bucket&& other)      = delete;   // 6. Move Assignment Operator
    void print_id_bucket() const;
    void free_id_bucket();
    bool empty() const noexcept;
    int id_bucket_insert(id_node& input);
    id_node pop();
};

std::ostream& operator<<(std::ostream& out, const id_node& input);
void print_id_node(const id_node* input);

class IdHashMap {
    private:
        std::vector<std::unique_ptr<id_bucket>> array;
        int num_entries_;
        int num_items_;
        int capacity_;
        std::shared_mutex table_lock;
        std::hash<int> hasher_;
        
    public:
        IdHashMap();                                                 // 0. Default Constructor
        IdHashMap(int capacity_);                                     // 1. Normal Constructor
        ~IdHashMap()                                     = default;  // 2. Destructor
        IdHashMap(const IdHashMap& other)                = delete;   // 3. Copy Constructor
        IdHashMap& operator=(const IdHashMap& other)     = delete;   // 4. Copy Assignment Operator
        IdHashMap(IdHashMap&& other) noexcept            = delete;  // 5. Move Constructor
        IdHashMap& operator=(IdHashMap&& other) noexcept = delete;  // 6. Move Assignment Operator
        void free_table();
        unsigned int hash(const int key) const;
        id_node& find_id_node(const int key);
        bool in_table(const int key);
        bool in_table(const int key, int index);
        std::string find_val(const int key);
        void rehash_to(IdHashMap& table);
        void expand();
        void insert(const int key, const std::string& val);
        void insert(id_node& input);
        void insert(id_node&& input);
        void expansion_insert(id_node& input);
        void expansion_insert(id_node&& input);
        void remove(const int key);
        void print_table();
        int* num_entries();
        int* num_items();
        int* capacity();
        std::vector<std::unique_ptr<id_bucket>>& get_array();
};

void print_chain(id_node* head);
void free_chain(std::unique_ptr<id_node> base);