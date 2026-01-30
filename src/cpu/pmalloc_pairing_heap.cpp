#include "pmalloc_pairing_heap.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <new>
#include <cstring>
#include <unordered_map>
#include <cstdlib>

namespace pmem_pairing {

namespace {

struct BlockHeader;
struct BlockFooter { BlockHeader* header; };

struct BlockHeader {
    std::size_t  size; // payload size in bytes
    bool         free; // in free or used structure
    BlockHeader* prev; // used-list prev (not used by heap itself) // only used for used list
    BlockHeader* next; // used-list next (not used by heap itself) // only used for used list
};

inline constexpr std::size_t HEADER_SIZE = sizeof(BlockHeader);
inline constexpr std::size_t FOOTER_SIZE = sizeof(BlockFooter);
inline constexpr std::size_t MIN_SPLIT   = HEADER_SIZE + 1 + FOOTER_SIZE;

// ======== globals -> pool + used list ========
std::byte*    g_pool      = nullptr; // first byte of heap pool
std::size_t   g_pool_size = 0;       // size of pool
BlockHeader*  g_used_head = nullptr; // used list head (unsorted)

// Pairing heap node
struct PHNode {
    BlockHeader* bh;
    PHNode* child;   // first child
    PHNode* sibling; // next sibling
    PHNode* parent;  // parent
};

PHNode* g_free_root = nullptr; // pairing heap root (max heap)

// fast lookup to delete arbitrary free block with hash map
std::unordered_map<BlockHeader*, PHNode*> g_node_of;

// pointer helpers
inline BlockFooter* footer_from_head(BlockHeader* h) {
    return reinterpret_cast<BlockFooter*>(
        reinterpret_cast<std::byte*>(h) + HEADER_SIZE + h->size
    );
}
inline void* user_ptr(BlockHeader* h) {
    return reinterpret_cast<void*>(
        reinterpret_cast<std::byte*>(h) + HEADER_SIZE
    );
}
inline BlockHeader* header_from_user(void* p) {
    return reinterpret_cast<BlockHeader*>(
        reinterpret_cast<std::byte*>(p) - HEADER_SIZE
    );
}
inline bool in_pool(void* p) {
    if (g_pool == nullptr || g_pool_size == 0) return false;
    std::byte* bp = static_cast<std::byte*>(p);
    if (bp >= g_pool && bp < (g_pool + g_pool_size)) return true;
    return false;
}

// Right neighbor starts immediately after our footer
inline BlockHeader* right_neighbors_header(BlockHeader* h) {
    std::byte* after = reinterpret_cast<std::byte*>(footer_from_head(h)) + FOOTER_SIZE;
    if (after + HEADER_SIZE > g_pool + g_pool_size) return nullptr;
    return reinterpret_cast<BlockHeader*>(after);
}

// Left neighbors footer sits just before our header
inline BlockHeader* left_neighbors_header(BlockHeader* h) {
    std::byte* before_footer = reinterpret_cast<std::byte*>(h) - FOOTER_SIZE;
    if (before_footer < g_pool) return nullptr;
    BlockFooter* left_footer = reinterpret_cast<BlockFooter*>(before_footer);
    BlockHeader* left_header = left_footer->header;
    if (left_header == nullptr) return nullptr;
    if (footer_from_head(left_header) != left_footer) return nullptr;
    return left_header;
}

// Pairing Heap helpers
inline bool bh_ge(BlockHeader* a, BlockHeader* b) {
    if (a->size > b->size) return true;
    if (a->size < b->size) return false;
    if (a > b) return true; // tiebreak by address for determinism
    return false;
}

inline PHNode* ph_new(BlockHeader* bh) {
    PHNode* n = new (std::nothrow) PHNode;
    if (n == nullptr) return nullptr;
    n->bh = bh;
    n->child = nullptr;
    n->sibling = nullptr;
    n->parent = nullptr;
    return n;
}

// Meld two roots -> return new root (max-heap)
inline PHNode* ph_meld(PHNode* a, PHNode* b) {
    if (a == nullptr) return b;
    if (b == nullptr) return a;

    BlockHeader* A = a->bh;
    BlockHeader* B = b->bh;

    if (bh_ge(A, B)) {
        // make b a's first child
        b->parent = a;
        b->sibling = a->child;
        a->child = b;
        return a;
    } else {
        // make a b's first child
        a->parent = b;
        a->sibling = b->child;
        b->child = a;
        return b;
    }
}

// two pass pairwise merging of a sibling list -> returns merged root
inline PHNode* ph_two_pass(PHNode* first) {
    if (first == nullptr) return nullptr;
    // reuses sibling as a stack next pointer.
    PHNode* pairs = nullptr; // stack of merged subheaps
    PHNode* cur = first;


    while (cur != nullptr) {
        cur->parent = nullptr;
        PHNode* a = cur;
        PHNode* b = nullptr;

        PHNode* next = cur->sibling;
        if (next != nullptr) {
            b = next;
            cur = next->sibling;
        } else {
            cur = nullptr;
        }

        a->sibling = nullptr;
        if (b != nullptr) b->sibling = nullptr;

        PHNode* merged = nullptr;
        if (b != nullptr) merged = ph_meld(a, b);
        else merged = a;

        merged->sibling = pairs;
        pairs = merged;
    }

    PHNode* result = nullptr;
    PHNode* s = pairs;
    while (s != nullptr) {
        PHNode* next = s->sibling;
        s->sibling = nullptr;
        result = ph_meld(result, s);
        s = next;
    }
    return result;
}

// Insert a free block into heap
inline void ph_insert(BlockHeader* bh) {
    PHNode* n = ph_new(bh);
    if (n == nullptr) {
        std::fprintf(stderr, "pairing-heap: node allocation failed\n");
        std::abort();
    }
    g_node_of[bh] = n;
    g_free_root = ph_meld(g_free_root, n);
}

// Return max block (root) without removing
inline BlockHeader* ph_top() {
    if (g_free_root == nullptr) return nullptr;
    return g_free_root->bh;
}

// Pop max block (root) from heap
inline BlockHeader* ph_pop_max() {
    if (g_free_root == nullptr) return nullptr;

    PHNode* old_root = g_free_root;
    BlockHeader* bh = old_root->bh;

    // Merge children pairwise to form new root
    PHNode* children = old_root->child;
    if (children != nullptr) {
        // detach children's parent links
        PHNode* t = children;
        while (t != nullptr) {
            t->parent = nullptr;
            t = t->sibling;
        }
    }
    g_free_root = ph_two_pass(children);

    // cleanup map and node
    auto it = g_node_of.find(bh);
    if (it != g_node_of.end()) g_node_of.erase(it);
    delete old_root;

    bh->free = false;
    return bh;
}

// Remove inputted block from heap 
inline void ph_remove(BlockHeader* bh) {
    auto it = g_node_of.find(bh);
    if (it == g_node_of.end()) return;

    PHNode* n = it->second;
    g_node_of.erase(it);

    // If n is root, just pop_max via child merging
    if (n == g_free_root) {
        PHNode* children = n->child;
        if (children != nullptr) {
            PHNode* t = children;
            while (t != nullptr) { t->parent = nullptr; t = t->sibling; }
        }
        g_free_root = ph_two_pass(children);
        delete n;
        return;
    }

    // Detach n from parent's child list
    PHNode* parent = n->parent;
    if (parent != nullptr) {
        PHNode* c = parent->child;
        if (c == n) {
            parent->child = n->sibling;
        } else {
            PHNode* prev = c;
            while (prev != nullptr && prev->sibling != n) prev = prev->sibling;
            if (prev != nullptr) prev->sibling = n->sibling;
        }
    }

    // Merge n's children into root
    PHNode* children = n->child;
    if (children != nullptr) {
        PHNode* t = children;
        while (t != nullptr) { t->parent = nullptr; t = t->sibling; }
        PHNode* merged = ph_two_pass(children);
        g_free_root = ph_meld(g_free_root, merged);
    }

    delete n;
}

// Clear heap
inline void ph_clear() {
    // We can walk and delete every node, but since nodes are freed only when blocks leave the heap,
    // and were destroying the pool here, we can just drop pointers and let the OS reclaim.
    // or i will iterative-delete roots:
    while (g_free_root != nullptr) {
        PHNode* old = g_free_root;
        PHNode* children = old->child;
        if (children != nullptr) {
            PHNode* t = children;
            while (t != nullptr) { t->parent = nullptr; t = t->sibling; }
        }
        g_free_root = ph_two_pass(children);
        delete old;
    }
    g_node_of.clear();
}


inline void free_list_insert(BlockHeader* h) {
    h->free = true;
    h->prev = nullptr;
    h->next = nullptr;
    ph_insert(h);
}


inline void free_list_remove(BlockHeader* h) {
    if (h == nullptr) return;
    ph_remove(h);
    h->free = false;
    h->prev = nullptr;
    h->next = nullptr;
}

inline BlockHeader* free_list_top() {
    return ph_top();
}

inline BlockHeader* free_list_pop_max() {
    return ph_pop_max();
}

inline void used_list_insert(BlockHeader* h) {
    h->prev = nullptr;
    h->next = g_used_head;
    if (g_used_head != nullptr) g_used_head->prev = h;
    g_used_head = h;
}

inline void used_list_remove(BlockHeader* h) {
    if (h == nullptr) return;
    if (h->prev != nullptr) {
        h->prev->next = h->next;
    } else {
        g_used_head = h->next;
    }
    if (h->next != nullptr) h->next->prev = h->prev;
    h->prev = nullptr;
    h->next = nullptr;
}


inline std::size_t total_free_bytes_impl_node(PHNode* n) {
    if (n == nullptr) return 0;
    std::size_t sum = 0;

    PHNode* stack = n;

    PHNode* cur = n;

    return sum; // placeholder 
}

inline std::size_t total_free_bytes_impl_rec(PHNode* n) {
    if (n == nullptr) return 0;
    std::size_t sum = n->bh->size;
    sum += total_free_bytes_impl_rec(n->child);
    sum += total_free_bytes_impl_rec(n->sibling);
    return sum;
}





inline BlockHeader* split_block(BlockHeader* h, std::size_t want) {
    h->free = false;
    std::size_t total  = HEADER_SIZE + h->size + FOOTER_SIZE;
    std::size_t used   = HEADER_SIZE + want   + FOOTER_SIZE;
    std::size_t remain = total - used;

    h->size = want;
    footer_from_head(h)->header = h;

    if (remain >= MIN_SPLIT) {
        std::byte* rem_start = reinterpret_cast<std::byte*>(footer_from_head(h)) + FOOTER_SIZE;
        BlockHeader* r = reinterpret_cast<BlockHeader*>(rem_start);
        r->size = remain - HEADER_SIZE - FOOTER_SIZE;
        r->free = true;
        r->prev = nullptr;
        r->next = nullptr;
        footer_from_head(r)->header = r;
        free_list_insert(r);
    }
    return h;
}

inline BlockHeader* coalesce(BlockHeader* h) {
    // left
    while (true) {
        BlockHeader* L = left_neighbors_header(h);
        if (L != nullptr && L->free) {
            free_list_remove(L);
            std::size_t new_size = (HEADER_SIZE + L->size + FOOTER_SIZE + h->size);
            L->size = new_size;
            footer_from_head(L)->header = L;
            h = L;
        } else {
            break;
        }
    }
    // right
    while (true) {
        BlockHeader* R = right_neighbors_header(h);
        if (R != nullptr && R->free) {
            free_list_remove(R);
            std::size_t new_size = (HEADER_SIZE + h->size + FOOTER_SIZE + R->size);
            h->size = new_size;
            footer_from_head(h)->header = h;
        } else {
            break;
        }
    }
    return h;
}

} // anonymous namespace

// Public API

void init_pool_with_bytes(std::size_t bytes) {
    if (g_pool != nullptr) {
        std::fprintf(stderr, "init_pool_with_bytes: pool already initialized\n");
        std::abort();
    }

    const std::size_t MIN_POOL = HEADER_SIZE + 1 + FOOTER_SIZE;
    if (bytes < MIN_POOL) {
        std::fprintf(stderr, "init_pool_with_bytes: need at least %zu bytes\n", MIN_POOL);
        std::abort();
    }

    g_pool = new (std::nothrow) std::byte[bytes];
    if (g_pool == nullptr) {
        std::fprintf(stderr, "init_pool_with_bytes: pool alloc failed for %zu bytes\n", bytes);
        std::abort();
    }
    g_pool_size = bytes;

    BlockHeader* h = reinterpret_cast<BlockHeader*>(g_pool);
    h->size = g_pool_size - HEADER_SIZE - FOOTER_SIZE;
    h->free = true;
    h->prev = nullptr;
    h->next = nullptr;
    footer_from_head(h)->header = h;

    g_used_head = nullptr;
    ph_clear();
    g_free_root = nullptr;
    g_node_of.clear();
    free_list_insert(h);
}

void init_pool() {
    if (g_pool != nullptr) {
        std::fprintf(stderr, "init_pool: pool already initialized\n");
        std::abort();
    }

    g_pool = new (std::nothrow) std::byte[DEFAULT_POOL_SIZE];
    if (g_pool == nullptr) {
        std::fprintf(stderr, "init_pool: pool alloc failed for %zu bytes\n", DEFAULT_POOL_SIZE);
        std::abort();
    }
    g_pool_size = DEFAULT_POOL_SIZE;

    BlockHeader* h = reinterpret_cast<BlockHeader*>(g_pool);
    h->size = g_pool_size - HEADER_SIZE - FOOTER_SIZE;
    h->free = true;
    h->prev = nullptr;
    h->next = nullptr;
    footer_from_head(h)->header = h;

    g_used_head = nullptr;
    ph_clear();
    g_free_root = nullptr;
    g_node_of.clear();
    free_list_insert(h);
}

void destroy_pool() {
    delete[] g_pool;
    g_pool = nullptr;
    g_pool_size = 0;
    g_used_head = nullptr;
    ph_clear();
    g_free_root = nullptr;
    g_node_of.clear();
}

void* pmalloc(std::size_t size) {
    if (size == 0) return nullptr;

    BlockHeader* top = free_list_top();
    if (top == nullptr || top->size < size) {
        if (top == nullptr || top->size < size) return nullptr;
    }

    BlockHeader* h = free_list_pop_max();
    if (h == nullptr) return nullptr;

    if (h->size > size && (h->size - size) >= MIN_SPLIT) {
        split_block(h, size);
    } else {
        h->size = size;
        footer_from_head(h)->header = h;
    }

    h->free = false;
    used_list_insert(h);
    return user_ptr(h);
}

void pfree(void* p) {
    if (p == nullptr) return;
    if (in_pool(p) == false) {
        std::fprintf(stderr, "pfree: pointer not from inside pool (%p)\n", p);
        return;
    }

    BlockHeader* h = header_from_user(p);

    used_list_remove(h);

    h->free = true;
    h->prev = nullptr;
    h->next = nullptr;

    BlockHeader* h2 = coalesce(h);
    free_list_insert(h2);
}

void* pmemcpy(void* dst, const void* src, std::size_t n) {
    if (dst == src || n == 0) return dst;
    std::byte* d = static_cast<std::byte*>(dst);
    const std::byte* s = static_cast<const std::byte*>(src);
    std::size_t i = 0;
    while (i < n) {
        d[i] = s[i];
        ++i;
    }
    return dst;
}

void* memfill(void* s, int c, std::size_t n) {
    std::byte* b = static_cast<std::byte*>(s);
    std::byte v = static_cast<std::byte>(c);
    std::size_t i = 0;
    while (i < n) {
        b[i] = v;
        ++i;
    }
    return s;
}

std::size_t total_free_bytes() {
    return total_free_bytes_impl_rec(g_free_root);
}

std::size_t largest_free_block() {
    BlockHeader* h = free_list_top();
    if (h != nullptr) return h->size;
    return 0;
}

} // namespace pmem_pairing
