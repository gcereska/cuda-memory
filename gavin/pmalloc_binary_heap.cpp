#include "pmalloc_binary_heap.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <new>
#include <cstring>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cstdlib>

namespace pmem_binheap {

namespace {

struct BlockHeader;
struct BlockFooter {
    BlockHeader* header;
};
struct BlockHeader {
    std::size_t  size;
    bool         free;
    BlockHeader* prev; // used-list prev (not used by heap itself) // only used for used list
    BlockHeader* next; // used-list next (not used by heap itself) // only used for used list
};

inline constexpr std::size_t HEADER_SIZE = sizeof(BlockHeader);
inline constexpr std::size_t FOOTER_SIZE = sizeof(BlockFooter);
inline constexpr std::size_t MIN_SPLIT   = HEADER_SIZE + 1 + FOOTER_SIZE;

// globals -> pool + lists/heap 
std::byte*    g_pool      = nullptr;
std::size_t   g_pool_size = 0;
BlockHeader*  g_used_head = nullptr;

// Binary heap for free blocks (max-heap by size)
std::vector<BlockHeader*>                 g_free_heap;
std::unordered_map<BlockHeader*, size_t>  g_heap_pos;

// Pointer Helpers
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

// Left neighbor's footer sits just before our header
inline BlockHeader* left_neighbors_header(BlockHeader* h) {
    std::byte* before_footer = reinterpret_cast<std::byte*>(h) - FOOTER_SIZE;
    if (before_footer < g_pool) return nullptr;
    BlockFooter* left_footer = reinterpret_cast<BlockFooter*>(before_footer);
    BlockHeader* left_header = left_footer->header;
    if (left_header == nullptr) return nullptr;
    if (footer_from_head(left_header) != left_footer) return nullptr;
    return left_header;
}

// Binary Heap Helpers
inline bool heap_less(size_t i, size_t j) {
    BlockHeader* a = g_free_heap[i];
    BlockHeader* b = g_free_heap[j];
    if (a->size < b->size) return true;
    if (a->size > b->size) return false;
    if (a < b) return true;    // tie-breaker is address could change to make a or b
    return false;
}

inline void heap_swap(size_t i, size_t j) {
    std::swap(g_free_heap[i], g_free_heap[j]);
    g_heap_pos[g_free_heap[i]] = i;
    g_heap_pos[g_free_heap[j]] = j;
}

inline void heap_sift_up(size_t i) {
    while (i > 0) {
        size_t p = (i - 1) / 2;
        if (!heap_less(p, i)) break;
        heap_swap(p, i);
        i = p;
    }
}

inline void heap_sift_down(size_t i) {
    const size_t n = g_free_heap.size();
    while (true) {
        size_t l = 2 * i + 1;
        size_t r = 2 * i + 2;
        size_t best = i;

        if (l < n && heap_less(best, l)) best = l;
        if (r < n && heap_less(best, r)) best = r;

        if (best == i) break;
        heap_swap(i, best);
        i = best;
    }
}

inline void free_heap_clear() {
    g_free_heap.clear();
    g_heap_pos.clear();
}

inline BlockHeader* free_list_top() {
    if (g_free_heap.empty()) return nullptr;
    return g_free_heap[0];
}

// pop max or returns nullptr if empty
inline BlockHeader* free_list_pop_max() {
    if (g_free_heap.empty()) return nullptr;
    BlockHeader* top = g_free_heap[0];

    size_t last = g_free_heap.size() - 1;
    heap_swap(0, last);
    g_heap_pos.erase(g_free_heap[last]);
    g_free_heap.pop_back();

    if (!g_free_heap.empty()) heap_sift_down(0);

    top->free = false;
    return top;
}


inline void free_list_insert(BlockHeader* h) {
    h->free = true;
    h->prev = nullptr;
    h->next = nullptr;

    g_heap_pos[h] = g_free_heap.size();
    g_free_heap.push_back(h);
    heap_sift_up(g_free_heap.size() - 1);
}

// Remove select block h 
inline void free_list_remove(BlockHeader* h) {
    if (h == nullptr) return;
    auto it = g_heap_pos.find(h);
    if (it == g_heap_pos.end()) return;

    size_t i  = it->second;
    size_t n1 = g_free_heap.size() - 1;

    if (i != n1) heap_swap(i, n1);

    g_heap_pos.erase(g_free_heap[n1]);
    g_free_heap.pop_back();

    if (i < g_free_heap.size()) {
        // i think you only need either up or down 
        // but doing both keeps it simple and correct
        // i could optimize later if needed
        heap_sift_up(i);
        heap_sift_down(i);
    }

    h->free = false;
    h->prev = nullptr;
    h->next = nullptr;
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


inline std::size_t total_free_bytes() {
    std::size_t total = 0;
    std::size_t i = 0;
    while (i < g_free_heap.size()) {
        total += g_free_heap[i]->size;
        ++i;
    }
    return total;
}
inline std::size_t largest_free_block() {
    BlockHeader* h = free_list_top();
    if (h != nullptr) return h->size;
    return 0;
}

// ======== Split / Coalesce ========
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
    free_heap_clear();
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
    free_heap_clear();
    free_list_insert(h);
}

void destroy_pool() {
    delete[] g_pool;
    g_pool = nullptr;
    g_pool_size = 0;
    g_used_head = nullptr;
    free_heap_clear();
}

void* pmalloc(std::size_t size) {
    if (size == 0) return nullptr;

    BlockHeader* top = free_list_top();
    if (top == nullptr || top->size < size) {
        // early exit if no free block large enough
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

} // namespace pmem_binheap
