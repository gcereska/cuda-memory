// pmalloc_intrusive.cpp
#include "pmalloc_intrusive.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <new>
#include <cstring>
#include <cstdlib>

namespace pmem_intrusive {

namespace {


struct BlockHeader;
struct BlockFooter {
    BlockHeader* header;
};
struct BlockHeader {
    std::size_t  size; // payload size in bytes
    bool         free; // in free list or used list
    BlockHeader* prev; // prev in free/used list
    BlockHeader* next; // next in free/used list
};

constexpr std::size_t HEADER_SIZE = sizeof(BlockHeader);
constexpr std::size_t FOOTER_SIZE = sizeof(BlockFooter);

// min size of a block header + at least 1 byte + footer
constexpr std::size_t MIN_SPLIT = HEADER_SIZE + 1 + FOOTER_SIZE;

// ======== globals -> pool + lists ========
std::byte*    g_pool      = nullptr; // points to first byte of heap pool
std::size_t   g_pool_size = 0;       // single source of truth for pool size
BlockHeader*  g_free_head = nullptr; // free list head (largest-first)
BlockHeader*  g_used_head = nullptr; // used list head (unsorted)

// ======== Pointer helpers ========
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
    if (after + HEADER_SIZE > g_pool + g_pool_size) return nullptr; // not enough room for a header
    return reinterpret_cast<BlockHeader*>(after);
}

// Left neighbors footer sits just before our header
inline BlockHeader* left_neighbors_header(BlockHeader* h) {
    std::byte* before_footer = reinterpret_cast<std::byte*>(h) - FOOTER_SIZE;
    if (before_footer < g_pool) return nullptr; // before pool -> none
    BlockFooter* left_footer = reinterpret_cast<BlockFooter*>(before_footer);
    BlockHeader* left_header = left_footer->header;
    if (left_header == nullptr) return nullptr;
    if (footer_from_head(left_header) != left_footer) return nullptr; // footer must match its header
    return left_header;
}

// intrusive doubly linked list functions size sorted largest top
inline void free_list_insert(BlockHeader* h) {
    h->free = true;
    h->prev = nullptr;
    h->next = nullptr;

    if (g_free_head == nullptr) {
        g_free_head = h;
        return;
    }
    if (h->size >= g_free_head->size) {
        h->next = g_free_head;
        g_free_head->prev = h;
        g_free_head = h;
        return;
    }
    BlockHeader* current = g_free_head;
    while (current->next != nullptr && current->next->size > h->size) {
        current = current->next;
    }
    h->next = current->next;
    h->prev = current;
    if (current->next != nullptr) current->next->prev = h;
    current->next = h;
}

inline void free_list_remove(BlockHeader* h) {
    if (h == nullptr) return;
    if (h->prev != nullptr) {
        h->prev->next = h->next;
    } else {
        g_free_head = h->next;
    }
    if (h->next != nullptr) {
        h->next->prev = h->prev;
    }
    h->prev = nullptr;
    h->next = nullptr;
    h->free = false;
}

// intrusive doubly linked list functions
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

// -------------------------------------------------------------------------------------------------- Split a free block into [used | remainder]
inline BlockHeader* split_block(BlockHeader* h, std::size_t want) {
    h->free = false; // we are going to use it
    const std::size_t total  = HEADER_SIZE + h->size + FOOTER_SIZE;
    const std::size_t used   = HEADER_SIZE + want   + FOOTER_SIZE;
    const std::size_t remain = total - used;

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

// -------------------------------------------------------------------------------------------------- Coalescing
inline BlockHeader* coalesce(BlockHeader* h) {
    // left
    while (true) {
        BlockHeader* L = left_neighbors_header(h);
        if (L != nullptr && L->free) {
            free_list_remove(L);
            const std::size_t new_size = (HEADER_SIZE + L->size + FOOTER_SIZE + h->size);
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
            const std::size_t new_size = (HEADER_SIZE + h->size + FOOTER_SIZE + R->size);
            h->size = new_size;
            footer_from_head(h)->header = h;
        } else {
            break;
        }
    }
    return h;
}

} // anonymous namespace

// -------------------------------------------------------------------------------------------------- Public API 

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

    g_free_head = nullptr;
    g_used_head = nullptr;
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

    g_free_head = nullptr;
    g_used_head = nullptr;
    free_list_insert(h);
}

void destroy_pool() {
    delete[] g_pool;
    g_pool = nullptr;
    g_pool_size = 0;
    g_free_head = nullptr;
    g_used_head = nullptr;
}

void* pmalloc(std::size_t size) {
    if (size == 0) return nullptr;

    // Take the largest free block (head of free list)
    BlockHeader* h = g_free_head;
    if (h == nullptr || h->size < size) {
        if (total_free_bytes() >= size) {
            // Potential place for a defrag/compress hook if you add one.
            h = g_free_head;
        }
        if (h == nullptr || h->size < size) return nullptr; // still no fit
    }

    free_list_remove(g_free_head); // pop head

    if (h->size > size && (h->size - size) >= MIN_SPLIT) {
        split_block(h, size); // h is now exact size
    } else {
        h->size = size; // trim to exact size
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
    std::size_t total = 0;
    BlockHeader* p = g_free_head;
    while (p != nullptr) {
        total += p->size;
        p = p->next;
    }
    return total;
}

std::size_t largest_free_block() {
    if (g_free_head != nullptr) {
        return g_free_head->size;
    } else {
        return 0;
    }
}

} // namespace pmem_intrusive
