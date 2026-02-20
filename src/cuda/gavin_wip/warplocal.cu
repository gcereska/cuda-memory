
#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>
#include <stdio.h>

#include "allocator.cuh"

namespace warp_pool {

static constexpr int WARP_SIZE = 32;
static constexpr int MAX_POOLS = 32;
static constexpr size_t ALIGNMENT = 8; 
static constexpr uint16_t DEFAULT_OFFSET = 0xFFFF;
static constexpr size_t HEADER_SIZE = sizeof(uint64_t); 
static constexpr size_t FOOTER_SIZE = sizeof(void*);

struct BlockHeader; 
struct BlockFooter;

struct BlockFooter {
    BlockHeader* header;
};

struct alignas(8) BlockHeader {
    union {
        uint64_t raw; // Allow single 64-bit atomic load/store if needed
        struct {
            // when looking at raw_data the bits go prev -> next -> size -> free
            uint64_t free    : 1;  // 1 bit                          bit 0
            uint64_t size    : 31; // 31 bits merged padding         bit 1-31
            uint64_t next    : 16; // 16 bits (0-65535)              bit 32-47
            uint64_t prev    : 16; // 16 bits (0-65535)              bit 48-63
            // next and prev are offsets in memory relative to start of the pool
        } bits;
    };
};

struct pool_managers {
    std::byte*   shared_memory_start;
    int          threads_per_pool;           // 
    int          num_pools;
    int          shared_memory_size;         // in bytes
    int          spinlocks[MAX_POOLS];       // Spinlocks 
    BlockHeader* free_heads[MAX_POOLS];      // Free list heads 
    std::byte*   pool_starts[MAX_POOLS];     // bound checking  
    std::byte*   pool_ends[MAX_POOLS];         
};



// round up payload size to multiple of 8
__device__ __forceinline__ size_t align_up(size_t size) {
    return (size + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
}

// helpers with offset
// convert header ptr to offset
__device__ __forceinline__ uint16_t to_offset(BlockHeader* ptr, std::byte* pool_start) {
    if (ptr == nullptr) return DEFAULT_OFFSET;
    return static_cast<uint16_t>(reinterpret_cast<std::byte*>(ptr) - pool_start);
}

// convert offset to header ptr
__device__ __forceinline__ BlockHeader* from_offset(uint16_t offset, std::byte* pool_start) {
    if (offset == DEFAULT_OFFSET) return nullptr;
    return reinterpret_cast<BlockHeader*>(pool_start + offset);
}

// -------------------------------------------------------------------------
// Block Access Helpers (Your Style)
// -------------------------------------------------------------------------

__device__ inline size_t block_total_size(BlockHeader* h_ptr) {
    return (HEADER_SIZE + h_ptr->bits.size + FOOTER_SIZE);
}

__device__ inline BlockFooter* footer_from_head(BlockHeader* h_ptr) {
    return reinterpret_cast<BlockFooter*>(
        reinterpret_cast<std::byte*>(h_ptr) + HEADER_SIZE + h_ptr->bits.size
    );
}

__device__ inline void* user_ptr(BlockHeader* h_ptr) {
    return reinterpret_cast<void*>(
        reinterpret_cast<std::byte*>(h_ptr) + HEADER_SIZE
    );
}

__device__ inline BlockHeader* header_from_user(void* user_ptr) {
    return reinterpret_cast<BlockHeader*>(
        reinterpret_cast<std::byte*>(user_ptr) - HEADER_SIZE
    );
}

__device__ inline BlockHeader* get_next_node(BlockHeader* h_ptr, std::byte* pool_start) {
    return from_offset(h_ptr->bits.next, pool_start);
}

__device__ inline BlockHeader* get_prev_node(BlockHeader* h_ptr, std::byte* pool_start) {
    return from_offset(h_ptr->bits.prev, pool_start);
}

__device__ inline BlockHeader* right_neighbors_header(BlockHeader* h_ptr, std::byte* pool_end) {
    std::byte* after = reinterpret_cast<std::byte*>(h_ptr) + block_total_size(h_ptr);
    if (after + HEADER_SIZE > pool_end) return nullptr;
    return reinterpret_cast<BlockHeader*>(after);
}

__device__ inline BlockHeader* left_neighbors_header(BlockHeader* h, std::byte* pool_start) {
    std::byte* before = reinterpret_cast<std::byte*>(h) - FOOTER_SIZE;
    if (before < pool_start) return nullptr;
    BlockFooter* left_footer = reinterpret_cast<BlockFooter*>(before);
    BlockHeader* left_header = left_footer->header;
    return left_header;
}

/*
std::size_t total_free_bytes(BlockHeader** free_heads, std::byte* pool_start) {

}
*/

/*
__device__ inline bool in_pool(void* user_ptr, std::byte* pool_start){

}
*/

__device__ inline void free_list_insert(BlockHeader** free_heads, BlockHeader* h_ptr, std::byte* pool_start) {

    h_ptr->bits.free = 1;
    h_ptr->bits.prev = DEFAULT_OFFSET;
    h_ptr->bits.next = DEFAULT_OFFSET;

    if (*free_heads == nullptr) {
        *free_heads = h_ptr;
        return;
    }

    // if new block is larger then head, it bcomes the new head
    if (h_ptr->bits.size >= (*free_heads)->bits.size) {
        h_ptr->bits.next = to_offset(*free_heads, pool_start);
        (*free_heads)->bits.prev = to_offset(h_ptr, pool_start);
        *free_heads = h_ptr;
        return;
    }

    BlockHeader* cur = *free_heads;
    BlockHeader* next = get_next_node(cur, pool_start);

    while (next != nullptr && next->bits.size > h_ptr->bits.size) {
        cur = next;
        next = get_next_node(cur, pool_start);
    }

    h_ptr->bits.next = to_offset(next, pool_start);
    h_ptr->bits.prev = to_offset(cur, pool_start);

    if (next != nullptr) {
        next->bits.prev = to_offset(h_ptr, pool_start);
    }
    cur->bits.next = to_offset(h_ptr, pool_start);
}

__device__ inline void free_list_remove(BlockHeader** free_heads, BlockHeader* h_ptr, std::byte* pool_start) {
    if (h_ptr == nullptr) return;
    
    BlockHeader* prev = get_prev_node(h_ptr, pool_start);
    BlockHeader* next = get_next_node(h_ptr, pool_start);

    if (prev != nullptr) {
        prev->bits.next = h_ptr->bits.next;
    } else {
        // removing head
        *free_heads = next; 
    }

    if (next != nullptr) {
        next->bits.prev = h_ptr->bits.prev;
    }

    h_ptr->bits.prev = DEFAULT_OFFSET;
    h_ptr->bits.next = DEFAULT_OFFSET;
    h_ptr->bits.free = 0; 
}

// -------------------------------------------------------------------------
// Split + Coalesce


__device__ inline BlockHeader* split_block(BlockHeader** free_heads, BlockHeader* h_ptr, size_t requested_size, std::byte* pool_start) {

    size_t original_total = block_total_size(h_ptr);

    // total_to_use must be multiple of ALIGNMENT to keep next header aligned
    size_t total_to_use = HEADER_SIZE + requested_size + FOOTER_SIZE;
    total_to_use = align_up(total_to_use);
    
    if (total_to_use > original_total) total_to_use = original_total;

    // derive aligned payload for used piece
    size_t aligned_payload = total_to_use - HEADER_SIZE - FOOTER_SIZE;

    h_ptr->bits.free = 0;
    h_ptr->bits.size = aligned_payload;
    footer_from_head(h_ptr)->header = h_ptr;

    size_t remain_total = original_total - total_to_use;
    const size_t MIN_TOTAL_SPLIT = HEADER_SIZE + ALIGNMENT + FOOTER_SIZE; // ensure at least ALIGNMENT payload

    if (remain_total >= MIN_TOTAL_SPLIT) {
        std::byte* rem_start = reinterpret_cast<std::byte*>(h_ptr) + total_to_use;
        BlockHeader* r = reinterpret_cast<BlockHeader*>(rem_start);
        
        r->bits.size = remain_total - HEADER_SIZE - FOOTER_SIZE;
        r->bits.free = 1;
        r->bits.prev = DEFAULT_OFFSET;
        r->bits.next = DEFAULT_OFFSET;
        
        footer_from_head(r)->header = r;
        
        free_list_insert(free_heads, r, pool_start);
    }
    return h_ptr;
}

__device__ inline BlockHeader* coalesce(BlockHeader** head_ptr, BlockHeader* h, std::byte* pool_start, std::byte* pool_end) {
    // left
    while (true) {
        BlockHeader* L = left_neighbors_header(h, pool_start);
        if (L != nullptr && L->bits.free) {
            free_list_remove(head_ptr, L, pool_start);
            
            size_t new_total = block_total_size(L) + block_total_size(h);
            L->bits.size = new_total - HEADER_SIZE - FOOTER_SIZE;
            
            footer_from_head(L)->header = L;
            h = L;
        } else {
            break;
        }
    }
    // right
    while (true) {
        BlockHeader* R = right_neighbors_header(h, pool_end);
        if (R != nullptr && R->bits.free) {
            free_list_remove(head_ptr, R, pool_start);
            
            size_t new_total = block_total_size(h) + block_total_size(R);
            h->bits.size = new_total - HEADER_SIZE - FOOTER_SIZE;
            
            footer_from_head(h)->header = h;
        } else {
            break;
        }
    }
    return h;
}

__device__ __forceinline__ std::byte* get_shared_heap_base() {
    extern __shared__ std::byte shared_heap_base[]; // points to first byte of shared memory
    return shared_heap_base;
}

// spinlock related helpers

__device__ __forceinline__ int get_pool_id() {
    std::byte* shared_mem_ptr = get_shared_heap_base();
    pool_managers* pool_manager = reinterpret_cast<pool_managers*>(shared_mem_ptr);
    return threadIdx.x / pool_manager->threads_per_pool;
}

__device__ __forceinline__ void acquire_lock(int* lock) {
    while (atomicCAS(lock, 0, 1) != 0) {
        // spin
    }
    __threadfence_block();
}

__device__ __forceinline__ void release_lock(int* lock) {
    __threadfence_block();
    atomicExch(lock, 0);
}

// -------------------------------------------------------------------------
// Pool Initialization
// -------------------------------------------------------------------------

// called by first warp, each thread in first warp initializes their own pool for the other warps
// first thread in first warp initailzies pool managers, the "global variables" that all pools need
// total bytes is the size of the entire shared pool, passed into pool_init
__device__ void pool_init(std::size_t total_bytes, int threads_per_pool) {
    std::byte* shared_mem_ptr = get_shared_heap_base();

    pool_managers* pool_manager = reinterpret_cast<pool_managers*>(shared_mem_ptr);
    size_t manager_struct_size = align_up(sizeof(pool_managers));

    

    

    int num_pools = (blockDim.x + threads_per_pool - 1) / threads_per_pool;

    if (num_pools > MAX_POOLS) {
        threads_per_pool = (blockDim.x + MAX_POOLS - 1) / MAX_POOLS;
        num_pools = (blockDim.x + threads_per_pool - 1) / threads_per_pool;

        if (threadIdx.x == 0){
            printf("warp_pool: %d pools requested but max is %d, "
                "auto-adjusting to %d threads/pool\n",
                num_pools, MAX_POOLS, threads_per_pool);
        }
    }

    // Calculate common constraints (all threads in warp need this info)
    //bytes per warp aka bytes per pool
    size_t bytes_per_warp = 0;
    if (total_bytes > manager_struct_size) {
        size_t remaining_bytes = total_bytes - manager_struct_size;
        bytes_per_warp = remaining_bytes / num_pools;
        bytes_per_warp = bytes_per_warp & ~(ALIGNMENT - 1);
        if (bytes_per_warp > 0xFFFF) bytes_per_warp = 0xFFFF;
    }

    // 1. Thread 0: Initialize Global Metadata
    if (threadIdx.x == 0) {
        pool_manager->shared_memory_start = shared_mem_ptr;
        pool_manager->shared_memory_size = total_bytes;
        pool_manager->threads_per_pool = threads_per_pool;
        pool_manager->num_pools = num_pools;
    }

    // 2. First Warp (Threads 0-31): Initialize Pools in Parallel
    if (threadIdx.x < num_pools) {
        int i = threadIdx.x;
        
        pool_manager->spinlocks[i] = 0;
        
        // Calculate bounds for this specific warp
        std::byte* start = shared_mem_ptr + manager_struct_size + (i * bytes_per_warp);
        std::byte* end   = start + bytes_per_warp;
        
        pool_manager->pool_starts[i] = start;
        pool_manager->pool_ends[i]   = end;

        size_t overhead = HEADER_SIZE + FOOTER_SIZE;
        
        // Only setup the pool if we have enough memory
        if (total_bytes > manager_struct_size && bytes_per_warp >= overhead) {
            size_t payload_size = bytes_per_warp - overhead;
            
            BlockHeader* header = reinterpret_cast<BlockHeader*>(start);
            header->bits.free = 1;
            header->bits.size = payload_size;
            header->bits.next = DEFAULT_OFFSET;
            header->bits.prev = DEFAULT_OFFSET;
            
            BlockFooter* footer = footer_from_head(header);
            footer->header = header;

            pool_manager->free_heads[i] = header;
        } else {
            pool_manager->free_heads[i] = nullptr;
        }
    }

    // waits for rest of threads to finish before doing anything
    __syncthreads();
}

__device__ void* pmalloc(std::size_t size) {
    int pool_id = get_pool_id();
    


    std::byte* shared_mem_ptr = get_shared_heap_base();
    pool_managers* pool_manager = reinterpret_cast<pool_managers*>(shared_mem_ptr);
    std::byte* pool_start = pool_manager->pool_starts[pool_id];

    if (pool_id >= pool_manager->num_pools) {
        printf("pool_id >= num_pools during pmalloc");
        return nullptr;
    }

    acquire_lock(&pool_manager->spinlocks[pool_id]);

    void* result = nullptr;

    BlockHeader* h_ptr = pool_manager->free_heads[pool_id];
    
    if (h_ptr != nullptr && block_total_size(h_ptr) >= align_up(HEADER_SIZE + size + FOOTER_SIZE)) {
        free_list_remove(&(pool_manager->free_heads[pool_id]), h_ptr, pool_start);

        std::size_t original_total = block_total_size(h_ptr);
        std::size_t needed_total   = align_up(HEADER_SIZE + size + FOOTER_SIZE);
        const std::size_t MIN_TOTAL_SPLIT = HEADER_SIZE + ALIGNMENT + FOOTER_SIZE;

        if (original_total >= needed_total + MIN_TOTAL_SPLIT) {
            h_ptr = split_block(&(pool_manager->free_heads[pool_id]), h_ptr, size, pool_start);
        } else {
            h_ptr->bits.free = 0;
            footer_from_head(h_ptr)->header = h_ptr;
        }

        result = user_ptr(h_ptr);
    }

    release_lock(&pool_manager->spinlocks[pool_id]);

    return result;
}

__device__ void* pmalloc_best_fit(std::size_t size) {
    int pool_id = get_pool_id();


    std::byte* shared_mem_ptr = get_shared_heap_base();
    pool_managers* pool_manager = reinterpret_cast<pool_managers*>(shared_mem_ptr);
    std::byte* pool_start = pool_manager->pool_starts[pool_id];
    size_t aligned_size = align_up(size);

    if (pool_id >= pool_manager->num_pools) {
        printf("pool_id >= num_pools during pmalloc best fit");
        return nullptr;
    }

    acquire_lock(&pool_manager->spinlocks[pool_id]);

    void* result = nullptr;

    // Walk free list to find smallest block that fits
    // List is sorted descending, so keep going until too small
    BlockHeader* best_fit = nullptr;
    BlockHeader* curr = pool_manager->free_heads[pool_id];

    while (curr != nullptr) {
        if (curr->bits.size >= aligned_size) {
            best_fit = curr;  // Fits, keep looking for smaller
        } else {
            break;  // Too small, stop
        }
        curr = get_next_node(curr, pool_start);
    }

    if (best_fit != nullptr) {
        free_list_remove(&(pool_manager->free_heads[pool_id]), best_fit, pool_start);

        std::size_t original_total = block_total_size(best_fit);
        std::size_t needed_total   = align_up(HEADER_SIZE + size + FOOTER_SIZE);
        const std::size_t MIN_TOTAL_SPLIT = HEADER_SIZE + ALIGNMENT + FOOTER_SIZE;

        if (original_total >= needed_total + MIN_TOTAL_SPLIT) {
            best_fit = split_block(&(pool_manager->free_heads[pool_id]), best_fit, size, pool_start);
        } else {
            best_fit->bits.free = 0;
            footer_from_head(best_fit)->header = best_fit;
        }

        result = user_ptr(best_fit);
    }

    release_lock(&pool_manager->spinlocks[pool_id]);

    return result;
}

__device__ void pfree(void* ptr) {
    if (ptr == nullptr) return;
    
    int pool_id = get_pool_id();
    

    std::byte* shared_mem_ptr = get_shared_heap_base();
    pool_managers* pool_manager = reinterpret_cast<pool_managers*>(shared_mem_ptr);
    std::byte* pool_start = pool_manager->pool_starts[pool_id];
    std::byte* pool_end = pool_manager->pool_ends[pool_id];

    if (pool_id >= pool_manager->num_pools) {
        printf("pool_id >= num_pools during pfree");
        return;
    }

    acquire_lock(&pool_manager->spinlocks[pool_id]);

    BlockHeader* h_ptr = header_from_user(ptr);
    
    h_ptr->bits.free = 1;
    h_ptr->bits.prev = DEFAULT_OFFSET;
    h_ptr->bits.next = DEFAULT_OFFSET;
    
    BlockHeader* merged = coalesce(&(pool_manager->free_heads[pool_id]), h_ptr, pool_start, pool_end);
    free_list_insert(&(pool_manager->free_heads[pool_id]), merged, pool_start);

    release_lock(&pool_manager->spinlocks[pool_id]);
}

} // namespace warp_pool