//threadlocal.cu

#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>
#include "allocator.cuh" 

namespace thread_pool {

static constexpr int MAX_THREADS = 32; // 1 pool per thread, max 32 threads
static constexpr size_t ALIGNMENT = 8; 
static constexpr uint16_t DEFAULT_OFFSET = 0xFFFF;

static constexpr size_t HEADER_SIZE = sizeof(uint64_t); // both 8 bytes
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
    int          shared_memory_size;

    BlockHeader* free_heads[MAX_THREADS];      
    std::byte*   pool_starts[MAX_THREADS];     
    std::byte*   pool_ends[MAX_THREADS];         
};

__device__ __forceinline__ std::byte* get_shared_heap_base() {
    extern __shared__ std::byte shared_heap_base[]; // points to first byte of shared memory
    return shared_heap_base;
}

__device__ __forceinline__ size_t align_up(size_t size) {
    return (size + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
}

__device__ __forceinline__ uint16_t to_offset(BlockHeader* h_ptr, std::byte* pool_start) {
    if (h_ptr == nullptr) return DEFAULT_OFFSET;
    return static_cast<uint16_t>(reinterpret_cast<std::byte*>(h_ptr) - pool_start);
}

__device__ __forceinline__ BlockHeader* from_offset(uint16_t offset, std::byte* pool_start) {
    if (offset == DEFAULT_OFFSET) return nullptr;
    return reinterpret_cast<BlockHeader*>(pool_start + offset);
}



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
    
    if (footer_from_head(left_header) != left_footer) return nullptr;

    return left_header;
}



__device__ inline void free_list_insert(BlockHeader** free_heads, BlockHeader* h_ptr, std::byte* pool_start) {
    h_ptr->bits.free = 1;
    h_ptr->bits.prev = DEFAULT_OFFSET;
    h_ptr->bits.next = DEFAULT_OFFSET;

    if (*free_heads == nullptr) {
        *free_heads = h_ptr;
        return;
    }

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
        *free_heads = next; 
    }

    if (next != nullptr) {
        next->bits.prev = h_ptr->bits.prev;
    }

    h_ptr->bits.prev = DEFAULT_OFFSET;
    h_ptr->bits.next = DEFAULT_OFFSET;
    h_ptr->bits.free = 0; 
}



__device__ inline BlockHeader* split_block(BlockHeader** free_heads, BlockHeader* h_ptr, size_t requested_size, std::byte* pool_start) {
    size_t original_total = block_total_size(h_ptr);

    size_t total_to_use = align_up(HEADER_SIZE + requested_size + FOOTER_SIZE);
    

    size_t aligned_payload = total_to_use - HEADER_SIZE - FOOTER_SIZE;

    h_ptr->bits.free = 0;
    h_ptr->bits.size = aligned_payload;
    footer_from_head(h_ptr)->header = h_ptr;

    size_t remain_total = original_total - total_to_use;

    std::byte* rem_start = reinterpret_cast<std::byte*>(h_ptr) + total_to_use;

    BlockHeader* r_ptr = reinterpret_cast<BlockHeader*>(rem_start);
    
    r_ptr->bits.size = remain_total - HEADER_SIZE - FOOTER_SIZE;
    r_ptr->bits.free = 1;
    r_ptr->bits.prev = DEFAULT_OFFSET;
    r_ptr->bits.next = DEFAULT_OFFSET;
    
    footer_from_head(r_ptr)->header = r_ptr;
    
    free_list_insert(free_heads, r_ptr, pool_start);
    
    return h_ptr;
}

__device__ inline BlockHeader* coalesce(BlockHeader** free_heads, BlockHeader* h, std::byte* pool_start, std::byte* pool_end) {
    // left
    while (true) {
        BlockHeader* L = left_neighbors_header(h, pool_start);
        if (L != nullptr && L->bits.free) {
            free_list_remove(free_heads, L, pool_start);
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
            free_list_remove(free_heads, R, pool_start);
            size_t new_total = block_total_size(h) + block_total_size(R);
            h->bits.size = new_total - HEADER_SIZE - FOOTER_SIZE;
            footer_from_head(h)->header = h;
        } else {
            break;
        }
    }
    return h;
}



// Public api
//total_bytes is size of shared memory when calling kernel

// SHARED MEMORY LIMITS BY ARCHITECTURE:
// - Kepler/Maxwell/Pascal (SM 3.x-6.x):  48 KB per block
// - Volta (SM 7.0):                      96 KB per block
// - Turing (SM 7.5):                     64 KB per block  
// - Ampere (SM 8.0/8.6):                 100-164 KB per block
// - Hopper (SM 9.0):                     up to 228 KB per block
//
// CONSTRAINT - 16-BIT OFFSET LIMITATION:
// Each pool uses 16-bit offsets for prev/next pointers in the free list.
// This limits individual pool size to 2^16 - 1 = 65,535 bytes (64 KB + 1).
//   - 48 KB shared memory:  ceil(48/64)  = 1 thread minimum
//   - 164 KB shared memory: ceil(164/64) = 3 threads minimum  
//   - 228 KB shared memory: ceil(228/64) = 4 threads minimum

__device__ void pool_init(std::size_t total_bytes) {
    std::byte* shared_mem_ptr = get_shared_heap_base();

    pool_managers* pool_manager = reinterpret_cast<pool_managers*>(shared_mem_ptr);
    size_t manager_struct_size = align_up(sizeof(pool_managers));

    // only works for 32 threads ONLY right now
    size_t bytes_per_thread = 0;
    if (total_bytes > manager_struct_size) {
        size_t remaining_bytes = total_bytes - manager_struct_size;
        bytes_per_thread = remaining_bytes / MAX_THREADS;
        bytes_per_thread = bytes_per_thread & ~(ALIGNMENT - 1);
        // rounds down each pool size to neareast multiple of 8
        // potentially wastes 7 bytes per pool
        // these wasted bytes sit at the end of all pools
        
        if (bytes_per_thread > 0xFFFF){ //only happens if very low number of threads
            bytes_per_thread = DEFAULT_OFFSET; // = 0xFFFF
        }
    } else {
        //something went really wrong
    }
    
    // Thread 0 -> initialize pool managers (global variable essentially)
    if (threadIdx.x == 0) {
        pool_manager->shared_memory_start = shared_mem_ptr;
        pool_manager->shared_memory_size = total_bytes;
    }

    // All active threads (0-31) initialize their own pool
    if (threadIdx.x < MAX_THREADS) {
        int i = threadIdx.x;
        
        // Calculate bounds for this specific thread
        std::byte* start = shared_mem_ptr + manager_struct_size + (i * bytes_per_thread);
        std::byte* end   = start + bytes_per_thread;
        
        pool_manager->pool_starts[i] = start;
        pool_manager->pool_ends[i]   = end;

        size_t overhead = HEADER_SIZE + FOOTER_SIZE;
        
        if (total_bytes > manager_struct_size && bytes_per_thread >= overhead) {
            size_t payload_size = bytes_per_thread - overhead;
            
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

    // Wait for Thread 0 (struct init) and all pool inits
    __syncthreads();
}

// Probably works
__device__ void* pmalloc_best_fit(std::size_t size) {



    if (threadIdx.x >= MAX_THREADS) return nullptr; // throw error in future

    std::byte* shared_mem_ptr = get_shared_heap_base();
    pool_managers* pool_manager = reinterpret_cast<pool_managers*>(shared_mem_ptr);
    
    // Thread ID = Pool ID
    int pool_id = threadIdx.x;

    size_t aligned_size = align_up(size);
    std::byte* pool_start = pool_manager->pool_starts[pool_id];
    

    BlockHeader* best_fit = nullptr;
    BlockHeader* curr = pool_manager->free_heads[pool_id];

    while (curr != nullptr) {
        if (curr->bits.size >= aligned_size) {
            best_fit = curr;
            break;
        }
        curr = get_next_node(curr, pool_start);
    }

    if (best_fit == false) {
        return nullptr;
    }

    free_list_remove(&(pool_manager->free_heads[pool_id]), best_fit, pool_start);
    best_fit = split_block(&(pool_manager->free_heads[pool_id]), best_fit, size, pool_start);
    
    return user_ptr(best_fit);
}

__device__ void* pmalloc(std::size_t size) {

    if (threadIdx.x >= MAX_THREADS) return nullptr; // throw error in future

    std::byte* shared_mem_ptr = get_shared_heap_base();
    pool_managers* pool_manager = reinterpret_cast<pool_managers*>(shared_mem_ptr);


    // keeping pool_idx variable because maybe we use multidimensional threads in future
    // only 1-D blocks for now
    // threadIdx.x + (threadIdx.y * blockDim.x) + (threadIdx.z * blockDim.x * blockDim.y)
    // Thread ID = Pool ID
    int pool_id = threadIdx.x;

    std::byte* pool_start = pool_manager->pool_starts[pool_id];

    BlockHeader* h_ptr = pool_manager->free_heads[pool_id];
    if (h_ptr == nullptr) return nullptr; // throw error in future
    if (block_total_size(h_ptr) < align_up(HEADER_SIZE + size + FOOTER_SIZE)){
        return nullptr; 
    }

    free_list_remove(&(pool_manager->free_heads[pool_id]), h_ptr, pool_start);

    std::size_t original_total = block_total_size(h_ptr);
    std::size_t needed_total   = align_up(HEADER_SIZE + size + FOOTER_SIZE);
    const std::size_t MIN_TOTAL_SPLIT = HEADER_SIZE + ALIGNMENT + FOOTER_SIZE;
    //fix bug with minimal total split

    if (original_total >= needed_total + MIN_TOTAL_SPLIT) {
        h_ptr = split_block(&(pool_manager->free_heads[pool_id]), h_ptr, size, pool_start);
        // h_ptr->bits.free = 0;                     both done in split block
        // footer_from_head(h_ptr)->header = h_ptr;
    } else {
        h_ptr->bits.free = 0;
        footer_from_head(h_ptr)->header = h_ptr;
    }

    return user_ptr(h_ptr);
}

__device__ void pfree(void* user_ptr) {
    if (user_ptr == nullptr) return;
    if (threadIdx.x >= MAX_THREADS) return; // throw an error in future

    std::byte* shared_mem_ptr = get_shared_heap_base();
    pool_managers* pool_manager = reinterpret_cast<pool_managers*>(shared_mem_ptr);
    
    //keeping pool_idx variable because maybe we use multidimensional threads in future
    //only 1-D blocks for now
    // threadIdx.x + (threadIdx.y * blockDim.x) + (threadIdx.z * blockDim.x * blockDim.y)
    int pool_idx = threadIdx.x; 
    std::byte* pool_start = pool_manager->pool_starts[pool_idx];
    std::byte* pool_end = pool_manager->pool_ends[pool_idx];
    


    BlockHeader* h_ptr = header_from_user(user_ptr);
    
    h_ptr->bits.free = 1;
    h_ptr->bits.prev = DEFAULT_OFFSET;
    h_ptr->bits.next = DEFAULT_OFFSET;
    
    // passing in a pointer to the free head ptrs
    // pool end is needed for bounds checking when coalescing right
    BlockHeader* merged = coalesce(&(pool_manager->free_heads[pool_idx]), h_ptr, pool_start, pool_end);
    free_list_insert(&(pool_manager->free_heads[pool_idx]), merged, pool_start);

}

} // namespace thread_pool
