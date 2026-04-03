//benchmark_all.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cassert>
#include <unordered_set>
#include <cstring>

#include <cumem/cuda/allocator.cuh>
#include <cumem/cuda/poolAlloc.cuh>
// #include <cumem/cuda/poolAllocBST.cuh>

#include "matrix.hpp"  // host-side PMatrix (available for host-side verification if needed)

// ===================== CUDA CHECK =====================
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        std::exit(1); \
    } \
} while(0)


// Set BENCH_VERBOSE=1 to see device prints
// device prints take incosistent times to run
#ifndef BENCH_VERBOSE
#define BENCH_VERBOSE 0
#endif

#if BENCH_VERBOSE
#define BENCH_PRINTF(...) printf(__VA_ARGS__)
#else
#define BENCH_PRINTF(...) do{}while(0)
#endif

// ===================== MODES =====================
enum Allocator_Modes_enum : int {
    MODE_THREAD_FIRST_FIT=0,  // thread local
    MODE_THREAD_BEST_FIT =1,   // thread local best fit
    MODE_FREELIST        =2,      // julian freelist
    // MODE_BST             =3,           // julian BST
    MODE_DEVICE_MALLOC   =4, // native cuda
    MODE_WARP_FIRST_FIT  =5,
    MODE_WARP_BEST_FIT   =6,
    MODE_GLOBAL_FIRST_FIT=7,
    MODE_GLOBAL_BEST_FIT =8,
};

// Managed globals on unified memory so kernels can stay the same signature
__device__ __managed__ int    g_mode = MODE_THREAD_FIRST_FIT; // mode of the kernel call
__device__ __managed__ size_t g_dyn_smem_bytes = 0; // number of bytes requested for a kernel call needed for pool_init
__device__ __managed__ int    g_threads_per_pool = 1;

// Global-memory heap pointer + size (set from host before launching global modes)
__device__ __managed__ std::byte* g_global_heap = nullptr;
__device__ __managed__ size_t     g_global_heap_bytes = 0;

// we run the same kernel calls but change these global (unified) variables to switch which allocator to use

__device__ __forceinline__ bool is_global_mode_device() {
    return g_mode == MODE_GLOBAL_FIRST_FIT || g_mode == MODE_GLOBAL_BEST_FIT;
}

__device__ __forceinline__ void init_based_on_g_mode() {
    switch (g_mode) {
        case MODE_THREAD_FIRST_FIT:
            thread_pool::pool_init(g_dyn_smem_bytes);
            break;
        case MODE_THREAD_BEST_FIT:
            thread_pool::pool_init(g_dyn_smem_bytes);
            break;
        case MODE_FREELIST:
            list_pool::pool_init(g_dyn_smem_bytes);
            break;
        // case MODE_BST:
        //     bst_pool::pool_init(g_dyn_smem_bytes);
        //     break;
        case MODE_WARP_FIRST_FIT:
            warp_pool::pool_init(g_dyn_smem_bytes, g_threads_per_pool);
            break;
        case MODE_WARP_BEST_FIT:
            warp_pool::pool_init(g_dyn_smem_bytes, g_threads_per_pool);
            break;
        case MODE_GLOBAL_FIRST_FIT:
            glob_pool::pool_init(g_global_heap, g_global_heap_bytes);
            break;
        case MODE_GLOBAL_BEST_FIT:
            glob_pool::pool_init(g_global_heap, g_global_heap_bytes);
            break;
        case MODE_DEVICE_MALLOC:
        default:
            break;
    }
}

__device__ __forceinline__ void* alloc_based_on_g_mode(size_t n) {
    switch (g_mode) {
        case MODE_THREAD_FIRST_FIT:  return thread_pool::pmalloc(n);
        case MODE_THREAD_BEST_FIT:   return thread_pool::pmalloc_best_fit(n);
        case MODE_FREELIST:          return list_pool::pmalloc(n);
        // case MODE_BST:               return bst_pool::pmalloc(n);
        case MODE_DEVICE_MALLOC:     return malloc(n);
        case MODE_WARP_BEST_FIT:     return warp_pool::pmalloc_best_fit(n);
        case MODE_WARP_FIRST_FIT:    return warp_pool::pmalloc(n);
        case MODE_GLOBAL_FIRST_FIT:  return glob_pool::pmalloc(n);
        case MODE_GLOBAL_BEST_FIT:   return glob_pool::pmalloc_best_fit(n);
        default:                     return nullptr;
    }
}

__device__ __forceinline__ void free_based_on_g_mode(void* p) {
    if (!p) return;
    switch (g_mode) {
        case MODE_THREAD_FIRST_FIT:
            thread_pool::pfree(p);
            break;
        case MODE_THREAD_BEST_FIT:
            thread_pool::pfree(p);
            break;
        case MODE_FREELIST:
            list_pool::pfree(p);
            break;
        // case MODE_BST:
        //     bst_pool::pfree(p);
        //     break;
        case MODE_WARP_BEST_FIT:
            warp_pool::pfree(p);
            break;
        case MODE_WARP_FIRST_FIT:
            warp_pool::pfree(p);
            break;
        case MODE_GLOBAL_FIRST_FIT:
            glob_pool::pfree(p);
            break;
        case MODE_GLOBAL_BEST_FIT:
            glob_pool::pfree(p);
            break;
        case MODE_DEVICE_MALLOC:
            free(p);
            break;
    }
}


static constexpr int    NUM_THREADS = 32;
static constexpr size_t SHARED_MEM_ALLOC_TEST = (48 * 1024) - 16;
static constexpr size_t SHARED_MEM_FUZZY = (32 * 1024);

// Global-memory heap sizes (same as shared mem counterparts for apples-to-apples)
static constexpr size_t GLOBAL_HEAP_ALLOC_TEST = SHARED_MEM_ALLOC_TEST;
static constexpr size_t GLOBAL_HEAP_FUZZY      = SHARED_MEM_FUZZY;

// copy and pasted helpers
__device__ bool verify_memory_writable(void* ptr, size_t size) {
    if (ptr == nullptr) return false;
    uint8_t* bytes = static_cast<uint8_t*>(ptr);
    for (size_t i = 0; i < size; i++) bytes[i] = static_cast<uint8_t>(i & 0xFF);
    for (size_t i = 0; i < size; i++) if (bytes[i] != static_cast<uint8_t>(i & 0xFF)) return false;
    return true;
}
__device__ void fill_pattern(void* ptr, size_t size, uint8_t pattern) {
    uint8_t* bytes = static_cast<uint8_t*>(ptr);
    for (size_t i = 0; i < size; i++) bytes[i] = pattern;
}
__device__ bool verify_pattern(void* ptr, size_t size, uint8_t pattern) {
    uint8_t* bytes = static_cast<uint8_t*>(ptr);
    for (size_t i = 0; i < size; i++) if (bytes[i] != pattern) return false;
    return true;
}

// ===================== allocator_test.cu kernels copy and pasted ===========================

__global__ void test_init() {
    extern __shared__ std::byte heap[];
    init_based_on_g_mode();
    if (threadIdx.x == 31) BENCH_PRINTF("[TEST 1] Pool initialization: PASSED\n");
}

__global__ void test_single_alloc_free() {
    extern __shared__ std::byte heap[];
    init_based_on_g_mode();

    bool passed = true;
    void* ptr = alloc_based_on_g_mode(64);
    if (ptr == nullptr) { BENCH_PRINTF("  Thread %d: allocation failed\n", threadIdx.x); passed = false; }
    if (passed && !verify_memory_writable(ptr, 64)) { BENCH_PRINTF("  Thread %d: memory verification failed\n", threadIdx.x); passed = false; }
    free_based_on_g_mode(ptr);

    void* ptr2 = alloc_based_on_g_mode(64);
    if (ptr2 == nullptr) { BENCH_PRINTF("  Thread %d: reallocation after free failed\n", threadIdx.x); passed = false; }
    free_based_on_g_mode(ptr2);

    __syncthreads();
    if (threadIdx.x == 0) BENCH_PRINTF("[TEST 2] Single alloc/free: %s\n", passed ? "PASSED" : "FAILED");
}

__global__ void test_multiple_allocs() {
    extern __shared__ std::byte heap[];
    init_based_on_g_mode();

    bool passed = true;
    constexpr int NUM_ALLOCS = 10;
    void* ptrs[NUM_ALLOCS]{};

    for (int i = 0; i < NUM_ALLOCS; i++) {
        ptrs[i] = alloc_based_on_g_mode(32);
        if (ptrs[i] == nullptr) { BENCH_PRINTF("  Thread %d: allocation %d failed\n", threadIdx.x, i); passed = false; break; }
        fill_pattern(ptrs[i], 32, static_cast<uint8_t>(i + threadIdx.x));
    }
    if (passed) {
        for (int i = 0; i < NUM_ALLOCS; i++) {
            if (!verify_pattern(ptrs[i], 32, static_cast<uint8_t>(i + threadIdx.x))) {
                BENCH_PRINTF("  Thread %d: pattern verification failed for block %d\n", threadIdx.x, i);
                passed = false; break;
            }
        }
    }
    for (int i = 0; i < NUM_ALLOCS; i++) free_based_on_g_mode(ptrs[i]);

    __syncthreads();
    if (threadIdx.x == 0) BENCH_PRINTF("[TEST 3] Multiple sequential allocations: %s\n", passed ? "PASSED" : "FAILED");
}

__global__ void test_coalescing() {
    extern __shared__ std::byte heap[];
    init_based_on_g_mode();

    bool passed = true;
    void* ptr1 = alloc_based_on_g_mode(64);
    void* ptr2 = alloc_based_on_g_mode(64);
    void* ptr3 = alloc_based_on_g_mode(64);

    if (!ptr1 || !ptr2 || !ptr3) { BENCH_PRINTF("  Thread %d: initial allocations failed\n", threadIdx.x); passed = false; }

    if (passed) {
        free_based_on_g_mode(ptr2);
        free_based_on_g_mode(ptr1);
        free_based_on_g_mode(ptr3);

        void* big_ptr = alloc_based_on_g_mode(180);
        if (big_ptr == nullptr) { BENCH_PRINTF("  Thread %d: large allocation after coalescing failed\n", threadIdx.x); passed = false; }
        else free_based_on_g_mode(big_ptr);
    }

    __syncthreads();
    if (threadIdx.x == 0) BENCH_PRINTF("[TEST 4] Coalescing: %s\n", passed ? "PASSED" : "FAILED");
}

__global__ void test_coalescing_orders() {
    extern __shared__ std::byte heap[];
    init_based_on_g_mode();

    bool passed = true;

    void* a1 = alloc_based_on_g_mode(32);
    void* a2 = alloc_based_on_g_mode(32);
    void* a3 = alloc_based_on_g_mode(32);
    if (!a1 || !a2 || !a3) passed = false;
    else { free_based_on_g_mode(a1); free_based_on_g_mode(a2); free_based_on_g_mode(a3); }

    void* b1 = alloc_based_on_g_mode(32);
    void* b2 = alloc_based_on_g_mode(32);
    void* b3 = alloc_based_on_g_mode(32);
    if (!b1 || !b2 || !b3) passed = false;
    else { free_based_on_g_mode(b3); free_based_on_g_mode(b2); free_based_on_g_mode(b1); }

    void* c1 = alloc_based_on_g_mode(32);
    void* c2 = alloc_based_on_g_mode(32);
    void* c3 = alloc_based_on_g_mode(32);
    if (!c1 || !c2 || !c3) passed = false;
    else { free_based_on_g_mode(c1); free_based_on_g_mode(c3); free_based_on_g_mode(c2); }

    __syncthreads();
    if (threadIdx.x == 0) BENCH_PRINTF("[TEST 5] Coalescing orders (left/right/both): %s\n", passed ? "PASSED" : "FAILED");
}

__global__ void test_splitting() {
    extern __shared__ std::byte heap[];
    init_based_on_g_mode();

    bool passed = true;
    void* ptr1 = alloc_based_on_g_mode(16);
    if (!ptr1) { BENCH_PRINTF("  Thread %d: first allocation failed\n", threadIdx.x); passed = false; }
    void* ptr2 = alloc_based_on_g_mode(16);
    if (!ptr2) { BENCH_PRINTF("  Thread %d: second allocation failed\n", threadIdx.x); passed = false; }

    if (passed) {
        fill_pattern(ptr1, 16, 0xAA);
        fill_pattern(ptr2, 16, 0xBB);
        if (!verify_pattern(ptr1, 16, 0xAA)) { BENCH_PRINTF("  Thread %d: ptr1 was corrupted\n", threadIdx.x); passed = false; }
        if (!verify_pattern(ptr2, 16, 0xBB)) { BENCH_PRINTF("  Thread %d: ptr2 was corrupted\n", threadIdx.x); passed = false; }
    }

    free_based_on_g_mode(ptr1);
    free_based_on_g_mode(ptr2);

    __syncthreads();
    if (threadIdx.x == 0) BENCH_PRINTF("[TEST 6] Block splitting: %s\n", passed ? "PASSED" : "FAILED");
}

__global__ void test_size_boundaries() {
    extern __shared__ std::byte heap[];
    init_based_on_g_mode();

    bool passed = true;
    void* tiny = alloc_based_on_g_mode(1);
    if (!tiny) { BENCH_PRINTF("  Thread %d: 1-byte allocation failed\n", threadIdx.x); passed = false; }
    else { verify_memory_writable(tiny, 1); free_based_on_g_mode(tiny); }

    size_t test_sizes[] = {7,8,9,15,16,17,31,32,33,63,64,65};
    for (size_t size : test_sizes) {
        void* ptr = alloc_based_on_g_mode(size);
        if (!ptr) { BENCH_PRINTF("  Thread %d: %zu-byte allocation failed\n", threadIdx.x, size); passed = false; }
        else {
            if (!verify_memory_writable(ptr, size)) { BENCH_PRINTF("  Thread %d: %zu-byte memory verification failed\n", threadIdx.x, size); passed = false; }
            free_based_on_g_mode(ptr);
        }
    }

    __syncthreads();
    if (threadIdx.x == 0) BENCH_PRINTF("[TEST 7] Size boundaries: %s\n", passed ? "PASSED" : "FAILED");
}

__global__ void test_exhaustion() {
    extern __shared__ std::byte heap[];
    init_based_on_g_mode();

    bool passed = true;
    constexpr int MAX_PTRS = 100;
    void* ptrs[MAX_PTRS]{};
    int num_allocated = 0;

    for (int i = 0; i < MAX_PTRS; i++) {
        ptrs[i] = alloc_based_on_g_mode(64);
        if (!ptrs[i]) break;
        fill_pattern(ptrs[i], 64, static_cast<uint8_t>(i));
        num_allocated++;
    }
    if (num_allocated == 0) { BENCH_PRINTF("  Thread %d: couldn't allocate_from_g_modething\n", threadIdx.x); passed = false; }

    for (int i = 0; i < num_allocated; i++) {
        if (!verify_pattern(ptrs[i], 64, static_cast<uint8_t>(i))) { BENCH_PRINTF("  Thread %d: allocation %d corrupted\n", threadIdx.x, i); passed = false; break; }
    }
    for (int i = 0; i < num_allocated; i++) free_based_on_g_mode(ptrs[i]);

    void* recovery = alloc_based_on_g_mode(64);
    if (!recovery) { BENCH_PRINTF("  Thread %d: recovery allocation failed\n", threadIdx.x); passed = false; }
    else free_based_on_g_mode(recovery);

    __syncthreads();
    if (threadIdx.x == 0) BENCH_PRINTF("[TEST 8] Exhaustion and recovery: %s\n", passed ? "PASSED" : "FAILED");
}

__global__ void test_fragmentation() {
    extern __shared__ std::byte heap[];
    init_based_on_g_mode();

    bool passed = true;
    constexpr int NUM_BLOCKS = 20;
    void* ptrs[NUM_BLOCKS]{};

    for (int i = 0; i < NUM_BLOCKS; i++) {
        ptrs[i] = alloc_based_on_g_mode(32);
        if (!ptrs[i]) { for (int j=i;j<NUM_BLOCKS;j++) ptrs[j]=nullptr; break; }
    }
    for (int i = 0; i < NUM_BLOCKS; i += 2) { free_based_on_g_mode(ptrs[i]); ptrs[i] = nullptr; }
    for (int i = 0; i < NUM_BLOCKS; i += 2) ptrs[i] = alloc_based_on_g_mode(32);
    for (int i = 0; i < NUM_BLOCKS; i++) free_based_on_g_mode(ptrs[i]);

    void* big = alloc_based_on_g_mode(256);
    if (!big) { BENCH_PRINTF("  Thread %d: large alloc after defrag failed\n", threadIdx.x); passed = false; }
    else free_based_on_g_mode(big);

    __syncthreads();
    if (threadIdx.x == 0) BENCH_PRINTF("[TEST 9] Fragmentation stress: %s\n", passed ? "PASSED" : "FAILED");
}

__global__ void test_free_list_order() {
    extern __shared__ std::byte heap[];
    init_based_on_g_mode();

    bool passed = true;
    void* small  = alloc_based_on_g_mode(16);
    void* medium = alloc_based_on_g_mode(64);
    void* large  = alloc_based_on_g_mode(128);

    if (!small || !medium || !large) passed = false;
    else {
        free_based_on_g_mode(small);
        free_based_on_g_mode(large);
        free_based_on_g_mode(medium);

        void* first  = alloc_based_on_g_mode(100);
        void* second = alloc_based_on_g_mode(50);
        void* third  = alloc_based_on_g_mode(10);

        if (!first || !second || !third) { BENCH_PRINTF("  Thread %d: reallocation failed\n", threadIdx.x); passed = false; }

        free_based_on_g_mode(first);
        free_based_on_g_mode(second);
        free_based_on_g_mode(third);
    }

    __syncthreads();
    if (threadIdx.x == 0) BENCH_PRINTF("[TEST 10] Free list ordering: %s\n", passed ? "PASSED" : "FAILED");
}

__global__ void test_null_free() {
    extern __shared__ std::byte heap[];
    init_based_on_g_mode();

    free_based_on_g_mode(nullptr);
    void* ptr = alloc_based_on_g_mode(32);
    bool passed = (ptr != nullptr);
    free_based_on_g_mode(ptr);

    __syncthreads();
    if (threadIdx.x == 0) BENCH_PRINTF("[TEST 11] Null pointer free: %s\n", passed ? "PASSED" : "FAILED");
}

__global__ void test_double_free() {
    extern __shared__ std::byte heap[];
    init_based_on_g_mode();

    void* ptr = alloc_based_on_g_mode(64);
    free_based_on_g_mode(ptr);
    //free_based_on_g_mode(ptr); // undefined but should not crash

    void* ptr2 = alloc_based_on_g_mode(32);
    bool passed = true;
    free_based_on_g_mode(ptr2);

    __syncthreads();
    if (threadIdx.x == 0) BENCH_PRINTF("[TEST 12] Double free (no crash): %s\n", passed ? "PASSED" : "FAILED");
}

__global__ void test_max_allocation() {
    extern __shared__ std::byte heap[];
    init_based_on_g_mode();

    bool passed = true;
    size_t heap_bytes = is_global_mode_device() ? g_global_heap_bytes : g_dyn_smem_bytes;
    size_t approx_pool_size = (heap_bytes - 512) / 32;

    void* ptr = nullptr;
    size_t successful_size = 0;

    for (size_t try_size = approx_pool_size; try_size >= 64; try_size -= 64) {
        ptr = alloc_based_on_g_mode(try_size);
        if (ptr) { successful_size = try_size; break; }
    }

    if (!ptr) { BENCH_PRINTF("  Thread %d: couldn't find working allocation size\n", threadIdx.x); passed = false; }
    else {
        if (!verify_memory_writable(ptr, successful_size)) { BENCH_PRINTF("  Thread %d: max allocation not fully writable\n", threadIdx.x); passed = false; }
        free_based_on_g_mode(ptr);
    }

    __syncthreads();
    if (threadIdx.x == 0) BENCH_PRINTF("[TEST 13] Maximum single allocation: %s\n", passed ? "PASSED" : "FAILED");
}

__global__ void test_alignment() {
    extern __shared__ std::byte heap[];
    init_based_on_g_mode();

    bool passed = true;
    size_t sizes[] = {1,3,5,7,9,13,17,31,33,63,65,100};

    for (size_t size : sizes) {
        void* ptr = alloc_based_on_g_mode(size);
        if (ptr) {
            uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
            if (addr % 8 != 0) { BENCH_PRINTF("  Thread %d: size %zu returned unaligned ptr %p\n", threadIdx.x, size, ptr); passed = false; }
            free_based_on_g_mode(ptr);
        }
    }

    __syncthreads();
    if (threadIdx.x == 0) BENCH_PRINTF("[TEST 14] Alignment verification: %s\n", passed ? "PASSED" : "FAILED");
}

__global__ void test_interleaved() {
    extern __shared__ std::byte heap[];
    init_based_on_g_mode();

    bool passed = true;
    void* a = alloc_based_on_g_mode(32);
    void* b = alloc_based_on_g_mode(32);

    if (a) fill_pattern(a, 32, 0xAA);
    if (b) fill_pattern(b, 32, 0xBB);

    free_based_on_g_mode(a);

    void* c = alloc_based_on_g_mode(32);
    if (c) fill_pattern(c, 32, 0xCC);

    if (b && !verify_pattern(b, 32, 0xBB)) { BENCH_PRINTF("  Thread %d: block b corrupted after interleaved ops\n", threadIdx.x); passed = false; }

    free_based_on_g_mode(b);
    free_based_on_g_mode(c);

    void* d = alloc_based_on_g_mode(64);
    void* e = alloc_based_on_g_mode(16);

    free_based_on_g_mode(d);
    void* f = alloc_based_on_g_mode(32);

    free_based_on_g_mode(e);
    free_based_on_g_mode(f);

    void* final_ptr = alloc_based_on_g_mode(48);
    if (!final_ptr) passed = false;
    free_based_on_g_mode(final_ptr);

    __syncthreads();
    if (threadIdx.x == 0) BENCH_PRINTF("[TEST 15] Interleaved alloc/free: %s\n", passed ? "PASSED" : "FAILED");
}

__global__ void test_zero_alloc() {
    extern __shared__ std::byte heap[];
    init_based_on_g_mode();

    void* ptr = alloc_based_on_g_mode(0);
    free_based_on_g_mode(ptr);

    void* normal = alloc_based_on_g_mode(32);
    bool passed = (normal != nullptr);
    free_based_on_g_mode(normal);

    __syncthreads();
    if (threadIdx.x == 0) BENCH_PRINTF("[TEST 16] Zero-size allocation: %s\n", passed ? "PASSED" : "FAILED");
}

__global__ void test_thread_isolation() {
    extern __shared__ std::byte heap[];
    init_based_on_g_mode();

    int* my_data = static_cast<int*>(alloc_based_on_g_mode(sizeof(int) * 4));
    if (my_data) for (int i = 0; i < 4; i++) my_data[i] = threadIdx.x * 1000 + i;

    __syncthreads();

    bool passed = true;
    if (my_data) {
        for (int i = 0; i < 4; i++) {
            if (my_data[i] != threadIdx.x * 1000 + i) { BENCH_PRINTF("  Thread %d: data corrupted by another thread\n", threadIdx.x); passed = false; break; }
        }
        free_based_on_g_mode(my_data);
    } else passed = false;

    __syncthreads();

    __shared__ int fail_count;
    if (threadIdx.x == 0) fail_count = 0;
    __syncthreads();

    if (!passed) atomicAdd(&fail_count, 1);
    __syncthreads();

    if (threadIdx.x == 0) BENCH_PRINTF("[TEST 17] Thread isolation: %s\n", fail_count == 0 ? "PASSED" : "FAILED");
}

__global__ void test_repeated_cycles() {
    extern __shared__ std::byte heap[];
    init_based_on_g_mode();

    bool passed = true;
    constexpr int CYCLES = 50;
    for (int cycle = 0; cycle < CYCLES; cycle++) {
        void* ptr = alloc_based_on_g_mode(64);
        if (!ptr) { BENCH_PRINTF("  Thread %d: allocation failed at cycle %d\n", threadIdx.x, cycle); passed = false; break; }
        fill_pattern(ptr, 64, static_cast<uint8_t>(cycle));
        if (!verify_pattern(ptr, 64, static_cast<uint8_t>(cycle))) { BENCH_PRINTF("  Thread %d: pattern mismatch at cycle %d\n", threadIdx.x, cycle); passed = false; free_based_on_g_mode(ptr); break; }
        free_based_on_g_mode(ptr);
    }

    __syncthreads();
    if (threadIdx.x == 0) BENCH_PRINTF("[TEST 18] Repeated alloc/free cycles: %s\n", passed ? "PASSED" : "FAILED");
}

__global__ void test_mixed_sizes() {
    extern __shared__ std::byte heap[];
    init_based_on_g_mode();

    bool passed = true;
    void* ptrs[6]{};
    size_t sizes[] = {8,128,16,64,32,24};

    for (int i = 0; i < 6; i++) {
        ptrs[i] = alloc_based_on_g_mode(sizes[i]);
        if (ptrs[i]) fill_pattern(ptrs[i], sizes[i], static_cast<uint8_t>(i + 100));
    }
    for (int i = 0; i < 6; i++) {
        if (ptrs[i] && !verify_pattern(ptrs[i], sizes[i], static_cast<uint8_t>(i + 100))) {
            BENCH_PRINTF("  Thread %d: mixed size block %d corrupted\n", threadIdx.x, i);
            passed = false;
        }
    }

    free_based_on_g_mode(ptrs[2]);
    free_based_on_g_mode(ptrs[0]);
    free_based_on_g_mode(ptrs[4]);
    free_based_on_g_mode(ptrs[1]);
    free_based_on_g_mode(ptrs[5]);
    free_based_on_g_mode(ptrs[3]);

    __syncthreads();
    if (threadIdx.x == 0) BENCH_PRINTF("[TEST 19] Mixed size allocations: %s\n", passed ? "PASSED" : "FAILED");
}


// ===================== FUZZY TEST copy and pasted =====================
struct TestOperation {
    bool isAlloc;
    unsigned long numBytes;
    int corrospondingAlloc;
};

static constexpr int FUZZ_NUM_THREADS = 32;
static constexpr int OPS_PER_THREAD   = 30;

void generateRandomOperations(TestOperation *ops) {
    for (int i = 0; i < FUZZ_NUM_THREADS; i++) {
        std::unordered_set<size_t> allocationIndices;
        size_t startingInd = (size_t)i * OPS_PER_THREAD;

        for (int j = 0; j < OPS_PER_THREAD; j++) {
            if (allocationIndices.empty()) {
                ops[startingInd + j] = {true, 4UL * sizeof(int), -1};
                allocationIndices.insert((size_t)j);
            } else if (rand() % 2 == 0) {
                ops[startingInd + j] = {true, 4UL * sizeof(int), -1};
                allocationIndices.insert((size_t)j);
            } else {
                ops[startingInd + j] = {false, 0UL, 0};
                auto it = allocationIndices.begin();
                std::advance(it, rand() % allocationIndices.size());
                ops[startingInd + j].corrospondingAlloc = (int)(*it);
                allocationIndices.erase(it);
            }
        }
    }
}

__global__ void runTestsFuzzy(const TestOperation *ops) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    init_based_on_g_mode();
    __syncthreads();

    void *allocatedPtrs[OPS_PER_THREAD]{};

    for (unsigned int i = 0; i < OPS_PER_THREAD; i++) {
        unsigned int opIndex = idx * OPS_PER_THREAD + i;

        if (ops[opIndex].isAlloc) {
            void *ptr = alloc_based_on_g_mode(ops[opIndex].numBytes);
            allocatedPtrs[i] = ptr;

            if (ptr) {
                for (unsigned long j = 0; j < ops[opIndex].numBytes; j++) ((char*)ptr)[j] = 'a';
            }

            BENCH_PRINTF("Thread %d: Allocated %lu bytes at %p\n", idx, ops[opIndex].numBytes, ptr);
        } else {
            int k = ops[opIndex].corrospondingAlloc;
            char *ptrToFree = (char*)allocatedPtrs[k];

            if (ptrToFree) {
                for (unsigned long j = 0; j < ops[idx * OPS_PER_THREAD + k].numBytes; j++) {
                    if (ptrToFree[j] != 'a') {
                        BENCH_PRINTF("Thread %d: failed to free allocation at index %d\n", idx, k);
                        assert(ptrToFree[j] == 'a');
                    }
                }
            }

            free_based_on_g_mode(ptrToFree);
            BENCH_PRINTF("Thread %d: Freed allocation at index %d\n", idx, k);
        }
    }
}


// ===================== EXHAUSTIVE LINEAR ALGEBRA SUITE =====================
//
// Device-side 4x4 float linear algebra, all allocations go through the
// allocator under test.  Each thread runs the full suite independently.
//
// Sub-tests:
//   1. Matrix multiply  A*I = A
//   2. Matrix multiply  A*B cross-check
//   3. LU decomposition (Doolittle, verify L*U = A)
//   4. Cholesky decomposition (verify L*L^T = A on SPD matrix)
//   5. Forward substitution  Lx = b
//   6. Back substitution     Ux = b
//   7. Full LU solve  Ax = b  (via forward + back sub)
//   8. Transpose       (A^T)^T = A
//   9. Matrix-vector multiply  manual check
//  10. Determinant via LU  det(I) = 1
//  11. Trace           tr(I) = dim
//  12. Frobenius norm  ||I||_F = sqrt(dim)
// -----------------------------------------------------------------------

static constexpr int   LDIM = 4;                     // matrix dimension
using lfloat = float;                                 // element type
static constexpr lfloat LEPS = 1e-3f;                // tolerance

static constexpr size_t LMAT_BYTES = LDIM * LDIM * sizeof(lfloat);
static constexpr size_t LVEC_BYTES = LDIM * sizeof(lfloat);

// row-major element access
#define LM(m, r, c) ((m)[(r) * LDIM + (c)])

// ---- tiny helpers (device only) ----

__device__ lfloat* lm_alloc()  { return (lfloat*)alloc_based_on_g_mode(LMAT_BYTES); }
__device__ lfloat* lv_alloc()  { return (lfloat*)alloc_based_on_g_mode(LVEC_BYTES); }
__device__ void    lm_free(lfloat* p) { free_based_on_g_mode(p); }
__device__ void    lv_free(lfloat* p) { free_based_on_g_mode(p); }

__device__ void lm_zero(lfloat* M) {
    for (int i = 0; i < LDIM * LDIM; i++) M[i] = 0.0f;
}
__device__ void lm_identity(lfloat* M) {
    for (int i = 0; i < LDIM; i++)
        for (int j = 0; j < LDIM; j++)
            LM(M, i, j) = (i == j) ? 1.0f : 0.0f;
}
__device__ void lm_copy(lfloat* dst, const lfloat* src) {
    for (int i = 0; i < LDIM * LDIM; i++) dst[i] = src[i];
}
__device__ void lv_copy(lfloat* dst, const lfloat* src) {
    for (int i = 0; i < LDIM; i++) dst[i] = src[i];
}

// deterministic fill seeded per-thread
__device__ void lm_fill(lfloat* M, int seed) {
    for (int i = 0; i < LDIM; i++)
        for (int j = 0; j < LDIM; j++)
            LM(M, i, j) = (lfloat)(((seed * 7 + i * 13 + j * 17 + 3) % 97) - 48) / 24.0f;
}

// make symmetric positive definite: out = A^T*A + dim*I
__device__ void lm_make_spd(lfloat* out, const lfloat* A) {
    for (int i = 0; i < LDIM; i++) {
        for (int j = 0; j < LDIM; j++) {
            lfloat s = 0.0f;
            for (int k = 0; k < LDIM; k++) s += LM(A, k, i) * LM(A, k, j);
            LM(out, i, j) = s;
        }
    }
    for (int i = 0; i < LDIM; i++) LM(out, i, i) += (lfloat)LDIM;
}

// ---- linear algebra kernels ----

// C = A * B
__device__ void lm_mul(lfloat* C, const lfloat* A, const lfloat* B) {
    for (int i = 0; i < LDIM; i++)
        for (int j = 0; j < LDIM; j++) {
            lfloat s = 0.0f;
            for (int k = 0; k < LDIM; k++) s += LM(A, i, k) * LM(B, k, j);
            LM(C, i, j) = s;
        }
}

// LU decomposition (Doolittle, no pivoting).
// L = unit lower triangular, U = upper triangular.
// Returns false if a zero pivot is hit.
__device__ bool lm_lu(const lfloat* A, lfloat* L, lfloat* U) {
    lm_zero(L); lm_zero(U);
    for (int i = 0; i < LDIM; i++) {
        for (int j = i; j < LDIM; j++) {
            lfloat s = 0.0f;
            for (int k = 0; k < i; k++) s += LM(L, i, k) * LM(U, k, j);
            LM(U, i, j) = LM(A, i, j) - s;
        }
        LM(L, i, i) = 1.0f;
        for (int j = i + 1; j < LDIM; j++) {
            lfloat s = 0.0f;
            for (int k = 0; k < i; k++) s += LM(L, j, k) * LM(U, k, i);
            if (fabsf(LM(U, i, i)) < 1e-10f) return false;
            LM(L, j, i) = (LM(A, j, i) - s) / LM(U, i, i);
        }
    }
    return true;
}

// Cholesky: A = L * L^T.  A must be SPD.
__device__ bool lm_cholesky(const lfloat* A, lfloat* L) {
    lm_zero(L);
    for (int i = 0; i < LDIM; i++) {
        for (int j = 0; j <= i; j++) {
            lfloat s = 0.0f;
            for (int k = 0; k < j; k++) s += LM(L, i, k) * LM(L, j, k);
            if (i == j) {
                lfloat v = LM(A, i, i) - s;
                if (v <= 0.0f) return false;
                LM(L, i, j) = sqrtf(v);
            } else {
                if (fabsf(LM(L, j, j)) < 1e-10f) return false;
                LM(L, i, j) = (LM(A, i, j) - s) / LM(L, j, j);
            }
        }
    }
    return true;
}

// B = A^T
__device__ void lm_transpose(lfloat* B, const lfloat* A) {
    for (int i = 0; i < LDIM; i++)
        for (int j = 0; j < LDIM; j++)
            LM(B, i, j) = LM(A, j, i);
}

// y = A * x
__device__ void lm_vec_mul(lfloat* y, const lfloat* A, const lfloat* x) {
    for (int i = 0; i < LDIM; i++) {
        lfloat s = 0.0f;
        for (int j = 0; j < LDIM; j++) s += LM(A, i, j) * x[j];
        y[i] = s;
    }
}

// dot product
__device__ lfloat lv_dot(const lfloat* a, const lfloat* b) {
    lfloat s = 0.0f;
    for (int i = 0; i < LDIM; i++) s += a[i] * b[i];
    return s;
}

// forward substitution: solve Lx = b  (L unit/lower triangular)
__device__ void lm_fwd_sub(lfloat* x, const lfloat* L, const lfloat* b) {
    for (int i = 0; i < LDIM; i++) {
        lfloat s = b[i];
        for (int j = 0; j < i; j++) s -= LM(L, i, j) * x[j];
        x[i] = s / LM(L, i, i);
    }
}

// back substitution: solve Ux = b  (U upper triangular)
__device__ void lm_back_sub(lfloat* x, const lfloat* U, const lfloat* b) {
    for (int i = LDIM - 1; i >= 0; i--) {
        lfloat s = b[i];
        for (int j = i + 1; j < LDIM; j++) s -= LM(U, i, j) * x[j];
        x[i] = s / LM(U, i, i);
    }
}

// trace
__device__ lfloat lm_trace(const lfloat* M) {
    lfloat s = 0.0f;
    for (int i = 0; i < LDIM; i++) s += LM(M, i, i);
    return s;
}

// Frobenius norm
__device__ lfloat lm_frobenius(const lfloat* M) {
    lfloat s = 0.0f;
    for (int i = 0; i < LDIM * LDIM; i++) s += M[i] * M[i];
    return sqrtf(s);
}

// determinant via LU = product of U diagonal
__device__ lfloat lm_det_from_u(const lfloat* U) {
    lfloat d = 1.0f;
    for (int i = 0; i < LDIM; i++) d *= LM(U, i, i);
    return d;
}

// max |A-B|
__device__ lfloat lm_max_diff(const lfloat* A, const lfloat* B) {
    lfloat mx = 0.0f;
    for (int i = 0; i < LDIM * LDIM; i++) {
        lfloat d = fabsf(A[i] - B[i]);
        if (d > mx) mx = d;
    }
    return mx;
}
__device__ lfloat lv_max_diff(const lfloat* a, const lfloat* b) {
    lfloat mx = 0.0f;
    for (int i = 0; i < LDIM; i++) {
        lfloat d = fabsf(a[i] - b[i]);
        if (d > mx) mx = d;
    }
    return mx;
}

// ---- the big kernel ----
// Each sub-test allocates through the pool, computes, verifies, then frees.
// We keep max simultaneous allocations small (~4-5 matrices) so thread_pool
// with ~1.5 KB/thread can handle it (4x4 float matrix = 64 bytes).

__global__ void test_linalg_exhaustive() {
    extern __shared__ std::byte heap[];
    init_based_on_g_mode();

    int tid = threadIdx.x;
    bool passed = true;

    // ---- Sub-test 1: A * I = A ----
    {
        lfloat* A = lm_alloc();
        lfloat* I = lm_alloc();
        lfloat* C = lm_alloc();
        if (!A || !I || !C) { passed = false; }
        else {
            lm_fill(A, tid + 1);
            lm_identity(I);
            lm_mul(C, A, I);
            if (lm_max_diff(A, C) > LEPS) {
                BENCH_PRINTF("  T%d LINALG-1: A*I != A (diff=%.6f)\n", tid, lm_max_diff(A, C));
                passed = false;
            }
        }
        lm_free(C); lm_free(I); lm_free(A);
    }

    // ---- Sub-test 2: A * B, then I * (A*B) = A*B ----
    {
        lfloat* A  = lm_alloc();
        lfloat* B  = lm_alloc();
        lfloat* AB = lm_alloc();
        lfloat* I  = lm_alloc();
        lfloat* C  = lm_alloc();
        if (!A || !B || !AB || !I || !C) { passed = false; }
        else {
            lm_fill(A, tid + 10);
            lm_fill(B, tid + 20);
            lm_mul(AB, A, B);
            lm_identity(I);
            lm_mul(C, I, AB);
            if (lm_max_diff(AB, C) > LEPS) {
                BENCH_PRINTF("  T%d LINALG-2: I*(A*B) != A*B (diff=%.6f)\n", tid, lm_max_diff(AB, C));
                passed = false;
            }
        }
        lm_free(C); lm_free(I); lm_free(AB); lm_free(B); lm_free(A);
    }

    // ---- Sub-test 3: LU decomposition, verify L*U = A ----
    {
        lfloat* A   = lm_alloc();
        lfloat* SPD = lm_alloc();  // use SPD to guarantee non-singular
        lfloat* L   = lm_alloc();
        lfloat* U   = lm_alloc();
        lfloat* LU  = lm_alloc();
        if (!A || !SPD || !L || !U || !LU) { passed = false; }
        else {
            lm_fill(A, tid + 30);
            lm_make_spd(SPD, A);  // SPD is guaranteed non-singular
            if (!lm_lu(SPD, L, U)) {
                BENCH_PRINTF("  T%d LINALG-3: LU failed (singular pivot)\n", tid);
                passed = false;
            } else {
                lm_mul(LU, L, U);
                if (lm_max_diff(SPD, LU) > LEPS) {
                    BENCH_PRINTF("  T%d LINALG-3: L*U != A (diff=%.6f)\n", tid, lm_max_diff(SPD, LU));
                    passed = false;
                }
            }
        }
        lm_free(LU); lm_free(U); lm_free(L); lm_free(SPD); lm_free(A);
    }

    // ---- Sub-test 4: Cholesky, verify L*L^T = A ----
    {
        lfloat* A   = lm_alloc();
        lfloat* SPD = lm_alloc();
        lfloat* L   = lm_alloc();
        lfloat* LT  = lm_alloc();
        lfloat* R   = lm_alloc();  // result L*L^T
        if (!A || !SPD || !L || !LT || !R) { passed = false; }
        else {
            lm_fill(A, tid + 40);
            lm_make_spd(SPD, A);
            if (!lm_cholesky(SPD, L)) {
                BENCH_PRINTF("  T%d LINALG-4: Cholesky failed\n", tid);
                passed = false;
            } else {
                lm_transpose(LT, L);
                lm_mul(R, L, LT);
                if (lm_max_diff(SPD, R) > LEPS) {
                    BENCH_PRINTF("  T%d LINALG-4: L*L^T != A (diff=%.6f)\n", tid, lm_max_diff(SPD, R));
                    passed = false;
                }
            }
        }
        lm_free(R); lm_free(LT); lm_free(L); lm_free(SPD); lm_free(A);
    }

    // ---- Sub-test 5: Forward substitution Lx = b, verify ----
    {
        lfloat* A   = lm_alloc();
        lfloat* SPD = lm_alloc();
        lfloat* L   = lm_alloc();
        lfloat* U   = lm_alloc();
        lfloat* b   = lv_alloc();
        lfloat* x   = lv_alloc();
        lfloat* chk = lv_alloc();
        if (!A || !SPD || !L || !U || !b || !x || !chk) { passed = false; }
        else {
            lm_fill(A, tid + 50);
            lm_make_spd(SPD, A);
            lm_lu(SPD, L, U);
            // set b = [1, 2, 3, 4]
            for (int i = 0; i < LDIM; i++) b[i] = (lfloat)(i + 1);
            lm_fwd_sub(x, L, b);
            // verify: L*x should = b
            lm_vec_mul(chk, L, x);
            if (lv_max_diff(b, chk) > LEPS) {
                BENCH_PRINTF("  T%d LINALG-5: L*x != b (diff=%.6f)\n", tid, lv_max_diff(b, chk));
                passed = false;
            }
        }
        lv_free(chk); lv_free(x); lv_free(b);
        lm_free(U); lm_free(L); lm_free(SPD); lm_free(A);
    }

    // ---- Sub-test 6: Back substitution Ux = b, verify ----
    {
        lfloat* A   = lm_alloc();
        lfloat* SPD = lm_alloc();
        lfloat* L   = lm_alloc();
        lfloat* U   = lm_alloc();
        lfloat* b   = lv_alloc();
        lfloat* x   = lv_alloc();
        lfloat* chk = lv_alloc();
        if (!A || !SPD || !L || !U || !b || !x || !chk) { passed = false; }
        else {
            lm_fill(A, tid + 60);
            lm_make_spd(SPD, A);
            lm_lu(SPD, L, U);
            for (int i = 0; i < LDIM; i++) b[i] = (lfloat)(LDIM - i);
            lm_back_sub(x, U, b);
            lm_vec_mul(chk, U, x);
            if (lv_max_diff(b, chk) > LEPS) {
                BENCH_PRINTF("  T%d LINALG-6: U*x != b (diff=%.6f)\n", tid, lv_max_diff(b, chk));
                passed = false;
            }
        }
        lv_free(chk); lv_free(x); lv_free(b);
        lm_free(U); lm_free(L); lm_free(SPD); lm_free(A);
    }

    // ---- Sub-test 7: Full LU solve  Ax=b  ->  Ly=b, Ux=y, verify Ax=b ----
    {
        lfloat* A   = lm_alloc();
        lfloat* SPD = lm_alloc();
        lfloat* L   = lm_alloc();
        lfloat* U   = lm_alloc();
        lfloat* b   = lv_alloc();
        lfloat* y   = lv_alloc();
        lfloat* x   = lv_alloc();
        lfloat* chk = lv_alloc();
        if (!A || !SPD || !L || !U || !b || !y || !x || !chk) { passed = false; }
        else {
            lm_fill(A, tid + 70);
            lm_make_spd(SPD, A);
            lm_lu(SPD, L, U);
            for (int i = 0; i < LDIM; i++) b[i] = (lfloat)(i * 3 + 1);
            lm_fwd_sub(y, L, b);   // Ly = b
            lm_back_sub(x, U, y);  // Ux = y
            lm_vec_mul(chk, SPD, x); // chk = A*x, should = b
            if (lv_max_diff(b, chk) > LEPS) {
                BENCH_PRINTF("  T%d LINALG-7: A*x != b (diff=%.6f)\n", tid, lv_max_diff(b, chk));
                passed = false;
            }
        }
        lv_free(chk); lv_free(x); lv_free(y); lv_free(b);
        lm_free(U); lm_free(L); lm_free(SPD); lm_free(A);
    }

    // ---- Sub-test 8: Transpose  (A^T)^T = A ----
    {
        lfloat* A  = lm_alloc();
        lfloat* AT = lm_alloc();
        lfloat* R  = lm_alloc();
        if (!A || !AT || !R) { passed = false; }
        else {
            lm_fill(A, tid + 80);
            lm_transpose(AT, A);
            lm_transpose(R, AT);
            if (lm_max_diff(A, R) > LEPS) {
                BENCH_PRINTF("  T%d LINALG-8: (A^T)^T != A\n", tid);
                passed = false;
            }
        }
        lm_free(R); lm_free(AT); lm_free(A);
    }

    // ---- Sub-test 9: Mat-vec multiply manual check ----
    //  A = I, x = [1..dim] -> y should = x
    {
        lfloat* I   = lm_alloc();
        lfloat* x   = lv_alloc();
        lfloat* y   = lv_alloc();
        if (!I || !x || !y) { passed = false; }
        else {
            lm_identity(I);
            for (int i = 0; i < LDIM; i++) x[i] = (lfloat)(i + 1);
            lm_vec_mul(y, I, x);
            if (lv_max_diff(x, y) > LEPS) {
                BENCH_PRINTF("  T%d LINALG-9: I*x != x\n", tid);
                passed = false;
            }
        }
        lv_free(y); lv_free(x); lm_free(I);
    }

    // ---- Sub-test 10: Determinant via LU: det(I) = 1 ----
    {
        lfloat* I = lm_alloc();
        lfloat* L = lm_alloc();
        lfloat* U = lm_alloc();
        if (!I || !L || !U) { passed = false; }
        else {
            lm_identity(I);
            lm_lu(I, L, U);
            lfloat det = lm_det_from_u(U);
            if (fabsf(det - 1.0f) > LEPS) {
                BENCH_PRINTF("  T%d LINALG-10: det(I) = %.6f != 1\n", tid, det);
                passed = false;
            }
        }
        lm_free(U); lm_free(L); lm_free(I);
    }

    // ---- Sub-test 11: Trace  tr(I) = dim ----
    {
        lfloat* I = lm_alloc();
        if (!I) { passed = false; }
        else {
            lm_identity(I);
            lfloat tr = lm_trace(I);
            if (fabsf(tr - (lfloat)LDIM) > LEPS) {
                BENCH_PRINTF("  T%d LINALG-11: tr(I) = %.6f != %d\n", tid, tr, LDIM);
                passed = false;
            }
        }
        lm_free(I);
    }

    // ---- Sub-test 12: Frobenius norm  ||I||_F = sqrt(dim) ----
    {
        lfloat* I = lm_alloc();
        if (!I) { passed = false; }
        else {
            lm_identity(I);
            lfloat nrm = lm_frobenius(I);
            if (fabsf(nrm - sqrtf((lfloat)LDIM)) > LEPS) {
                BENCH_PRINTF("  T%d LINALG-12: ||I||_F = %.6f != %.6f\n", tid, nrm, sqrtf((lfloat)LDIM));
                passed = false;
            }
        }
        lm_free(I);
    }

    // ---- aggregate result across threads ----
    __syncthreads();
    __shared__ int linalg_fail_count;
    if (threadIdx.x == 0) linalg_fail_count = 0;
    __syncthreads();
    if (!passed) atomicAdd(&linalg_fail_count, 1);
    __syncthreads();
    if (threadIdx.x == 0) {
        BENCH_PRINTF("[LINALG] Exhaustive linalg suite: %s (%d/%d threads passed)\n",
            linalg_fail_count == 0 ? "PASSED" : "FAILED",
            NUM_THREADS - linalg_fail_count, NUM_THREADS);
    }
}


// ===================== EXHAUSTIVE CAPACITY TEST (optin smem) =====================
//
// Uses cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize)
// to opt-in for 100% of the device's shared memory, then allocates ~90% of
// each thread's share.  Tests near-capacity behaviour: single huge alloc,
// pack-with-small-allocs, checkerboard free/refill, alloc-until-failure +
// recovery, and repeated near-capacity cycles.
//
// The host sets g_exhaustive_budget before launch so the kernel knows
// exactly how many usable bytes each thread should target.
// -----------------------------------------------------------------------

// Host writes this before launch; kernel reads it.
__device__ __managed__ size_t g_exhaustive_budget = 0;  // per-thread usable bytes

__global__ void test_exhaustive_capacity() {
    extern __shared__ std::byte heap[];
    init_based_on_g_mode();

    int tid = threadIdx.x;
    bool passed = true;

    // Per-thread budget was pre-computed on host; 90% target
    size_t budget   = g_exhaustive_budget;
    size_t target90 = budget * 9 / 10;
    size_t target85 = budget * 85 / 100;
    size_t target80 = budget * 80 / 100;

    // ---- Sub-test 1: single 90% allocation, write + verify ----
    {
        void* big = alloc_based_on_g_mode(target90);
        if (!big) {
            BENCH_PRINTF("  T%d EXHAUST-1: 90%% alloc (%zu B) failed\n", tid, target90);
            passed = false;
        } else {
            uint8_t* p = (uint8_t*)big;
            for (size_t i = 0; i < target90; i++) p[i] = (uint8_t)(i & 0xFF);
            for (size_t i = 0; i < target90; i++) {
                if (p[i] != (uint8_t)(i & 0xFF)) {
                    BENCH_PRINTF("  T%d EXHAUST-1: verify failed at byte %zu\n", tid, i);
                    passed = false; break;
                }
            }
            free_based_on_g_mode(big);
        }
    }

    // ---- Sub-test 2: free + re-allocate same size (recovery) ----
    {
        void* a = alloc_based_on_g_mode(target90);
        free_based_on_g_mode(a);
        void* b = alloc_based_on_g_mode(target90);
        if (!b) {
            BENCH_PRINTF("  T%d EXHAUST-2: re-alloc after free failed\n", tid);
            passed = false;
        }
        free_based_on_g_mode(b);
    }

    // ---- Sub-test 3: many small allocs summing to ~85% ----
    {
        constexpr size_t CHUNK = 32;           // 32-byte chunks
        constexpr int    MAX_CHUNKS = 512;     // upper bound on array size
        void* chunks[MAX_CHUNKS]{};
        size_t total = 0;
        int n = 0;

        while (total + CHUNK <= target85 && n < MAX_CHUNKS) {
            chunks[n] = alloc_based_on_g_mode(CHUNK);
            if (!chunks[n]) break;
            fill_pattern(chunks[n], CHUNK, (uint8_t)(n & 0xFF));
            total += CHUNK;
            n++;
        }

        // verify every chunk still intact
        for (int i = 0; i < n; i++) {
            if (!verify_pattern(chunks[i], CHUNK, (uint8_t)(i & 0xFF))) {
                BENCH_PRINTF("  T%d EXHAUST-3: chunk %d corrupted\n", tid, i);
                passed = false; break;
            }
        }
        for (int i = 0; i < n; i++) free_based_on_g_mode(chunks[i]);
    }

    // ---- Sub-test 4: checkerboard free + refill ----
    {
        constexpr size_t CHUNK = 32;
        constexpr int    MAX_CHUNKS = 512;
        void* chunks[MAX_CHUNKS]{};
        size_t total = 0;
        int n = 0;

        // fill up to 80%
        while (total + CHUNK <= target80 && n < MAX_CHUNKS) {
            chunks[n] = alloc_based_on_g_mode(CHUNK);
            if (!chunks[n]) break;
            fill_pattern(chunks[n], CHUNK, 0xAA);
            total += CHUNK;
            n++;
        }

        // free every other
        for (int i = 0; i < n; i += 2) {
            free_based_on_g_mode(chunks[i]);
            chunks[i] = nullptr;
        }

        // refill the gaps
        for (int i = 0; i < n; i += 2) {
            chunks[i] = alloc_based_on_g_mode(CHUNK);
            if (chunks[i]) fill_pattern(chunks[i], CHUNK, 0xBB);
        }

        // verify survivors kept 0xAA, refills got 0xBB
        for (int i = 0; i < n; i++) {
            if (!chunks[i]) continue;
            uint8_t expect = (i % 2 == 0) ? 0xBB : 0xAA;
            if (!verify_pattern(chunks[i], CHUNK, expect)) {
                BENCH_PRINTF("  T%d EXHAUST-4: chunk %d pattern wrong\n", tid, i);
                passed = false; break;
            }
        }
        for (int i = 0; i < n; i++) free_based_on_g_mode(chunks[i]);
    }

    // ---- Sub-test 5: alloc-until-failure + full recovery ----
    // For bounded pools (shared/global), we expect to hit nullptr.
    // For device malloc (1 GB heap), nullptr never comes, so we cap at
    // the same number of chunks a bounded pool would produce to keep
    // the workload identical for timing comparison.
    {
        constexpr size_t CHUNK = 64;
        constexpr int    MAX_CHUNKS = 512;
        void* chunks[MAX_CHUNKS]{};
        int n = 0;

        bool unbounded = (g_mode == MODE_DEVICE_MALLOC);
        int  chunk_cap = unbounded
                         ? (int)(target85 / CHUNK)   // match bounded pool count
                         : MAX_CHUNKS;
        if (chunk_cap > MAX_CHUNKS) chunk_cap = MAX_CHUNKS;
        if (chunk_cap < 1)         chunk_cap = 1;

        while (n < chunk_cap) {
            void* p = alloc_based_on_g_mode(CHUNK);
            if (!p) break;              // bounded pools will hit this
            fill_pattern(p, CHUNK, (uint8_t)(n & 0xFF));
            chunks[n++] = p;
        }

        if (n == 0) {
            BENCH_PRINTF("  T%d EXHAUST-5: zero allocs succeeded\n", tid);
            passed = false;
        }

        // verify all live allocations
        for (int i = 0; i < n; i++) {
            if (!verify_pattern(chunks[i], CHUNK, (uint8_t)(i & 0xFF))) {
                BENCH_PRINTF("  T%d EXHAUST-5: chunk %d corrupted\n", tid, i);
                passed = false; break;
            }
        }

        // free everything
        for (int i = 0; i < n; i++) free_based_on_g_mode(chunks[i]);

        // recovery: one big alloc should work now.
        // Only meaningful for bounded pools; device malloc always succeeds.
        if (!unbounded) {
            void* recov = alloc_based_on_g_mode(target80);
            if (!recov) {
                BENCH_PRINTF("  T%d EXHAUST-5: recovery alloc failed\n", tid);
                passed = false;
            }
            free_based_on_g_mode(recov);
        }
    }

    // ---- Sub-test 6: repeated near-capacity cycles ----
    {
        constexpr int CYCLES = 20;
        for (int c = 0; c < CYCLES; c++) {
            void* big = alloc_based_on_g_mode(target80);
            if (!big) {
                BENCH_PRINTF("  T%d EXHAUST-6: cycle %d alloc failed\n", tid, c);
                passed = false; break;
            }
            uint8_t* p = (uint8_t*)big;
            for (size_t i = 0; i < target80; i++) p[i] = (uint8_t)(c + i);
            // spot-check a few locations
            bool ok = true;
            for (size_t i = 0; i < target80; i += target80 / 8 + 1) {
                if (p[i] != (uint8_t)(c + i)) { ok = false; break; }
            }
            if (!ok) {
                BENCH_PRINTF("  T%d EXHAUST-6: verify failed cycle %d\n", tid, c);
                passed = false;
                free_based_on_g_mode(big);
                break;
            }
            free_based_on_g_mode(big);
        }
    }

    // ---- Sub-test 7: graduated sizes until failure, verify last success ----
    // Device malloc will succeed at all percentages (unbounded heap).
    // That's fine — this test finds the allocator's ceiling for bounded
    // pools and still exercises the same alloc/free pattern for timing.
    {
        void* last_ok = nullptr;
        size_t last_sz = 0;
        // try 50%, 60%, 70%, 80%, 85%, 90%, 95%, 98% of budget
        size_t pcts[] = {50, 60, 70, 80, 85, 90, 95, 98};
        for (size_t pct : pcts) {
            size_t sz = budget * pct / 100;
            if (sz == 0) continue;
            void* p = alloc_based_on_g_mode(sz);
            if (!p) break;
            // write first + last byte to confirm writable
            ((uint8_t*)p)[0]      = 0xDE;
            ((uint8_t*)p)[sz - 1] = 0xAD;
            free_based_on_g_mode(p);
            last_sz = sz;
        }
        if (last_sz == 0) {
            BENCH_PRINTF("  T%d EXHAUST-7: no graduated alloc succeeded\n", tid);
            passed = false;
        } else {
            // re-allocate at the last successful size
            last_ok = alloc_based_on_g_mode(last_sz);
            if (!last_ok) {
                BENCH_PRINTF("  T%d EXHAUST-7: re-alloc at %zu failed\n", tid, last_sz);
                passed = false;
            } else {
                if (!verify_memory_writable(last_ok, last_sz)) {
                    BENCH_PRINTF("  T%d EXHAUST-7: full verify at %zu failed\n", tid, last_sz);
                    passed = false;
                }
                free_based_on_g_mode(last_ok);
            }
        }
    }

    // ---- aggregate ----
    __syncthreads();
    __shared__ int exhaust_fail;
    if (threadIdx.x == 0) exhaust_fail = 0;
    __syncthreads();
    if (!passed) atomicAdd(&exhaust_fail, 1);
    __syncthreads();
    if (threadIdx.x == 0) {
        BENCH_PRINTF("[EXHAUST] Exhaustive capacity suite: %s (%d/%d passed)\n",
            exhaust_fail == 0 ? "PASSED" : "FAILED",
            NUM_THREADS - exhaust_fail, NUM_THREADS);
    }
}


// ===================== SUITE RUNNERS (host) =====================
const char* mode_name(int m) {
    switch (m) {
        case MODE_THREAD_FIRST_FIT: return "thread_pool (first-fit)";
        case MODE_THREAD_BEST_FIT:  return "thread_pool (best-fit)";
        case MODE_FREELIST:         return "julian (freelist)";
        // case MODE_BST:              return "julian (BST best-fit)";
        case MODE_DEVICE_MALLOC:    return "device malloc/free";
        case MODE_WARP_FIRST_FIT:   return "warp_pool (first-fit)";
        case MODE_WARP_BEST_FIT:    return "warp_pool (best-fit)";
        case MODE_GLOBAL_FIRST_FIT: return "glob_pool (first-fit)";
        case MODE_GLOBAL_BEST_FIT:  return "glob_pool (best-fit)";
        default:                    return "unknown";
    }
}

const char* mode_name_csv(int m) {
    switch (m) {
        case MODE_THREAD_FIRST_FIT: return "thread_first_fit";
        case MODE_THREAD_BEST_FIT:  return "thread_best_fit";
        case MODE_FREELIST:         return "freelist";
        // case MODE_BST:              return "bst";
        case MODE_DEVICE_MALLOC:    return "device_malloc";
        case MODE_WARP_FIRST_FIT:   return "warp_first_fit";
        case MODE_WARP_BEST_FIT:    return "warp_best_fit";
        case MODE_GLOBAL_FIRST_FIT: return "global_first_fit";
        case MODE_GLOBAL_BEST_FIT:  return "global_best_fit";
        default:                    return "unknown";
    }
}

// Helper: is this a global-memory mode? (host side)
static bool is_global_mode_host(int m) {
    return m == MODE_GLOBAL_FIRST_FIT || m == MODE_GLOBAL_BEST_FIT;
}

float time_allocator_suite_once(size_t dynBytes) {
    g_dyn_smem_bytes = dynBytes;

    // For global modes, set the heap size managed var
    if (is_global_mode_host(g_mode)) {
        g_global_heap_bytes = GLOBAL_HEAP_ALLOC_TEST;
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // Global modes need 0 shared mem; others need dynBytes
    size_t smem = is_global_mode_host(g_mode) ? 0 : dynBytes;

    cudaEvent_t s,e;
    CUDA_CHECK(cudaEventCreate(&s));
    CUDA_CHECK(cudaEventCreate(&e));

    CUDA_CHECK(cudaEventRecord(s));

    test_init<<<1, NUM_THREADS, smem>>>();
    test_single_alloc_free<<<1, NUM_THREADS, smem>>>();
    test_multiple_allocs<<<1, NUM_THREADS, smem>>>();
    test_coalescing<<<1, NUM_THREADS, smem>>>();
    test_coalescing_orders<<<1, NUM_THREADS, smem>>>();
    test_splitting<<<1, NUM_THREADS, smem>>>();
    test_size_boundaries<<<1, NUM_THREADS, smem>>>();
    //test_exhaustion<<<1, NUM_THREADS, smem>>>();
    test_fragmentation<<<1, NUM_THREADS, smem>>>();
    test_free_list_order<<<1, NUM_THREADS, smem>>>();
    test_null_free<<<1, NUM_THREADS, smem>>>();
    test_double_free<<<1, NUM_THREADS, smem>>>();
    test_max_allocation<<<1, NUM_THREADS, smem>>>();
    test_alignment<<<1, NUM_THREADS, smem>>>();
    test_interleaved<<<1, NUM_THREADS, smem>>>();
    test_zero_alloc<<<1, NUM_THREADS, smem>>>();
    test_thread_isolation<<<1, NUM_THREADS, smem>>>();
    test_repeated_cycles<<<1, NUM_THREADS, smem>>>();
    test_mixed_sizes<<<1, NUM_THREADS, smem>>>();
    //test_best_fit<<<1, NUM_THREADS, smem>>>();

    CUDA_CHECK(cudaEventRecord(e));
    CUDA_CHECK(cudaEventSynchronize(e));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
    CUDA_CHECK(cudaEventDestroy(s));
    CUDA_CHECK(cudaEventDestroy(e));
    return ms;
}

float time_fuzzy_once(const TestOperation* d_ops, size_t dynBytes) {
    g_dyn_smem_bytes = dynBytes;

    if (is_global_mode_host(g_mode)) {
        g_global_heap_bytes = GLOBAL_HEAP_FUZZY;
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    size_t smem = is_global_mode_host(g_mode) ? 0 : dynBytes;

    cudaEvent_t s,e;
    CUDA_CHECK(cudaEventCreate(&s));
    CUDA_CHECK(cudaEventCreate(&e));

    CUDA_CHECK(cudaEventRecord(s));
    runTestsFuzzy<<<1, FUZZ_NUM_THREADS, smem>>>(d_ops);
    CUDA_CHECK(cudaEventRecord(e));

    CUDA_CHECK(cudaEventSynchronize(e));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
    CUDA_CHECK(cudaEventDestroy(s));
    CUDA_CHECK(cudaEventDestroy(e));
    return ms;
}

// Linalg suite shared-mem / global-heap sizes.  Same pool as alloc_test
// so the comparison is apples-to-apples.
static constexpr size_t SHARED_MEM_LINALG = SHARED_MEM_ALLOC_TEST;
static constexpr size_t GLOBAL_HEAP_LINALG = GLOBAL_HEAP_ALLOC_TEST;

float time_linalg_once(size_t dynBytes) {
    g_dyn_smem_bytes = dynBytes;

    if (is_global_mode_host(g_mode)) {
        g_global_heap_bytes = GLOBAL_HEAP_LINALG;
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    size_t smem = is_global_mode_host(g_mode) ? 0 : dynBytes;

    cudaEvent_t s,e;
    CUDA_CHECK(cudaEventCreate(&s));
    CUDA_CHECK(cudaEventCreate(&e));

    CUDA_CHECK(cudaEventRecord(s));
    test_linalg_exhaustive<<<1, NUM_THREADS, smem>>>();
    CUDA_CHECK(cudaEventRecord(e));

    CUDA_CHECK(cudaEventSynchronize(e));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
    CUDA_CHECK(cudaEventDestroy(s));
    CUDA_CHECK(cudaEventDestroy(e));
    return ms;
}

// Exhaustive capacity test — uses the full optin shared memory.
// `optin_bytes` is sharedMemPerBlockOptin - 16 (CUDA internal overhead).
// `budget_per_thread` is pre-computed on the host and written to the managed
// variable g_exhaustive_budget before launch.
// NOTE: never called for MODE_DEVICE_MALLOC (skipped in main loop).
float time_exhaustive_once(size_t optin_bytes, size_t budget_per_thread,
                           size_t global_heap_exhaustive) {
    g_dyn_smem_bytes     = optin_bytes;
    g_exhaustive_budget  = budget_per_thread;

    if (is_global_mode_host(g_mode)) {
        g_global_heap_bytes = global_heap_exhaustive;
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    size_t smem = is_global_mode_host(g_mode) ? 0 : optin_bytes;

    // Opt-in: tell the runtime this kernel needs more than the default 48 KB.
    // Global modes use 0 smem so the attribute is irrelevant for them.
    if (!is_global_mode_host(g_mode)) {
        CUDA_CHECK(cudaFuncSetAttribute(
            test_exhaustive_capacity,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)optin_bytes));
    }

    cudaEvent_t s,e;
    CUDA_CHECK(cudaEventCreate(&s));
    CUDA_CHECK(cudaEventCreate(&e));

    CUDA_CHECK(cudaEventRecord(s));
    test_exhaustive_capacity<<<1, NUM_THREADS, smem>>>();
    CUDA_CHECK(cudaEventRecord(e));

    CUDA_CHECK(cudaEventSynchronize(e));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
    CUDA_CHECK(cudaEventDestroy(s));
    CUDA_CHECK(cudaEventDestroy(e));
    return ms;
}

// ===================== MAIN =====================
int main(int argc, char** argv) {
    int seed = 0;
    int iters = 100;
    int sample_interval = 5;
    const char* csv_path = "benchmark_results.csv";

    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            seed = std::atoi(argv[++i]);
        } else if ((std::strcmp(argv[i], "-i") == 0 || std::strcmp(argv[i], "--iters") == 0) && i + 1 < argc) {
            iters = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--interval") == 0 && i + 1 < argc) {
            sample_interval = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--csv") == 0 && i + 1 < argc) {
            csv_path = argv[++i];
        }
    }
    std::srand(seed);

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    // ── Exhaustive test parameters (runtime, device-dependent) ──
    // Use the optin shared memory for the exhaustive test.
    // Subtract 16 bytes for CUDA internal overhead (kernel args / driver);
    // requesting the full sharedMemPerBlockOptin causes "invalid argument"
    // on some architectures (confirmed on Ada Lovelace / RTX 4090).
    // Reserve a conservative 128 bytes per thread for allocator metadata.
    const size_t optin_bytes = (size_t)prop.sharedMemPerBlockOptin - 16;
    const size_t overhead_per_thread  = 128;  // header / alignment overhead estimate
    const size_t usable_per_thread    = (optin_bytes / NUM_THREADS) - overhead_per_thread;
    const size_t global_heap_exhaust  = optin_bytes;  // match optin size for global modes

    printf("============================================\n");
    printf("  benchmark_all.cu  (runtime switch, inc. global mem)\n");
    printf("============================================\n");
    printf("Device: %s\n", prop.name);
    printf("sharedMemPerBlock:       %zu bytes\n", (size_t)prop.sharedMemPerBlock);
    printf("sharedMemPerBlockOptin:  %zu bytes\n\n", (size_t)prop.sharedMemPerBlockOptin);
    printf("allocator_test dyn smem: %zu bytes\n", (size_t)SHARED_MEM_ALLOC_TEST);
    printf("fuzzy_test dyn smem:     %zu bytes\n", (size_t)SHARED_MEM_FUZZY);
    printf("linalg_test dyn smem:    %zu bytes\n", (size_t)SHARED_MEM_LINALG);
    printf("exhaustive dyn smem:     %zu bytes (optin 100%%)\n", optin_bytes);
    printf("  per-thread budget:     %zu bytes (after %zu B overhead)\n", usable_per_thread, overhead_per_thread);
    printf("global heap (suite):     %zu bytes\n", (size_t)GLOBAL_HEAP_ALLOC_TEST);
    printf("global heap (fuzzy):     %zu bytes\n", (size_t)GLOBAL_HEAP_FUZZY);
    printf("global heap (linalg):    %zu bytes\n", (size_t)GLOBAL_HEAP_LINALG);
    printf("global heap (exhaust):   %zu bytes\n", global_heap_exhaust);
    printf("linalg dim:              %d x %d  (%s)\n", LDIM, LDIM, "float");
    printf("BENCH_VERBOSE:           %d (0 recommended)\n\n", (int)BENCH_VERBOSE);

    // For device malloc/free -> give it a heap 1GB
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1ULL * 1024ULL * 1024ULL * 1024ULL));

    // Pre-allocate global-memory heap (reused across global mode runs)
    // Must be large enough for the biggest user — the exhaustive test at optin size.
    size_t max_global = GLOBAL_HEAP_ALLOC_TEST;
    if (GLOBAL_HEAP_FUZZY   > max_global) max_global = GLOBAL_HEAP_FUZZY;
    if (GLOBAL_HEAP_LINALG  > max_global) max_global = GLOBAL_HEAP_LINALG;
    if (global_heap_exhaust > max_global) max_global = global_heap_exhaust;

    std::byte* d_global_heap = nullptr;
    CUDA_CHECK(cudaMalloc(&d_global_heap, max_global));
    g_global_heap = d_global_heap;

    // Prepare fuzzy ops on device (same generation)
    TestOperation h_ops[FUZZ_NUM_THREADS * OPS_PER_THREAD];
    generateRandomOperations(h_ops);

    TestOperation* d_ops = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ops, sizeof(h_ops)));
    CUDA_CHECK(cudaMemcpy(d_ops, h_ops, sizeof(h_ops), cudaMemcpyHostToDevice));

    FILE* csv = std::fopen(csv_path, "w");
    if (!csv) { printf("ERROR: Could not open %s\n", csv_path); return 1; }
    std::fprintf(csv, "allocator,test_type,iteration,cumulative_ms,avg_ms_per_iter\n");

    int run_idx = -1;
    for(int mode : {
        // ── Round 1 ──────────────────────────
        MODE_THREAD_FIRST_FIT,   // idx 0
        MODE_THREAD_BEST_FIT,    // idx 1
        MODE_WARP_FIRST_FIT,     // idx 2  — 1t/pool
        MODE_WARP_BEST_FIT,      // idx 3  — 1t/pool
        MODE_WARP_FIRST_FIT,     // idx 4  — 8t/pool
        MODE_WARP_BEST_FIT,      // idx 5  — 8t/pool
        //MODE_WARP_FIRST_FIT,     // idx 6  — 32t/pool
        //MODE_WARP_BEST_FIT,      // idx 7  — 32t/pool
        MODE_FREELIST,           // idx 8
        MODE_GLOBAL_FIRST_FIT,   // idx 9
        MODE_GLOBAL_BEST_FIT,    // idx 10
        MODE_DEVICE_MALLOC,      // idx 11
        // ── Round 2 ──────────────────────────
        MODE_THREAD_FIRST_FIT,   // idx 12
        MODE_THREAD_BEST_FIT,    // idx 13
        MODE_WARP_FIRST_FIT,     // idx 14 — 1t/pool  12
        MODE_WARP_BEST_FIT,      // idx 15 — 1t/pool  13
        MODE_WARP_FIRST_FIT,     // idx 16 — 8t/pool  14
        MODE_WARP_BEST_FIT,      // idx 17 — 8t/pool  15
        //MODE_WARP_FIRST_FIT,     // idx 18 — 32t/pool 16
       // MODE_WARP_BEST_FIT,      // idx 19 — 32t/pool 17
        MODE_FREELIST,           // idx 20
        MODE_GLOBAL_FIRST_FIT,   // idx 21
        MODE_GLOBAL_BEST_FIT,    // idx 22
        MODE_DEVICE_MALLOC       // idx 23
    }){
        run_idx++;

        // HARDCODED SWITCH TO N THREADS PER POOL FOR WARP POOL
        if (run_idx == 2  || run_idx == 12) g_threads_per_pool = 1;
        if (run_idx == 4  || run_idx == 14) g_threads_per_pool = 8;
       // if (run_idx == 6  || run_idx == 18) g_threads_per_pool = 32;

        g_mode = mode;
        const char* name_csv = mode_name_csv(mode);
        CUDA_CHECK(cudaDeviceSynchronize());

        printf("-------------------------------------------------\n");
        printf("MODE: %s\n", mode_name(mode));

        // ── Allocator suite ──────────────────
        (void)time_allocator_suite_once(SHARED_MEM_ALLOC_TEST);  // warmup

        float cumulative_suite = 0.0f;
        for (int i = 1; i <= iters; i++) {
            cumulative_suite += time_allocator_suite_once(SHARED_MEM_ALLOC_TEST);
            if (i % sample_interval == 0) {
                std::fprintf(csv, "%s,suite,%d,%.4f,%.6f\n",
                    name_csv, i, cumulative_suite, cumulative_suite / i);
            }
        }
        if (iters % sample_interval != 0) {
            std::fprintf(csv, "%s,suite,%d,%.4f,%.6f\n",
                name_csv, iters, cumulative_suite, cumulative_suite / iters);
        }
        printf("allocator_test suite: total %.3f ms, avg %.3f ms/iter\n",
            cumulative_suite, cumulative_suite / iters);

        // ── Fuzzy test ───────────────────────
        (void)time_fuzzy_once(d_ops, SHARED_MEM_FUZZY);  // warmup

        float cumulative_fuzzy = 0.0f;
        for (int i = 1; i <= iters; i++) {
            cumulative_fuzzy += time_fuzzy_once(d_ops, SHARED_MEM_FUZZY);
            if (i % sample_interval == 0) {
                std::fprintf(csv, "%s,fuzzy,%d,%.4f,%.6f\n",
                    name_csv, i, cumulative_fuzzy, cumulative_fuzzy / i);
            }
        }
        if (iters % sample_interval != 0) {
            std::fprintf(csv, "%s,fuzzy,%d,%.4f,%.6f\n",
                name_csv, iters, cumulative_fuzzy, cumulative_fuzzy / iters);
        }
        printf("fuzzy_test kernel:    total %.3f ms, avg %.3f ms/iter\n",
            cumulative_fuzzy, cumulative_fuzzy / iters);

        // ── Linalg suite ─────────────────────
        (void)time_linalg_once(SHARED_MEM_LINALG);  // warmup

        float cumulative_linalg = 0.0f;
        for (int i = 1; i <= iters; i++) {
            cumulative_linalg += time_linalg_once(SHARED_MEM_LINALG);
            if (i % sample_interval == 0) {
                std::fprintf(csv, "%s,linalg,%d,%.4f,%.6f\n",
                    name_csv, i, cumulative_linalg, cumulative_linalg / i);
            }
        }
        if (iters % sample_interval != 0) {
            std::fprintf(csv, "%s,linalg,%d,%.4f,%.6f\n",
                name_csv, iters, cumulative_linalg, cumulative_linalg / iters);
        }
        printf("linalg_test suite:    total %.3f ms, avg %.3f ms/iter\n",
            cumulative_linalg, cumulative_linalg / iters);

        // ── Exhaustive capacity (optin smem) ──
        // Skipped for device malloc: the 1 GB device heap makes pool-
        // exhaustion tests meaningless and the budget is artificial.
        if (mode != MODE_DEVICE_MALLOC) {
            (void)time_exhaustive_once(optin_bytes, usable_per_thread,
                                       global_heap_exhaust);  // warmup

            float cumulative_exhaust = 0.0f;
            for (int i = 1; i <= iters; i++) {
                cumulative_exhaust += time_exhaustive_once(optin_bytes,
                    usable_per_thread, global_heap_exhaust);
                if (i % sample_interval == 0) {
                    std::fprintf(csv, "%s,exhaustive,%d,%.4f,%.6f\n",
                        name_csv, i, cumulative_exhaust, cumulative_exhaust / i);
                }
            }
            if (iters % sample_interval != 0) {
                std::fprintf(csv, "%s,exhaustive,%d,%.4f,%.6f\n",
                    name_csv, iters, cumulative_exhaust, cumulative_exhaust / iters);
            }
            printf("exhaustive_test:      total %.3f ms, avg %.3f ms/iter\n",
                cumulative_exhaust, cumulative_exhaust / iters);
        } else {
            printf("exhaustive_test:      SKIPPED (device malloc)\n");
        }
    }

    std::fclose(csv);
    printf("CSV written to: %s\n", csv_path);

    CUDA_CHECK(cudaFree(d_ops));
    if (d_global_heap) CUDA_CHECK(cudaFree(d_global_heap));
    printf("-------------------------------------------------\n");
    printf("Done.\n");
    return 0;
}
