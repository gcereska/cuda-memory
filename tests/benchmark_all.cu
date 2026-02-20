//benchmark_all.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cassert>
#include <unordered_set>
#include <cstring>

#include "allocator.cuh"
#include "poolAlloc.cuh"
#include "poolAllocBST.cuh"

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
    MODE_BST             =3,           // julian BST
    MODE_DEVICE_MALLOC   =4, // native cuda
    MODE_WARP_FIRST_FIT  =5,  
    MODE_WARP_BEST_FIT   =6,  
};

// Managed globals on unified memory so kernels can stay the same signature 
__device__ __managed__ int    g_mode = MODE_THREAD_FIRST_FIT; // mode of the kernel call
__device__ __managed__ size_t g_dyn_smem_bytes = 0; // number of bytes requested for a kernel call needed for pool_init
// we run the same kernel calls but change these global (unified) variables to switch which allocator to use


__device__ __forceinline__ void init_based_on_g_mode() {
    switch (g_mode) {
        case MODE_THREAD_FIRST_FIT:
        case MODE_THREAD_BEST_FIT:
            thread_pool::pool_init(g_dyn_smem_bytes);
            break;
        case MODE_FREELIST:
            pmalloc_freelist::init_gpu_buffer(g_dyn_smem_bytes);
            break;
        case MODE_BST:
            pmalloc_bst::init_gpu_buffer(g_dyn_smem_bytes);
            break;
        case MODE_WARP_FIRST_FIT:
            //warp_pool::pool_init(g_dyn_smem_bytes, g_threads_per_pool);
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
        case MODE_FREELIST:          return pmalloc_freelist::cmalloc(n);
        case MODE_BST:               return pmalloc_bst::cmalloc(n);
        case MODE_DEVICE_MALLOC:     return malloc(n);
        default:                     return nullptr;
    }
}

__device__ __forceinline__ void free_based_on_g_mode(void* p) {
    if (!p) return;
    switch (g_mode) {
        case MODE_THREAD_FIRST_FIT:
        case MODE_THREAD_BEST_FIT:
            thread_pool::pfree(p);
            break;
        case MODE_FREELIST:
            pmalloc_freelist::cfree(p);
            break;
        case MODE_BST:
            pmalloc_bst::cfree(p);
            break;
        case MODE_DEVICE_MALLOC:
            free(p);
            break;
    }
}


static constexpr int    NUM_THREADS = 32;
static constexpr size_t SHARED_MEM_ALLOC_TEST = (48 * 1024) - 16;
static constexpr size_t SHARED_MEM_FUZZY = (32 * 1024);

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
    size_t approx_pool_size = (g_dyn_smem_bytes - 512) / 32;

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

     //

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

            // if ptr is nullptr then failed to allocate
            if (ptr) {
                for (unsigned long j = 0; j < ops[opIndex].numBytes; j++) ((char*)ptr)[j] = 'a';
            }

            BENCH_PRINTF("Thread %d: Allocated %lu bytes at %p\n", idx, ops[opIndex].numBytes, ptr);
        } else {
            int k = ops[opIndex].corrospondingAlloc;
            char *ptrToFree = (char*)allocatedPtrs[k];

            // if ptr is nullptr then failed to allocate
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

// ===================== SUITE RUNNERS (host) =====================
const char* mode_name(int m) {
    switch (m) {
        case MODE_THREAD_FIRST_FIT: return "thread_pool (first-fit)";
        case MODE_THREAD_BEST_FIT:  return "thread_pool (best-fit)";
        case MODE_FREELIST:         return "julian (freelist)";
        case MODE_BST:              return "julian (BST best-fit)";
        case MODE_DEVICE_MALLOC:    return "device malloc/free";
        // case MODE_WARP_FIRST_FIT    return "warp_pool (first-fit)"
        // case MODE_WARP_BEST_FIT     return "warp_pool (best-fit)"
        default:                    return "unknown";
    }
}

float time_allocator_suite_once(size_t dynBytes) {
    g_dyn_smem_bytes = dynBytes;

    CUDA_CHECK(cudaDeviceSynchronize()); 

    cudaEvent_t s,e;
    CUDA_CHECK(cudaEventCreate(&s));
    CUDA_CHECK(cudaEventCreate(&e));

    CUDA_CHECK(cudaEventRecord(s));

    test_init<<<1, NUM_THREADS, dynBytes>>>();
    test_single_alloc_free<<<1, NUM_THREADS, dynBytes>>>();
    test_multiple_allocs<<<1, NUM_THREADS, dynBytes>>>();
    test_coalescing<<<1, NUM_THREADS, dynBytes>>>();
    test_coalescing_orders<<<1, NUM_THREADS, dynBytes>>>();
    test_splitting<<<1, NUM_THREADS, dynBytes>>>();
    test_size_boundaries<<<1, NUM_THREADS, dynBytes>>>();
    test_exhaustion<<<1, NUM_THREADS, dynBytes>>>();
    test_fragmentation<<<1, NUM_THREADS, dynBytes>>>();
    test_free_list_order<<<1, NUM_THREADS, dynBytes>>>();
    test_null_free<<<1, NUM_THREADS, dynBytes>>>();
    test_double_free<<<1, NUM_THREADS, dynBytes>>>();
    test_max_allocation<<<1, NUM_THREADS, dynBytes>>>();
    test_alignment<<<1, NUM_THREADS, dynBytes>>>();
    test_interleaved<<<1, NUM_THREADS, dynBytes>>>();
    test_zero_alloc<<<1, NUM_THREADS, dynBytes>>>();
    test_thread_isolation<<<1, NUM_THREADS, dynBytes>>>();
    test_repeated_cycles<<<1, NUM_THREADS, dynBytes>>>();
    test_mixed_sizes<<<1, NUM_THREADS, dynBytes>>>();
    //test_best_fit<<<1, NUM_THREADS, dynBytes>>>();

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
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t s,e;
    CUDA_CHECK(cudaEventCreate(&s));
    CUDA_CHECK(cudaEventCreate(&e));

    CUDA_CHECK(cudaEventRecord(s));
    runTestsFuzzy<<<1, FUZZ_NUM_THREADS, dynBytes>>>(d_ops);
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
    // Seed control for fuzzy test
    int seed = 0;
    int iters  = 1000; 



    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            seed = std::atoi(argv[++i]);
        } else if ((std::strcmp(argv[i], "-i") == 0 || std::strcmp(argv[i], "--iters") == 0) && i + 1 < argc) {
            iters = std::atoi(argv[++i]);
        }
    }
    std::srand(seed);

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("============================================\n");
    printf("  benchmark_all.cu  (runtime switch, 3 modes)\n");
    printf("============================================\n");
    printf("Device: %s\n", prop.name);
    printf("sharedMemPerBlock:       %zu bytes\n", (size_t)prop.sharedMemPerBlock);
    printf("sharedMemPerBlockOptin:  %zu bytes\n\n", (size_t)prop.sharedMemPerBlockOptin);
    printf("allocator_test dyn smem: %zu bytes\n", (size_t)SHARED_MEM_ALLOC_TEST);
    printf("fuzzy_test dyn smem:     %zu bytes\n", (size_t)SHARED_MEM_FUZZY);
    printf("BENCH_VERBOSE:           %d (0 recommended)\n\n", (int)BENCH_VERBOSE);

    // For device malloc/free -. give it a heap 256MB
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 256ULL * 1024ULL * 1024ULL));

    // Prepare fuzzy ops on device (same generation)
    TestOperation h_ops[FUZZ_NUM_THREADS * OPS_PER_THREAD];
    generateRandomOperations(h_ops);

    TestOperation* d_ops = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ops, sizeof(h_ops)));
    CUDA_CHECK(cudaMemcpy(d_ops, h_ops, sizeof(h_ops), cudaMemcpyHostToDevice));

    // Benchmark config

    //for (int mode : {MODE_THREAD_FIRST_FIT, MODE_TL_BEST_FIT, MODE_DEVICE_MALLOC}) {
    for(int mode : {
        MODE_THREAD_FIRST_FIT, 
        MODE_THREAD_FIRST_FIT, 
        //MODE_THREAD_BEST_FIT,
        //MODE_DEVICE_MALLOC,
        MODE_FREELIST,
        MODE_BST,
        MODE_DEVICE_MALLOC,
    }){
        g_mode = mode;
        CUDA_CHECK(cudaDeviceSynchronize());

        printf("-------------------------------------------------\n");
        printf("MODE: %s\n", mode_name(mode));

        // First kernel launch has more overhead so we need a warmup
        (void)time_allocator_suite_once(SHARED_MEM_ALLOC_TEST);

        // Time allocator suite
        float ms_suite = 0.0f;
        for (int i = 0; i < iters; i++) ms_suite += time_allocator_suite_once(SHARED_MEM_ALLOC_TEST);
        printf("allocator_test suite: total %.3f ms, avg %.3f ms/iter\n",
               ms_suite, ms_suite / iters);

        // Warmup 
        (void)time_fuzzy_once(d_ops, SHARED_MEM_FUZZY);

        // Time fuzzy
        float ms_fuzzy = 0.0f;
        for (int i = 0; i < iters; i++) ms_fuzzy += time_fuzzy_once(d_ops, SHARED_MEM_FUZZY);
        printf("fuzzy_test kernel:    total %.3f ms, avg %.3f ms/iter\n",
               ms_fuzzy, ms_fuzzy / iters);
    }

    CUDA_CHECK(cudaFree(d_ops));
    printf("-------------------------------------------------\n");
    printf("Done.\n");
    return 0;
}
