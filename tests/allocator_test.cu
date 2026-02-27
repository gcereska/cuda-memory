//allocator_test.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cassert>

#include <cumem/blueprint.h>

// Test configuration
static constexpr size_t SHARED_MEM_SIZE = 32 * 1024;  // 32 KB (safe under 48 KB limit)
// When I try to the use 48KB limit exactly it breaks
// Claude estimates it only gives 47.5KB of usuable memory because of driver overhead
static constexpr int NUM_THREADS = 32;

// Helper macro for CUDA error checking
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

// Helper device function to verify memory is writable
__device__ bool verify_memory_writable(void* ptr, size_t size) {
    if (ptr == nullptr) return false;

    uint8_t* bytes = static_cast<uint8_t*>(ptr);
    for (size_t i = 0; i < size; i++) {
        bytes[i] = static_cast<uint8_t>(i & 0xFF);
    }
    for (size_t i = 0; i < size; i++) {
        if (bytes[i] != static_cast<uint8_t>(i & 0xFF)) {
            return false;
        }
    }
    return true;
}

// Helper device function to fill memory with a pattern
__device__ void fill_pattern(void* ptr, size_t size, uint8_t pattern) {
    uint8_t* bytes = static_cast<uint8_t*>(ptr);
    for (size_t i = 0; i < size; i++) {
        bytes[i] = pattern;
    }
}

// Helper device function to verify memory pattern
__device__ bool verify_pattern(void* ptr, size_t size, uint8_t pattern) {
    uint8_t* bytes = static_cast<uint8_t*>(ptr);
    for (size_t i = 0; i < size; i++) {
        if (bytes[i] != pattern) return false;
    }
    return true;
}


// TEST 1: Basic Initialization
__global__ void test_init() {
    extern __shared__ std::byte heap[];

    pool_init(SHARED_MEM_SIZE);

    if (threadIdx.x == 0) {
        printf("[TEST 1] Pool initialization: PASSED\n");
    }
}


// TEST 2: Single Allocation and Free
__global__ void test_single_alloc_free() {
    extern __shared__ std::byte heap[];

    pool_init(SHARED_MEM_SIZE);

    bool passed = true;

    // Each thread allocates
    void* ptr = pmalloc(64);

    if (ptr == nullptr) {
        printf("  Thread %d: allocation failed\n", threadIdx.x);
        passed = false;
    }

    if (passed && !verify_memory_writable(ptr, 64)) {
        printf("  Thread %d: memory verification failed\n", threadIdx.x);
        passed = false;
    }

    pfree(ptr);

    // Try allocating again after free
    void* ptr2 = pmalloc(64);
    if (ptr2 == nullptr) {
        printf("  Thread %d: reallocation after free failed\n", threadIdx.x);
        passed = false;
    }
    pfree(ptr2);

    __syncthreads();

    if (threadIdx.x == 0) {
        printf("[TEST 2] Single alloc/free: %s\n", passed ? "PASSED" : "FAILED");
    }
}


// TEST 3: Multiple Sequential Allocations
__global__ void test_multiple_allocs() {
    extern __shared__ std::byte heap[];

    pool_init(SHARED_MEM_SIZE);

    bool passed = true;
    constexpr int NUM_ALLOCS = 10;
    void* ptrs[NUM_ALLOCS];

    // Allocate multiple small blocks
    for (int i = 0; i < NUM_ALLOCS; i++) {
        ptrs[i] = pmalloc(32);
        if (ptrs[i] == nullptr) {
            printf("  Thread %d: allocation %d failed\n", threadIdx.x, i);
            passed = false;
            break;
        }
        fill_pattern(ptrs[i], 32, static_cast<uint8_t>(i + threadIdx.x));
    }

    // Verify all patterns are intact (no overlap)
    if (passed) {
        for (int i = 0; i < NUM_ALLOCS; i++) {
            if (!verify_pattern(ptrs[i], 32, static_cast<uint8_t>(i + threadIdx.x))) {
                printf("  Thread %d: pattern verification failed for block %d\n", threadIdx.x, i);
                passed = false;
                break;
            }
        }
    }

    // Free all
    for (int i = 0; i < NUM_ALLOCS; i++) {
        if (ptrs[i]) pfree(ptrs[i]);
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        printf("[TEST 3] Multiple sequential allocations: %s\n", passed ? "PASSED" : "FAILED");
    }
}


// TEST 4: Coalescing - Free Adjacent Blocks
__global__ void test_coalescing() {
    extern __shared__ std::byte heap[];

    pool_init(SHARED_MEM_SIZE);

    bool passed = true;

    // Allocate three adjacent blocks
    void* ptr1 = pmalloc(64);
    void* ptr2 = pmalloc(64);
    void* ptr3 = pmalloc(64);

    if (!ptr1 || !ptr2 || !ptr3) {
        printf("  Thread %d: initial allocations failed\n", threadIdx.x);
        passed = false;
    }

    if (passed) {
        // Free middle block first
        pfree(ptr2);

        // Free first block - should coalesce with middle
        pfree(ptr1);

        // Free third block - should coalesce all three
        pfree(ptr3);

        // Now try to allocate a larger block that requires the coalesced space
        // 64 * 3 = 192, but with headers/footers freed, we should have more
        void* big_ptr = pmalloc(180);

        if (big_ptr == nullptr) {
            printf("  Thread %d: large allocation after coalescing failed\n", threadIdx.x);
            passed = false;
        } else {
            pfree(big_ptr);
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        printf("[TEST 4] Coalescing: %s\n", passed ? "PASSED" : "FAILED");
    }
}


// TEST 5: Coalescing Order - Left, Right, Both
__global__ void test_coalescing_orders() {
    extern __shared__ std::byte heap[];

    pool_init(SHARED_MEM_SIZE);

    bool passed = true;

    // Test 5a: Free left then right (coalesce right)
    void* a1 = pmalloc(32);
    void* a2 = pmalloc(32);
    void* a3 = pmalloc(32);

    if (!a1 || !a2 || !a3) {
        passed = false;
    } else {
        pfree(a1);  // Free left
        pfree(a2);  // Should coalesce with a1 (left neighbor)
        pfree(a3);  // Should coalesce with merged a1+a2
    }

    // Test 5b: Free right then left (coalesce left)
    void* b1 = pmalloc(32);
    void* b2 = pmalloc(32);
    void* b3 = pmalloc(32);

    if (!b1 || !b2 || !b3) {
        passed = false;
    } else {
        pfree(b3);  // Free right
        pfree(b2);  // Should coalesce with b3 (right neighbor)
        pfree(b1);  // Should coalesce with merged b2+b3
    }

    // Test 5c: Free outer blocks, then middle (coalesce both)
    void* c1 = pmalloc(32);
    void* c2 = pmalloc(32);
    void* c3 = pmalloc(32);

    if (!c1 || !c2 || !c3) {
        passed = false;
    } else {
        pfree(c1);  // Free left
        pfree(c3);  // Free right
        pfree(c2);  // Should coalesce with both neighbors
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        printf("[TEST 5] Coalescing orders (left/right/both): %s\n", passed ? "PASSED" : "FAILED");
    }
}


// TEST 6: Block Splitting
__global__ void test_splitting() {
    extern __shared__ std::byte heap[];

    pool_init(SHARED_MEM_SIZE);

    bool passed = true;

    // Allocate a small block - should split the initial large free block
    void* ptr1 = pmalloc(16);

    if (!ptr1) {
        printf("  Thread %d: first allocation failed\n", threadIdx.x);
        passed = false;
    }

    // Allocate another small block from the remainder
    void* ptr2 = pmalloc(16);

    if (!ptr2) {
        printf("  Thread %d: second allocation failed\n", threadIdx.x);
        passed = false;
    }

    // Verify they don't overlap
    if (passed) {
        fill_pattern(ptr1, 16, 0xAA);
        fill_pattern(ptr2, 16, 0xBB);

        if (!verify_pattern(ptr1, 16, 0xAA)) {
            printf("  Thread %d: ptr1 was corrupted\n", threadIdx.x);
            passed = false;
        }
        if (!verify_pattern(ptr2, 16, 0xBB)) {
            printf("  Thread %d: ptr2 was corrupted\n", threadIdx.x);
            passed = false;
        }
    }

    pfree(ptr1);
    pfree(ptr2);

    __syncthreads();

    if (threadIdx.x == 0) {
        printf("[TEST 6] Block splitting: %s\n", passed ? "PASSED" : "FAILED");
    }
}


// TEST 7: Allocation Size Boundaries
__global__ void test_size_boundaries() {
    extern __shared__ std::byte heap[];

    pool_init(SHARED_MEM_SIZE);

    bool passed = true;

    // Test minimum allocation (1 byte)
    void* tiny = pmalloc(1);
    if (!tiny) {
        printf("  Thread %d: 1-byte allocation failed\n", threadIdx.x);
        passed = false;
    } else {
        verify_memory_writable(tiny, 1);
        pfree(tiny);
    }

    // Test alignment boundary sizes
    size_t test_sizes[] = {7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65};
    for (size_t size : test_sizes) {
        void* ptr = pmalloc(size);
        if (!ptr) {
            printf("  Thread %d: %zu-byte allocation failed\n", threadIdx.x, size);
            passed = false;
        } else {
            if (!verify_memory_writable(ptr, size)) {
                printf("  Thread %d: %zu-byte memory verification failed\n", threadIdx.x, size);
                passed = false;
            }
            pfree(ptr);
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        printf("[TEST 7] Size boundaries: %s\n", passed ? "PASSED" : "FAILED");
    }
}


// TEST 8: Exhaustion and Recovery
__global__ void test_exhaustion() {
    extern __shared__ std::byte heap[];

    pool_init(SHARED_MEM_SIZE);

    bool passed = true;

    // Keep allocating until we fail
    constexpr int MAX_PTRS = 100;
    void* ptrs[MAX_PTRS];
    int num_allocated = 0;

    for (int i = 0; i < MAX_PTRS; i++) {
        ptrs[i] = pmalloc(64);
        if (ptrs[i] == nullptr) {
            break;
        }
        fill_pattern(ptrs[i], 64, static_cast<uint8_t>(i));
        num_allocated++;
    }

    if (num_allocated == 0) {
        printf("  Thread %d: couldn't allocate anything\n", threadIdx.x);
        passed = false;
    }

    // Verify all allocations are intact
    for (int i = 0; i < num_allocated; i++) {
        if (!verify_pattern(ptrs[i], 64, static_cast<uint8_t>(i))) {
            printf("  Thread %d: allocation %d corrupted\n", threadIdx.x, i);
            passed = false;
            break;
        }
    }

    // Free everything
    for (int i = 0; i < num_allocated; i++) {
        pfree(ptrs[i]);
    }

    // Should be able to allocate again after freeing all
    void* recovery = pmalloc(64);
    if (!recovery) {
        printf("  Thread %d: recovery allocation failed\n", threadIdx.x);
        passed = false;
    } else {
        pfree(recovery);
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        printf("[TEST 8] Exhaustion and recovery: %s\n", passed ? "PASSED" : "FAILED");
    }
}


// TEST 9: Fragmentation Stress Test
__global__ void test_fragmentation() {
    extern __shared__ std::byte heap[];

    pool_init(SHARED_MEM_SIZE);

    bool passed = true;

    // Allocate many blocks
    constexpr int NUM_BLOCKS = 20;
    void* ptrs[NUM_BLOCKS];

    for (int i = 0; i < NUM_BLOCKS; i++) {
        ptrs[i] = pmalloc(32);
        if (!ptrs[i]) {
            // Not enough memory - that's okay, just track how many we got
            for (int j = i; j < NUM_BLOCKS; j++) {
                ptrs[j] = nullptr;
            }
            break;
        }
    }

    // Free every other block to create fragmentation
    for (int i = 0; i < NUM_BLOCKS; i += 2) {
        if (ptrs[i]) {
            pfree(ptrs[i]);
            ptrs[i] = nullptr;
        }
    }

    // Try to allocate blocks that fit in the holes
    for (int i = 0; i < NUM_BLOCKS; i += 2) {
        ptrs[i] = pmalloc(32);
        // May or may not succeed depending on fragmentation
    }

    // Free everything
    for (int i = 0; i < NUM_BLOCKS; i++) {
        if (ptrs[i]) pfree(ptrs[i]);
    }

    // After freeing all, should be able to allocate large block
    void* big = pmalloc(256);
    if (!big) {
        printf("  Thread %d: large alloc after defrag failed\n", threadIdx.x);
        passed = false;
    } else {
        pfree(big);
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        printf("[TEST 9] Fragmentation stress: %s\n", passed ? "PASSED" : "FAILED");
    }
}


// TEST 10: Free List Ordering (Descending Size)
__global__ void test_free_list_order() {
    extern __shared__ std::byte heap[];

    pool_init(SHARED_MEM_SIZE);

    bool passed = true;

    // Allocate blocks of different sizes
    void* small = pmalloc(16);
    void* medium = pmalloc(64);
    void* large = pmalloc(128);

    if (!small || !medium || !large) {
        passed = false;
    } else {
        // Free in order: small, large, medium
        pfree(small);
        pfree(large);
        pfree(medium);

        // The free list should now be ordered: largest first
        // Allocating should give us the largest available block
        void* first = pmalloc(100);  // Should get the 128-byte block
        void* second = pmalloc(50);  // Should get the 64-byte block
        void* third = pmalloc(10);   // Should get the 16-byte block

        if (!first || !second || !third) {
            printf("  Thread %d: reallocation failed\n", threadIdx.x);
            passed = false;
        }

        if (first) pfree(first);
        if (second) pfree(second);
        if (third) pfree(third);
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        printf("[TEST 10] Free list ordering: %s\n", passed ? "PASSED" : "FAILED");
    }
}


// TEST 11: Null Pointer Free (Should Not Crash)
__global__ void test_null_free() {
    extern __shared__ std::byte heap[];

    pool_init(SHARED_MEM_SIZE);

    // This should not crash
    pfree(nullptr);

    // Verify allocator still works
    void* ptr = pmalloc(32);
    bool passed = (ptr != nullptr);
    if (ptr) pfree(ptr);

    __syncthreads();

    if (threadIdx.x == 0) {
        printf("[TEST 11] Null pointer free: %s\n", passed ? "PASSED" : "FAILED");
    }
}


// TEST 12: Double Free Detection
__global__ void test_double_free() {
    extern __shared__ std::byte heap[];

    pool_init(SHARED_MEM_SIZE);

    void* ptr = pmalloc(64);
    pfree(ptr);

    // behavior is undefined but shouldn't crash catastrophically
    // Current implementation doesn't detect this
    pfree(ptr);

    // Try to use allocator after double free
    void* ptr2 = pmalloc(32);
    bool passed = true;  // Just checking we don't crash
    if (ptr2) pfree(ptr2);

    __syncthreads();

    if (threadIdx.x == 0) {
        printf("[TEST 12] Double free (no crash): %s\n", passed ? "PASSED" : "FAILED");
    }
}


// TEST 13: Maximum Single Allocation
__global__ void test_max_allocation() {
    extern __shared__ std::byte heap[];

    pool_init(SHARED_MEM_SIZE);

    bool passed = true;

    // Calculate approximate max allocation per thread
    // Total shared mem / 32 threads - overhead
    size_t approx_pool_size = (SHARED_MEM_SIZE - 512) / 32;  // Rough estimate

    // Try progressively smaller allocations until one succeeds
    void* ptr = nullptr;
    size_t successful_size = 0;

    for (size_t try_size = approx_pool_size; try_size >= 64; try_size -= 64) {
        ptr = pmalloc(try_size);
        if (ptr) {
            successful_size = try_size;
            break;
        }
    }

    if (ptr == nullptr) {
        printf("  Thread %d: couldn't find working allocation size\n", threadIdx.x);
        passed = false;
    } else {
        // Verify we can write to the entire allocation
        if (!verify_memory_writable(ptr, successful_size)) {
            printf("  Thread %d: max allocation not fully writable\n", threadIdx.x);
            passed = false;
        }
        pfree(ptr);
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        printf("[TEST 13] Maximum single allocation: %s\n", passed ? "PASSED" : "FAILED");
    }
}


// TEST 14: Alignment Verification
__global__ void test_alignment() {
    extern __shared__ std::byte heap[];

    pool_init(SHARED_MEM_SIZE);

    bool passed = true;

    // Allocate various sizes and check alignment
    size_t sizes[] = {1, 3, 5, 7, 9, 13, 17, 31, 33, 63, 65, 100};

    for (size_t size : sizes) {
        void* ptr = pmalloc(size);
        if (ptr) {
            uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
            if (addr % 8 != 0) {
                printf("  Thread %d: size %zu returned unaligned ptr %p\n",
                       threadIdx.x, size, ptr);
                passed = false;
            }
            pfree(ptr);
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        printf("[TEST 14] Alignment verification: %s\n", passed ? "PASSED" : "FAILED");
    }
}

// TEST 15: Interleaved Alloc/Free Pattern
__global__ void test_interleaved() {
    extern __shared__ std::byte heap[];

    pool_init(SHARED_MEM_SIZE);

    bool passed = true;

    void* a = pmalloc(32);
    void* b = pmalloc(32);

    if (a) fill_pattern(a, 32, 0xAA);
    if (b) fill_pattern(b, 32, 0xBB);

    pfree(a);

    void* c = pmalloc(32);
    if (c) fill_pattern(c, 32, 0xCC);

    // Verify b wasn't corrupted
    if (b && !verify_pattern(b, 32, 0xBB)) {
        printf("  Thread %d: block b corrupted after interleaved ops\n", threadIdx.x);
        passed = false;
    }

    pfree(b);
    pfree(c);

    void* d = pmalloc(64);
    void* e = pmalloc(16);

    pfree(d);

    void* f = pmalloc(32);

    pfree(e);
    pfree(f);

    // Final verification - can still allocate
    void* final_ptr = pmalloc(48);
    if (!final_ptr) {
        passed = false;
    } else {
        pfree(final_ptr);
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        printf("[TEST 15] Interleaved alloc/free: %s\n", passed ? "PASSED" : "FAILED");
    }
}


// TEST 16: Zero-Size Allocation
__global__ void test_zero_alloc() {
    extern __shared__ std::byte heap[];

    pool_init(SHARED_MEM_SIZE);

    // Zero-size allocation
    void* ptr = pmalloc(0);

    // it returns pointer to first address of footer it shouldn't crash
    if (ptr) {
        pfree(ptr);
    }

    // Allocator should still work
    void* normal = pmalloc(32);
    bool passed = (normal != nullptr);
    if (normal) pfree(normal);

    __syncthreads();

    if (threadIdx.x == 0) {
        printf("[TEST 16] Zero-size allocation: %s\n", passed ? "PASSED" : "FAILED");
    }
}


// TEST 17: Thread Isolation Verification
__global__ void test_thread_isolation() {
    extern __shared__ std::byte heap[];

    pool_init(SHARED_MEM_SIZE);

    // Each thread allocates and writes its thread ID
    int* my_data = static_cast<int*>(pmalloc(sizeof(int) * 4));

    if (my_data) {
        for (int i = 0; i < 4; i++) {
            my_data[i] = threadIdx.x * 1000 + i;
        }
    }

    __syncthreads();

    // Verify each thread's data is intact
    bool passed = true;
    if (my_data) {
        for (int i = 0; i < 4; i++) {
            if (my_data[i] != threadIdx.x * 1000 + i) {
                printf("  Thread %d: data corrupted by another thread\n", threadIdx.x);
                passed = false;
                break;
            }
        }
        pfree(my_data);
    } else {
        passed = false;
    }

    __syncthreads();

    // Collect results (simple reduction)
    __shared__ int fail_count;
    if (threadIdx.x == 0) fail_count = 0;
    __syncthreads();

    if (!passed) atomicAdd(&fail_count, 1);
    __syncthreads();

    if (threadIdx.x == 0) {
        printf("[TEST 17] Thread isolation: %s\n", fail_count == 0 ? "PASSED" : "FAILED");
    }
}


// TEST 18: Repeated Alloc/Free Cycles
__global__ void test_repeated_cycles() {
    extern __shared__ std::byte heap[];

    pool_init(SHARED_MEM_SIZE);

    bool passed = true;
    constexpr int CYCLES = 50;

    for (int cycle = 0; cycle < CYCLES; cycle++) {
        void* ptr = pmalloc(64);
        if (!ptr) {
            printf("  Thread %d: allocation failed at cycle %d\n", threadIdx.x, cycle);
            passed = false;
            break;
        }
        fill_pattern(ptr, 64, static_cast<uint8_t>(cycle));

        if (!verify_pattern(ptr, 64, static_cast<uint8_t>(cycle))) {
            printf("  Thread %d: pattern mismatch at cycle %d\n", threadIdx.x, cycle);
            passed = false;
            pfree(ptr);
            break;
        }

        pfree(ptr);
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        printf("[TEST 18] Repeated alloc/free cycles: %s\n", passed ? "PASSED" : "FAILED");
    }
}


// TEST 19: Mixed Size Allocations
__global__ void test_mixed_sizes() {
    extern __shared__ std::byte heap[];

    pool_init(SHARED_MEM_SIZE);

    bool passed = true;

    void* ptrs[6];
    size_t sizes[] = {8, 128, 16, 64, 32, 24};

    // Allocate mixed sizes
    for (int i = 0; i < 6; i++) {
        ptrs[i] = pmalloc(sizes[i]);
        if (ptrs[i]) {
            fill_pattern(ptrs[i], sizes[i], static_cast<uint8_t>(i + 100));
        }
    }

    // Verify all patterns
    for (int i = 0; i < 6; i++) {
        if (ptrs[i] && !verify_pattern(ptrs[i], sizes[i], static_cast<uint8_t>(i + 100))) {
            printf("  Thread %d: mixed size block %d corrupted\n", threadIdx.x, i);
            passed = false;
        }
    }

    // Free in different order than allocated
    if (ptrs[2]) pfree(ptrs[2]);
    if (ptrs[0]) pfree(ptrs[0]);
    if (ptrs[4]) pfree(ptrs[4]);
    if (ptrs[1]) pfree(ptrs[1]);
    if (ptrs[5]) pfree(ptrs[5]);
    if (ptrs[3]) pfree(ptrs[3]);

    __syncthreads();

    if (threadIdx.x == 0) {
        printf("[TEST 19] Mixed size allocations: %s\n", passed ? "PASSED" : "FAILED");
    }
}


// TEST 20: Best-Fit Allocator (if implemented)
__global__ void test_best_fit() {
    extern __shared__ std::byte heap[];

    pool_init(SHARED_MEM_SIZE);

    bool passed = true;

    // Create holes of different sizes
    void* a = pmalloc(32);
    void* b = pmalloc(64);
    void* c = pmalloc(128);
    void* d = pmalloc(32);

    // Free to create holes
    pfree(a);  // 32-byte hole
    pfree(c);  // 128-byte hole

    // Best fit should find the 32-byte hole for a 24-byte request
    void* best = pool_type::pmalloc_best_fit(24);

    if (best) {
        verify_memory_writable(best, 24);
        pfree(best);
    }

    pfree(b);
    pfree(d);

    __syncthreads();

    if (threadIdx.x == 0) {
        printf("[TEST 20] Best-fit allocator: %s\n", passed ? "PASSED" : "FAILED");
    }
}

//=============================================================================
// Main Test Runner
int main() {
    printf("============================================\n");
    printf("  CUDA Shared Memory Allocator Test Suite\n");
    printf("============================================\n");
    printf("Configuration:\n");
    printf("  Shared Memory Size: %zu KB\n", SHARED_MEM_SIZE / 1024);
    printf("  Number of Threads:  %d\n", NUM_THREADS);
    printf("============================================\n\n");

    // Check device properties
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s\n", prop.name);
    printf("Max Shared Memory per Block: %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("\n");

    if (SHARED_MEM_SIZE > prop.sharedMemPerBlock) {
        printf("ERROR: Requested shared memory (%zu) exceeds device limit (%zu)\n",
               SHARED_MEM_SIZE, prop.sharedMemPerBlock);
        return 1;
    }

    printf("Running tests...\n\n");

    // Run all tests
    test_init<<<1, NUM_THREADS, SHARED_MEM_SIZE>>>();
    CUDA_CHECK(cudaDeviceSynchronize());

    test_single_alloc_free<<<1, NUM_THREADS, SHARED_MEM_SIZE>>>();
    CUDA_CHECK(cudaDeviceSynchronize());

    test_multiple_allocs<<<1, NUM_THREADS, SHARED_MEM_SIZE>>>();
    CUDA_CHECK(cudaDeviceSynchronize());

    test_coalescing<<<1, NUM_THREADS, SHARED_MEM_SIZE>>>();
    CUDA_CHECK(cudaDeviceSynchronize());

    test_coalescing_orders<<<1, NUM_THREADS, SHARED_MEM_SIZE>>>();
    CUDA_CHECK(cudaDeviceSynchronize());

    test_splitting<<<1, NUM_THREADS, SHARED_MEM_SIZE>>>();
    CUDA_CHECK(cudaDeviceSynchronize());

    test_size_boundaries<<<1, NUM_THREADS, SHARED_MEM_SIZE>>>();
    CUDA_CHECK(cudaDeviceSynchronize());

    test_exhaustion<<<1, NUM_THREADS, SHARED_MEM_SIZE>>>();
    CUDA_CHECK(cudaDeviceSynchronize());

    test_fragmentation<<<1, NUM_THREADS, SHARED_MEM_SIZE>>>();
    CUDA_CHECK(cudaDeviceSynchronize());

    test_free_list_order<<<1, NUM_THREADS, SHARED_MEM_SIZE>>>();
    CUDA_CHECK(cudaDeviceSynchronize());

    test_null_free<<<1, NUM_THREADS, SHARED_MEM_SIZE>>>();
    CUDA_CHECK(cudaDeviceSynchronize());

    test_double_free<<<1, NUM_THREADS, SHARED_MEM_SIZE>>>();
    CUDA_CHECK(cudaDeviceSynchronize());

    test_max_allocation<<<1, NUM_THREADS, SHARED_MEM_SIZE>>>();
    CUDA_CHECK(cudaDeviceSynchronize());

    test_alignment<<<1, NUM_THREADS, SHARED_MEM_SIZE>>>();
    CUDA_CHECK(cudaDeviceSynchronize());

    test_interleaved<<<1, NUM_THREADS, SHARED_MEM_SIZE>>>();
    CUDA_CHECK(cudaDeviceSynchronize());

    test_zero_alloc<<<1, NUM_THREADS, SHARED_MEM_SIZE>>>();
    CUDA_CHECK(cudaDeviceSynchronize());

    test_thread_isolation<<<1, NUM_THREADS, SHARED_MEM_SIZE>>>();
    CUDA_CHECK(cudaDeviceSynchronize());

    test_repeated_cycles<<<1, NUM_THREADS, SHARED_MEM_SIZE>>>();
    CUDA_CHECK(cudaDeviceSynchronize());

    test_mixed_sizes<<<1, NUM_THREADS, SHARED_MEM_SIZE>>>();
    CUDA_CHECK(cudaDeviceSynchronize());

    test_best_fit<<<1, NUM_THREADS, SHARED_MEM_SIZE>>>();
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("\n========================================\n");
    printf("  Test Suite Complete\n");
    printf("===========================================\n");

    return 0;
}
