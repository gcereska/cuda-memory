#include <stdio.h>
//#include <cstdio>
#include <cuda_runtime.h>

#include "allocator.cuh"
#include "poolAlloc.cuh"
#include "poolAllocBST.cuh"
// "typeDefs.cuh" also required in directory
// libcuda_memory_cuda_release.a contains the .o for all allocators

#if defined(USE_LOCAL_ALLOCATOR)

    // --- Thread Local or Warp Local
    #if defined(USE_THREAD_LOCAL)
        #if defined(USE_PMALLOC_BEST_FIT)
            #define ALLOCATOR_NAME "Thread Local (Best Fit)"
            #define pool_init(size) thread_pool::pool_init(size)
            #define pmalloc(size)   thread_pool::pmalloc_best_fit(size)
            #define pfree(ptr)      thread_pool::pfree(ptr)
        #else
            #define ALLOCATOR_NAME "Thread Local (First Fit)"
            #define pool_init(size) thread_pool::pool_init(size)
            #define pmalloc(size)   thread_pool::pmalloc(size)
            #define pfree(ptr)      thread_pool::pfree(ptr)
        #endif

    #else // Warp Local
        #if defined(USE_PMALLOC_BEST_FIT)
            #define ALLOCATOR_NAME "Warp Local (Best Fit)"
            #define pool_init(size) warp_pool::pool_init(size) 
            #define pmalloc(size)   warp_pool::pmalloc_best_fit(size)
            #define pfree(ptr)      warp_pool::pfree(ptr)
        #else
            #define ALLOCATOR_NAME "Warp Local (First Fit)"
            #define pool_init(size) warp_pool::pool_init(size)
            #define pmalloc(size)   warp_pool::pmalloc(size)
            #define pfree(ptr)      warp_pool::pfree(ptr)
        #endif
    #endif

#else

    // Julian's Allocators
    #if defined(USE_POOL_ALLOC_BST)
        #define ALLOCATOR_NAME "Julian (BST)"
        #define pool_init(size) pmalloc_bst::init_gpu_buffer(size)
        #define pmalloc(size)   pmalloc_bst::cmalloc(size)
        #define pfree(ptr)      pmalloc_bst::cfree(ptr)
    #else
        #define ALLOCATOR_NAME "Julian (Standard Freelist)"
        #define pool_init(size) pmalloc_freelist::init_gpu_buffer(size)
        #define pmalloc(size)   pmalloc_freelist::cmalloc(size)
        #define pfree(ptr)      pmalloc_freelist::cfree(ptr)
    #endif

#endif

__global__ void simple_test_kernel(size_t pool_size) {

    pool_init(pool_size);
    void* ptr = pmalloc(64);

    if (ptr != nullptr) {
        printf("Thread %d: allocated 64 bytes at %p\n", threadIdx.x, ptr);
        pfree(ptr);
        printf("Thread %d: freed memory\n", threadIdx.x);
    } else {
        printf("Thread %d: allocation failed\n", threadIdx.x);
    }
}

int main() {

    
    printf(" Allocator Mode: %s\n", ALLOCATOR_NAME);
    printf("============================================\n\n");
    
    size_t dyn_smem = 32 * 1024; // 32 KB shared memory
    

    
    simple_test_kernel<<<1, 32, dyn_smem>>>(dyn_smem);
    
    cudaDeviceSynchronize();
    
    printf("\nDone\n");
    
    return 0;
}