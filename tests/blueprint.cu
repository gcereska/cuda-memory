#include <stdio.h>
//#include <cstdio>
#include <cuda_runtime.h>

#include "allocator.cuh"
#include "poolAlloc.cuh"
#include "poolAllocBST.cuh"
// "typeDefs.cuh" also required in directory
// libcuda_memory_cuda_release.a contains the .o for all allocators

#if defined(USE_THREAD_LOCAL_BEST_FIT)
    #define ALLOCATOR_NAME "Thread Local (Best Fit)"
    #define pool_init(size) thread_pool::pool_init(size)
    #define pmalloc(size)   thread_pool::pmalloc_best_fit(size)
    #define pfree(ptr)      thread_pool::pfree(ptr)

#elif defined(USE_WARP_LOCAL_BEST_FIT)
    #define ALLOCATOR_NAME "Warp Local (Best Fit)"
    #define pool_init(size) warp_pool::pool_init(size) 
    #define pmalloc(size)   warp_pool::pmalloc_best_fit(size)
    #define pfree(ptr)      warp_pool::pfree(ptr)

#elif defined(USE_THREAD_LOCAL)
    #define ALLOCATOR_NAME "Thread Local (First Fit)"
    #define pool_init(size) thread_pool::pool_init(size)
    #define pmalloc(size)   thread_pool::pmalloc(size)
    #define pfree(ptr)      thread_pool::pfree(ptr)

#elif defined(USE_WARP_LOCAL)
    #define ALLOCATOR_NAME "Warp Local (First Fit)"
    #define pool_init(size) warp_pool::pool_init(size)
    #define pmalloc(size)   warp_pool::pmalloc(size)
    #define pfree(ptr)      warp_pool::pfree(ptr)

#elif defined(USE_FREELIST_ALLOCATOR)
    #define ALLOCATOR_NAME "Julian (Standard Freelist)"
    #define pool_init(size) pmalloc_freelist::init_gpu_buffer(size)
    #define pmalloc(size)   pmalloc_freelist::cmalloc(size)
    #define pfree(ptr)      pmalloc_freelist::cfree(ptr)

#elif defined(USE_BST_ALLOCATOR)
    #define ALLOCATOR_NAME "Julian (BST)"
    #define pool_init(size) pmalloc_bst::init_gpu_buffer(size)
    #define pmalloc(size)   pmalloc_bst::cmalloc(size)
    #define pfree(ptr)      pmalloc_bst::cfree(ptr)

#else
    #define ALLOCATOR_NAME "Native Device Malloc"
    #define pool_init(size) /* no pool init */
    #define pmalloc(size)   malloc(size)
    #define pfree(ptr)      free(ptr)
#endif

__global__ void simple_test_kernel(size_t pool_size) {


    printf("Kernel Entered\n");

    pool_init(pool_size);
    printf("Pool init success\n");

    void* ptr = pmalloc(64);
    printf("Malloc Success\n");

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
    
    cudaError_t launchErr = cudaGetLastError();
    if (launchErr != cudaSuccess) {
        printf("Launch error: %s\n", cudaGetErrorString(launchErr));
    }

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Runtime error: %s\n", cudaGetErrorString(err));
    }
        
    printf("\nDone\n");
    
    return 0;
}