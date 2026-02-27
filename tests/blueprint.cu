#include <stdio.h>
//#include <cstdio>
#include <cuda_runtime.h>

#include <configure.h>
#include <cumem/blueprint.h>


__global__ void simple_test_kernel(size_t pool_size) {

    if (threadIdx.x == 11) printf("Kernel Entered\n");

    pool_init(pool_size);

    if (threadIdx.x == 11) printf("pool_init success\n");

    void* ptr = pmalloc(64);

    if (threadIdx.x == 11) printf("pmalloc success\n");

    if (ptr != nullptr) {
        if (threadIdx.x == 11) printf("Thread %d: allocated 64 bytes at %p\n", threadIdx.x, ptr);
        pfree(ptr);
        if (threadIdx.x == 11) printf("pfree success\n");
        if (threadIdx.x == 11) printf("Thread %d: freed memory\n", threadIdx.x);
    } else {
        if (threadIdx.x == 11) printf("Thread %d: allocation failed\n", threadIdx.x);
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
