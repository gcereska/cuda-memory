#include <stdio.h>
#include <cuda_runtime.h>

#include "poolAlloc.cuh"

__global__ void allocate_and_write(int **ptrs, int n, uint sharedMemSize) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // poolinit(poolMemoryBlock, idx);
    
    init_gpu_buffer(sharedMemSize);

    if(idx == 0){
        debug_print_buffer();
    }

    if (idx < n) {

        int *mem = (int*)cmalloc(4 * sizeof(int));

        printf("thread: %d, mem pointer: %p\n",idx, mem);

        if (mem != NULL) {
            for (int i = 0; i < 4; ++i) {
                mem[i] = idx * 10 + i;
            }
            ptrs[idx] = mem;
        } else {
            ptrs[idx] = NULL;
        }

        if(idx == 0){
            debug_print_buffer();
        }

        printf("Thread %d: ", idx);
        for (int i = 0; i < 4; ++i) {
            printf("%d ", mem[i]);
        }
        printf("\n");
        cfree(mem);
    }

    if(idx == 0){
        debug_print_buffer();
    }

    // if (idx < n && ptrs[idx] != NULL) {
        
    // }
    
}

//attempting to access the same piece of shared memory across multiple kernels is illegal and will result in undefined behaviors
__global__ void read_and_free(int **ptrs, int n) {
    // unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // printf("Thread %d: ", idx);
    // if (idx < n && ptrs[idx] != NULL) {
    //     for (int i = 0; i < 4; ++i) {
    //         printf("%d ", ptrs[idx][i]);
    //     }
    //     cfree(ptrs[idx]);
    // }
}

int main() {
    int n = 8;
    int **d_ptrs;

    uint sharedMemSize = 49152;

    printf("here\n");
    cudaMalloc(&d_ptrs, n * sizeof(int*));


    allocate_and_write<<<1, n, sharedMemSize>>>(d_ptrs, n, sharedMemSize);
    cudaDeviceSynchronize();

    read_and_free<<<1, n, sharedMemSize>>>(d_ptrs, n);
    cudaDeviceSynchronize();
    

    cudaFree(d_ptrs);
    return 0;
}
