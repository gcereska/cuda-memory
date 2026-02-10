#include <stdio.h>
#include <cuda_runtime.h>

#include <memorypool/gpu/poolalloc.cuh>

__global__ void allocate_and_write(int **ptrs, int n, void *poolMemoryBlock) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    poolinit(poolMemoryBlock, idx);

    if (idx < n) {

        int *mem = (int*)poolmalloc(4 * sizeof(int));


        if (mem != NULL) {
            for (int i = 0; i < 4; ++i) {
                mem[i] = idx * 10 + i;
            }
            ptrs[idx] = mem;
        } else {
            ptrs[idx] = NULL;
        }
    }
}

__global__ void read_and_free(int **ptrs, int n) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    printf("Thread %d: ", idx);
    if (idx < n && ptrs[idx] != NULL) {
        for (int i = 0; i < 4; ++i) {
            printf("%d ", ptrs[idx][i]);
        }
        poolfree(ptrs[idx]);
    }
}

int main() {
    int n = 8;
    int **d_ptrs;
    printf("here");
    cudaMalloc(&d_ptrs, n * sizeof(int*));

    void *poolPtr = allocatePools(n);
    allocate_and_write<<<1, n>>>(d_ptrs, n, poolPtr);
    cudaDeviceSynchronize();

    read_and_free<<<1, n>>>(d_ptrs, n);
    cudaDeviceSynchronize();
    freePools(poolPtr);

    cudaFree(d_ptrs);
    return 0;
}
