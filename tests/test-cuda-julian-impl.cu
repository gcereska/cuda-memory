#include <stdio.h>
#include <cuda_runtime.h>

#include <cumem/blueprint.h>

__global__ void allocate_and_write(int **ptrs, int n, uint sharedMemSize) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // poolinit(poolMemoryBlock, idx);
    printf("entering kernel correctly\n");
    pool_init(sharedMemSize);
    printf("init buffer correctly\n");

    if(idx == 0){
        //pool::debug_print_buffer();
    }

    if (idx < n) {

        int *mem = (int*)pmalloc(4 * sizeof(int));

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
            //pool::debug_print_buffer();
        }

        printf("Thread %d: ", idx);
        for (int i = 0; i < 4; ++i) {
            printf("%d ", mem[i]);
        }
        printf("\n");
        pfree(mem);
    }

    if(idx == 0){
        //pool::debug_print_buffer();
    }

    // if (idx < n && ptrs[idx] != NULL) {
        
    // }
    
}

//attempting to access the same piece of shared memory across multiple kernels is illegal and will result in undefined behaviors
// __global__ void read_and_free(int **ptrs, int n) {
//     // unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
//     printf("WE READING AND FREEING");
//     // if (idx < n && ptrs[idx] != NULL) {
//     //     for (int i = 0; i < 4; ++i) {
//     //         printf("%d ", ptrs[idx][i]);
//     //     }
//     //     cfree(ptrs[idx]);
//     // }
// }

int main() {
    int n = 8;
    int **d_ptrs;

    uint sharedMemSize = 24000;

    printf("here\n");
    cudaMalloc(&d_ptrs, n * sizeof(int*));

    printf("post mem alloc\n");

    uint threadPoolSize = sharedMemSize/n;
    allocate_and_write<<<1, n, sharedMemSize>>>(d_ptrs, n, threadPoolSize);
    cudaDeviceSynchronize();

    printf("post alloc & write\n");


    // read_and_free<<<1, n, sharedMemSize>>>(d_ptrs, n);
    cudaDeviceSynchronize();
    
    printf("post read & free\n");


    cudaFree(d_ptrs);
    return 0;
}
