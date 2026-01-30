//fuzzy-test-gpu.cu
#include <stdio.h>
#include <cuda_runtime.h>
#include <unordered_set>
//#include <getopt.h> im doing the tests on my own windows device
#include <assert.h>

//#include <memorypool/gpu/poolalloc.cuh>
#include "../include/allocator.cuh"

#define NUM_THREADS 32 //changed from 128 to 32 // must be 32 exactly 1 thread per warp/pool
#define OPS_PER_THREAD 30
#define SHARED_MEM_SIZE (32 * 1024) // 32KB

//__device__ extern MemoryPool g_memoryPools[1024];

struct TestOperation {
    bool isAlloc;
    unsigned long numBytes;
    int corrospondingAlloc;
};

size_t get_max_shared_mem(int device = 0) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    return prop.sharedMemPerBlock - 2048;    // -2048 because it overexaggerates how much shared memory you actually get
                                             // to some extent and it crashes elsewise
}

void generateRandomOperations(TestOperation *ops) {
    for (int i = 0; i < NUM_THREADS; i++) {
        std::unordered_set<size_t> allocationIndices;
        size_t startingInd = i * OPS_PER_THREAD;
        for (int j = 0; j < OPS_PER_THREAD; j++) {
            if (allocationIndices.size() == 0) {
                ops[startingInd + j].isAlloc = true;
                ops[startingInd + j].numBytes = 4 * sizeof(int);
                ops[startingInd + j].corrospondingAlloc = -1;
                allocationIndices.insert(j);
            } else if (rand() % 2 == 0) {
                ops[startingInd + j].isAlloc = true;
                ops[startingInd + j].numBytes = 4 * sizeof(int);
                ops[startingInd + j].corrospondingAlloc = -1;
                allocationIndices.insert(j);
            } else {
                ops[startingInd + j].isAlloc = false;
                ops[startingInd + j].numBytes = 0;
                auto it = allocationIndices.begin();
                std::advance(it, rand() % allocationIndices.size());
                ops[startingInd + j].corrospondingAlloc = *it;
                allocationIndices.erase(it);
            }
        }
    }
}

__global__ void runTests(TestOperation *ops, size_t shared_mem_size) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // #ifdef USE_MEMORY_POOL
    // poolinit(poolMemoryBlock, idx);
    // #endif
    pool_init(shared_mem_size);

    void *allocatedPtrs[OPS_PER_THREAD];

    for (unsigned int i = 0; i < OPS_PER_THREAD; i++) {
        unsigned int opIndex = idx * OPS_PER_THREAD + i;
        if (ops[opIndex].isAlloc) {
            void *ptr = pmalloc(ops[opIndex].numBytes);
            allocatedPtrs[i] = ptr;
            for (unsigned long j = 0; j < ops[opIndex].numBytes; j++) {
                ((char*)ptr)[j] = 'a';
            }
            printf("Thread %d: Allocated %lu bytes at %p\n", idx, ops[opIndex].numBytes, ptr);
        } else {
            char *ptrToFree = (char*)allocatedPtrs[ops[opIndex].corrospondingAlloc];
            for (unsigned long j = 0; j < ops[ops[opIndex].corrospondingAlloc].numBytes; j++) {
                if (ptrToFree[j] != 'a') {
                    printf("Thread %d: failed to free allocation at index %d\n", idx, ops[opIndex].corrospondingAlloc);
                    assert(ptrToFree[j] == 'a');
                }
            }
            pfree(ptrToFree);
            printf("Thread %d: Freed allocation at index %d\n", idx, ops[opIndex].corrospondingAlloc);
        }

    }
}

int main(int argc, char **argv) {
    //int opt;
    int seed = 0;
    //while ((opt = getopt(argc, argv, "s:")) != -1) {
    //    switch (opt) {
    //        case 's':
    //            seed = atoi(optarg);
    //            break;
    //        default:
    //            fprintf(stderr, "Usage: %s [-s seed]\n", argv[0]);
    //            exit(1);
    //    }
    //}

    // manual arg parsing because no getopt
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            seed = atoi(argv[++i]);
        }
    }
    srand(seed);
    printf("Random seed: %d\n", seed);

    TestOperation ops[NUM_THREADS * OPS_PER_THREAD];
    generateRandomOperations(ops);

    for (int i = 0; i < NUM_THREADS; i++) {
        printf("Thread %d operations:\n", i);
        for (int j = 0; j < OPS_PER_THREAD; j++) {
            int idx = i * OPS_PER_THREAD + j;
            printf("  Op %d: %s, numBytes=%lu, corrospondingAlloc=%d\n",
                   j,
                   ops[idx].isAlloc ? "ALLOC" : "FREE",
                   ops[idx].numBytes,
                   ops[idx].corrospondingAlloc);
        }
    }
    
    TestOperation *d_ops;
    cudaMalloc(&d_ops, sizeof(TestOperation) * NUM_THREADS * OPS_PER_THREAD);
    cudaMemcpy(d_ops, ops, sizeof(TestOperation) * NUM_THREADS * OPS_PER_THREAD, cudaMemcpyHostToDevice);

    // size_t shared_mem_size = get_max_shared_mem();
    runTests<<<1, NUM_THREADS, SHARED_MEM_SIZE>>>(d_ops, SHARED_MEM_SIZE);

    // AI recommended error check
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    cudaFree(d_ops);

    printf("Done Testing");
    return 0;
}
