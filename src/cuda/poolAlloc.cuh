#pragma once

#include "typeDefs.cuh"


namespace list_pool {

    struct MemBufferStorage {
        unsigned char* memBuffer;
        BlockHeader* freeList;     // <-- BlockHeader
        BlockHeader* fullList;
    };

    __device__ void pool_init(uint sharedMemSize);

    __device__ void *pmalloc(unsigned long size);

    __device__ void pfree(void *ptr);


    __device__ void debug_print_buffer();
    __device__ void debug_print_free_list();
    __device__ void debug_print_full_list();

}

// __device__ void printlayout();

// __device__ void printbytes();

// __device__ int dataBytes(BlockHeader *head);

// __device__ int headerBytes(BlockHeader *head);
