#ifndef POOLALLOC_CUH
#define POOLALLOC_CUH

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

#endif
