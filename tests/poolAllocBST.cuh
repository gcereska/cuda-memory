#ifndef POOLALLOCBST_CUH
#define POOLALLOCBST_CUH

#include "typeDefs.cuh"
#include "RBTree.cuh"




namespace pmalloc_bst {

    struct MemBufferStorage {
        unsigned char* memBuffer;
        RBTreeBlockHeader* freeList;  // <-- RBTreeBlockHeader
        RBTreeBlockHeader* fullList;
    };

    __device__ void init_gpu_buffer(unsigned int incomingMemSize);

    __device__ void *cmalloc(unsigned long size);

    __device__ void cfree(void *ptr);

    __device__ void debug_print_buffer();
    __device__ void debug_print_free_list();
    __device__ void debug_print_full_list();
}



// __device__ void printlayout();

// __device__ void printbytes();

// __device__ int dataBytes(BlockHeader *head);

// __device__ int headerBytes(BlockHeader *head);
#endif