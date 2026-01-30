#ifndef POOLALLOC_CUH
#define POOLALLOC_CUH

#include "typeDefs.cuh"
#include "RBTree.cuh"

typedef struct {

    /*while 6 byte accesses are bad, 2 bytes should be ok because they will still be fully accessed in one word. 
    I will prob pad the size to align the total allocation to 4 bytes however*/
    
    int16_t headerOffset;
} BlockFooter;


typedef struct {
    unsigned char* memBuffer;
    RBTreeBlockHeader *freeList;
    RBTreeBlockHeader *fullList;
} MemBufferStorage;



__device__ void init_gpu_buffer(unsigned int incomingMemSize);

__device__ void *cmalloc(unsigned long size);

__device__ void cfree(void *ptr);

__device__ void debug_print_buffer();
__device__ void debug_print_free_list();
__device__ void debug_print_full_list();

// __device__ void printlayout();

// __device__ void printbytes();

// __device__ int dataBytes(BlockHeader *head);

// __device__ int headerBytes(BlockHeader *head);
#endif