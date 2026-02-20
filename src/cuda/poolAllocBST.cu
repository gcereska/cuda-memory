//#ifndef POOLALLOCBST_CU
//#define POOLALLOCBST_CU

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "typeDefs.cuh"
#include "RBTree.cuh"
#include "poolAllocBST.cuh"

namespace pmalloc_bst {

// #define GPU_MEM_POOL_SIZE 49152

#define MEM_MAX_THREADS 128

// typedef struct {
//     unsigned char* memBuffer;
//     RBTreeBlockHeader *freeList;
//     RBTreeBlockHeader *fullList;
// } MemBufferStorage;

__device__ MemBufferStorage memPools[MEM_MAX_THREADS];

__device__ uint sharedMemSize; 

//------------------------------------------------------------------------------------CUDA HELPER FUNCTIONS--------------------------------------------------------------------------------------------

__device__ uint get_thread_pool_size(){
    int total_threads =
        gridDim.x * gridDim.y * gridDim.z *
        blockDim.x * blockDim.y * blockDim.z;

    return (sharedMemSize/total_threads);
}

__device__ uint get_linear_thread_index(){
    return (((blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x) + blockIdx.x)
    * (blockDim.x * blockDim.y * blockDim.z)
    +
    (threadIdx.z * blockDim.y * blockDim.x +
     threadIdx.y * blockDim.x +
     threadIdx.x);
}

//------------------------------------------------------------------------------------HELPER FUNCTIONS TO DEAL WITH THE STRUCT--------------------------------------------------------------------------------------------


__device__ int8_t get_full(RBTreeBlockHeader* currentHeader){
    return currentHeader->fullSize < 0;
}

__device__ int8_t get_full(BlockHeader* currentHeader){
    return currentHeader->fullSize < 0;
}

//the size and full used to be packed into an int16 hence the helper functions
__device__ int16_t get_node_size(RBTreeBlockHeader* currentHeader){
    return currentHeader->fullSize & 0x3FFF;
}

__device__ int16_t get_node_size(BlockHeader* currentHeader){
    return currentHeader->fullSize & 0x7FFF;
}

__device__ int8_t get_node_color(RBTreeBlockHeader* currentHeader){
    return currentHeader->fullSize & 0x4000;
}

//these are all relatively simple helper functions to make the pointer math easier
static __device__ int16_t get_offset(unsigned char* from, unsigned char* to){
    return (unsigned char*)to - (unsigned char*)from;
}

__device__ BlockFooter* get_footer(RBTreeBlockHeader* header){
    return (BlockFooter*)((unsigned char*)header + get_node_size(header) + sizeof(RBTreeBlockHeader));
}

//gets the previous header from the total memPool
__device__ RBTreeBlockHeader* get_prev_header(RBTreeBlockHeader* currentHeader){

    uint threadIndex = get_linear_thread_index();


    if((void*)currentHeader == (void*)memPools[threadIndex].memBuffer){
        return NULL; 
    }

    BlockFooter* prev_footer = (BlockFooter*)((unsigned char*)currentHeader - sizeof(BlockFooter));
    RBTreeBlockHeader* prevHeader = (RBTreeBlockHeader*)((unsigned char*)prev_footer + prev_footer->headerOffset);
    return prevHeader;
}

//gets the next header from the total memPool
__device__ RBTreeBlockHeader* get_next_header(RBTreeBlockHeader* currentHeader){

    uint threadIndex = get_linear_thread_index();

    
    uint threadPoolSize = get_thread_pool_size(); 
    
    int currentHeaderOffset = get_offset(memPools[threadIndex].memBuffer, (unsigned char*)currentHeader);
    uint64_t temp_size = get_node_size(currentHeader);
    

    if((currentHeaderOffset + temp_size + sizeof(RBTreeBlockHeader) + sizeof(BlockFooter)) >= threadPoolSize){
        return NULL;
    }
    
    RBTreeBlockHeader* nextHeader = (RBTreeBlockHeader*)((unsigned char*)currentHeader + temp_size + sizeof(BlockFooter) + sizeof(RBTreeBlockHeader));
    return nextHeader;
}

//gets the next header from the list it current resides in
__device__ BlockHeader* get_next_list_header(BlockHeader* currentHeader){
    if(currentHeader->nextOffset == 0){
        return NULL;
    }
    return (BlockHeader*)((unsigned char*)currentHeader + currentHeader->nextOffset);
}

//gets the previous header from the list it current resides in
__device__ BlockHeader* get_prev_list_header(BlockHeader* currentHeader){
    if(currentHeader->prevOffset == 0){
        return NULL;
    }
    return (BlockHeader*)((unsigned char*)currentHeader + currentHeader->prevOffset);
}

__device__ void list_push_front(BlockHeader** list, BlockHeader* insertionNode){
    if(*list == NULL){
        *list = insertionNode;
        insertionNode->nextOffset = 0;
        insertionNode->prevOffset = 0;
        return;
    }
    
    (*list)->prevOffset = get_offset((unsigned char*)*list, (unsigned char*)insertionNode);
    insertionNode->nextOffset = get_offset((unsigned char*)insertionNode, (unsigned char*)*list);
    insertionNode->prevOffset = 0;
    *list = insertionNode;
}

__device__ void list_remove(BlockHeader** list, BlockHeader* deletionNode){
    BlockHeader* currentNodeNext = NULL;
    BlockHeader* currentNode_prev = NULL;
    
    if(deletionNode->nextOffset != 0){
        currentNodeNext = (BlockHeader*)((unsigned char*)deletionNode + deletionNode->nextOffset);
    }
    
    if(deletionNode->prevOffset != 0){
        currentNode_prev = (BlockHeader*)((unsigned char*)deletionNode + deletionNode->prevOffset);
    }
    
    // Fix the previous node's next pointer
    if(currentNode_prev != NULL){
        if(currentNodeNext != NULL){
            currentNode_prev->nextOffset = get_offset((unsigned char*)currentNode_prev, (unsigned char*)currentNodeNext);
        } else {
            currentNode_prev->nextOffset = 0;
        }
    } else {
        // deletionNode was the head of the list
        *list = currentNodeNext;
    }
    
    // Fix the next node's prev pointer
    if(currentNodeNext != NULL){
        if(currentNode_prev != NULL){
            currentNodeNext->prevOffset = get_offset((unsigned char*)currentNodeNext, (unsigned char*)currentNode_prev);
        } else {
            currentNodeNext->prevOffset = 0;
        }
    }
    
    // Clear the deleted node's pointers
    deletionNode->nextOffset = 0;
    deletionNode->prevOffset = 0;
}

//------------------------------------------------------------------------------------DEBUG FUNCTIONS--------------------------------------------------------------------------------------------


//used solely for debugging help and correctness checking
__device__ void debug_print_buffer(){

    uint threadIndex = get_linear_thread_index();
    

    printf("\n|");
    RBTreeBlockHeader* currentNode = (RBTreeBlockHeader*)memPools[threadIndex].memBuffer;
    while (currentNode != NULL){
        printf(" %4u |", get_node_size(currentNode));
        currentNode = get_next_header(currentNode);
    }
    printf("\n|");
    currentNode = (RBTreeBlockHeader*)memPools[threadIndex].memBuffer;

    while (currentNode != NULL){
        if(get_full(currentNode)){
            printf(" FULL |");
        } else {
            printf(" FREE |");
        }
        currentNode = get_next_header(currentNode);
    }
    printf(" \n\n");
}


// __device__ void debug_print_free_list(){

//     uint threadIndex = get_linear_thread_index();


//     printf("\n|");
//     RBTreeBlockHeader* currentNode = (RBTreeBlockHeader*)memPools[threadIndex].freeList;
//     while (currentNode != NULL){
//         printf(" %4u |", get_node_size(currentNode));
//         currentNode = get_next_list_header(currentNode);
//     }
//     printf("\n|");
//     currentNode = (RBTreeBlockHeader*)memPools[threadIndex].freeList;

//     while (currentNode != NULL){
//         if(get_full(currentNode)){
//             printf(" FULL |");
//         } else {
//             printf(" FREE |");
//         }
//         currentNode = get_next_list_header(currentNode);
//     }
//     printf(" \n\n");
// }


__device__ void debug_print_full_list(){

    uint threadIndex = get_linear_thread_index();


    printf("\n|");
    BlockHeader* currentNode = (BlockHeader*)memPools[threadIndex].fullList;
    while (currentNode != NULL){
        printf(" %4u |", get_node_size(currentNode));
        currentNode = get_next_list_header(currentNode);
    }
    printf("\n|");
    currentNode = (BlockHeader*)memPools[threadIndex].fullList;

    while (currentNode != NULL){
        if(get_full(currentNode)){
            printf(" FULL |");
        } else {
            printf(" FREE |");
        }
        currentNode = get_next_list_header(currentNode);
    }
    printf(" \n\n");
}

//------------------------------------------------------------------------------------START OF ACTUAL FUNCTIONS FOR MALLOC--------------------------------------------------------------------------------------------

//This function will take in the size of shared mem/#threads
__device__ void init_gpu_buffer(unsigned int incomingMemSize){

    

    extern __shared__ unsigned char smem[];

    uint threadIndex = get_linear_thread_index();

    // // printf("linear thread index: %d\n",threadIndex);
   
    if(threadIndex == 0){
        sharedMemSize = incomingMemSize;
    }
    __syncthreads();
    /*
    
    Because all operations use an offset based scheme to find neighbors,
    it is important to be very careful about the size of these the pool per thread. 

    At most it is acceptable to have 2^15 bytes,
    however past that the bit packing will be upset and the size of the offset variables will need to be upgraded

    THIS IS PARTICULARLY IMPORTANT IF A KERNEL IS LAUNCHED WITH FEW THREADS BUT A LARGE SHARED MEMORY POOL
    */



    uint threadPoolSize = get_thread_pool_size();

    //printf("thread pool size: %d\n",threadPoolSize);


    memPools[threadIndex].memBuffer = smem + (threadPoolSize * threadIndex);

    // printf("thread index %d's memory pool pointer: %p\n",threadIndex,memPools[threadIndex].memBuffer);


    // printf("mempool start pointer: %p\n", memPools[threadIndex].memBuffer);

    RBTreeBlockHeader *header = (RBTreeBlockHeader*)memPools[threadIndex].memBuffer;


    int16_t tempSize = (threadPoolSize - (sizeof(RBTreeBlockHeader) + sizeof(BlockFooter)));
    header->fullSize = tempSize & 0x7FFF;
    header->leftOffset = 0;
    header->rightOffset = 0;
    header->parentOffset = 0;

    memPools[threadIndex].freeList = NULL;


    insert_node(&memPools[threadIndex].freeList,header,tempSize);



    //list_push_front(&memPools[threadIndex].freeList, header);

    BlockFooter *footer = get_footer(header);
    footer->headerOffset = get_offset((unsigned char*)footer, (unsigned char*)header);


}

__device__ void* cmalloc(unsigned long size){

    uint threadIndex = get_linear_thread_index();

    if(&memPools[threadIndex].freeList == NULL || get_node_size(memPools[threadIndex].freeList) < (size + (int16_t)sizeof(RBTreeBlockHeader) + (int16_t)sizeof(BlockFooter))){
        printf("malloc eval null exit\n");
        return NULL;
    }

    uint16_t preAllocationSize = get_node_size(memPools[threadIndex].freeList);
    RBTreeBlockHeader* newlyAllocatedHeader = memPools[threadIndex].freeList;

    // Align size to 8 bytes, leaving the last 2 bytes for the block footer
    uint8_t offset = (8 - ((size + sizeof(BlockFooter)) % 8));
    uint16_t sizeToAlloc = size + offset;

    newlyAllocatedHeader->fullSize = sizeToAlloc & 0x3FFF;
    newlyAllocatedHeader->fullSize |= 0x8000; //sets full

    BlockFooter* newlyAllocatedFooter = get_footer(newlyAllocatedHeader);
    newlyAllocatedFooter->headerOffset = get_offset((unsigned char*)newlyAllocatedFooter, (unsigned char*)newlyAllocatedHeader);

    // Remove from free list before modifying
    remove(&memPools[threadIndex].freeList, newlyAllocatedHeader);
    // list_remove(&memPools[threadIndex].freeList, newlyAllocatedHeader);
    // list_push_front(&memPools[threadIndex].fullList, newlyAllocatedHeader);

    // Create new free block if there's enough space
    uint16_t remainingSize = preAllocationSize - (sizeToAlloc + sizeof(RBTreeBlockHeader) + sizeof(BlockFooter));
    
    if(remainingSize > 0){
        RBTreeBlockHeader* newFreeHeader = (RBTreeBlockHeader*)((unsigned char*)(newlyAllocatedFooter) + sizeof(BlockFooter));
        newFreeHeader->fullSize = remainingSize & 0x7FFF; //both sets size and sets empty
        newFreeHeader->leftOffset = 0;
        newFreeHeader->rightOffset = 0;
        newFreeHeader->parentOffset = 0;

        BlockFooter* newFreeFooter = get_footer(newFreeHeader);
        newFreeFooter->headerOffset = get_offset((unsigned char*)newFreeFooter, (unsigned char*)newFreeHeader);

        // list_push_front(&memPools[threadIndex].freeList, newFreeHeader);
        insert_node(&memPools[threadIndex].freeList,newFreeHeader,remainingSize);
    }


    void* return_index = (unsigned char*)newlyAllocatedHeader + sizeof(RBTreeBlockHeader);
    return return_index;
}

__device__ void cfree(void* addressForDeletion){
    uint threadIndex = get_linear_thread_index();


    if(addressForDeletion == NULL){
        return;
    }

    if(memPools[threadIndex].fullList == NULL){
        return;
    }

    RBTreeBlockHeader* deletion_target = (RBTreeBlockHeader*)((unsigned char*)addressForDeletion - sizeof(RBTreeBlockHeader));
    
    deletion_target->fullSize &= 0x7FFF;

    int8_t forward_coalesce_valid = 0;
    int8_t backward_coalesce_valid = 0;

    RBTreeBlockHeader* prevHeader = get_prev_header(deletion_target);
    RBTreeBlockHeader* nextHeader = get_next_header(deletion_target);

    if(prevHeader != NULL && !get_full(prevHeader)){
        backward_coalesce_valid = 1;
    }
    
    if(nextHeader != NULL && !get_full(nextHeader)){
        forward_coalesce_valid = 1;
    }

    // Forward coalesce
    if(forward_coalesce_valid){
        int16_t tempSize = get_node_size(deletion_target);
        tempSize += sizeof(BlockFooter) + sizeof(RBTreeBlockHeader) + get_node_size(nextHeader);
        
        deletion_target->fullSize = tempSize & 0x7FFF;

        BlockFooter* next_block_footer = get_footer(deletion_target);
        next_block_footer->headerOffset = get_offset((unsigned char*)next_block_footer, (unsigned char*)deletion_target);

        // list_remove(&memPools[threadIndex].freeList, nextHeader);
        remove(&memPools[threadIndex].freeList,nextHeader);
    }

    // Backward coalesce
    if(backward_coalesce_valid){
        // list_remove(&memPools[threadIndex].freeList, prevHeader);
        remove(&memPools[threadIndex].freeList, prevHeader);
        int16_t tempSize = get_node_size(prevHeader);

        tempSize += sizeof(BlockFooter) + sizeof(RBTreeBlockHeader) + get_node_size(deletion_target);
        prevHeader->fullSize = tempSize & 0x7FFF;

        
        BlockFooter* current_block_footer = get_footer(prevHeader);
        current_block_footer->headerOffset = get_offset((unsigned char*)current_block_footer, (unsigned char*)prevHeader);
        
        // list_push_front(&memPools[threadIndex].freeList, prevHeader);
        insert_node(&memPools[threadIndex].freeList, prevHeader, tempSize);
    } else {
        insert_node(&memPools[threadIndex].freeList, deletion_target, deletion_target->fullSize & 0x3FFF);
        // list_push_front(&memPools[threadIndex].freeList, deletion_target);
    }

    // sort_free_list(threadIndex);
}

} // namespace pmalloc_bst

//#endif
