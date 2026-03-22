//#ifndef POOLALLOC_CU
//#define POOLALLOC_CU

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "poolAlloc.cuh"

namespace list_pool {

// #define GPU_MEM_POOL_SIZE 49152

#define MEM_MAX_THREADS 128


struct poolMetaData{
    uint sharedMemSize;
    uint16_t freeListOffset;
    uint16_t _padding;
};

// __device__ uint sharedMemSize;

//------------------------------------------------------------------------------------CUDA HELPER FUNCTIONS--------------------------------------------------------------------------------------------

__device__ uint get_shared_mem_size(){
    extern __shared__ unsigned char smem[];
    return ((poolMetaData*)smem)->sharedMemSize;
}

__device__  uint get_thread_pool_size(uint sharedMemSize){
    int total_threads =
        gridDim.x * gridDim.y * gridDim.z *
        blockDim.x * blockDim.y * blockDim.z;

    return (sharedMemSize/total_threads) & ~7u;
}

__device__  uint get_mem_buffer_size(uint sharedMemSize){
    int total_threads =
        gridDim.x * gridDim.y * gridDim.z *
        blockDim.x * blockDim.y * blockDim.z;

    return ((sharedMemSize/total_threads) & ~7u) - sizeof(poolMetaData);
}

__device__  uint get_linear_thread_index(){
    return (((blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x) + blockIdx.x)
    * (blockDim.x * blockDim.y * blockDim.z)
    +
    (threadIdx.z * blockDim.y * blockDim.x +
     threadIdx.y * blockDim.x +
     threadIdx.x);
}

__device__  unsigned char* get_thread_pool_start(uint threadIndex, uint sharedMemSize){
    extern __shared__ unsigned char smem[];
    return smem + (get_thread_pool_size(sharedMemSize) * threadIndex);
}


__device__  unsigned char* get_mem_buffer_start(uint threadIndex, uint sharedMemSize){
    return get_thread_pool_start(threadIndex, sharedMemSize) + sizeof(poolMetaData);
}

__device__ poolMetaData* get_pool_metadata(uint threadIndex, uint sharedMemSize){
    return (poolMetaData*)get_thread_pool_start(threadIndex, sharedMemSize);
}



__device__  BlockHeader** get_free_list_ptr(uint threadIndex, uint sharedMemSize){
    return (BlockHeader**)get_thread_pool_start(threadIndex, sharedMemSize);
}

__device__  BlockHeader* get_free_list(uint threadIndex, uint sharedMemSize){
    poolMetaData* tempMetaData = get_pool_metadata(threadIndex, sharedMemSize);
    uint16_t offset = tempMetaData->freeListOffset;
    if(offset == 0xFFFF) return NULL;

    return (BlockHeader*)(get_mem_buffer_start(threadIndex, sharedMemSize) + offset);
}

__device__  void set_free_list(uint threadIndex, uint sharedMemSize, BlockHeader* header){
    if(header == NULL){
        get_pool_metadata(threadIndex, sharedMemSize)->freeListOffset = 0xFFFF;
        return;
    }
    // dont want to mess with sign extension so dont use get_offset() here
    get_pool_metadata(threadIndex, sharedMemSize)->freeListOffset = (uint16_t)((unsigned char*)header - get_mem_buffer_start(threadIndex, sharedMemSize));
}


//------------------------------------------------------------------------------------HELPER FUNCTIONS TO DEAL WITH THE STRUCT--------------------------------------------------------------------------------------------

__device__ void set_full(BlockHeader* currentHeader){
    currentHeader->fullSize |= 0x8000;
}

__device__ void set_empty(BlockHeader* currentHeader){
    currentHeader->fullSize &= 0x7FFF;
}

__device__ int8_t get_full(BlockHeader* currentHeader){
    return currentHeader->fullSize < 0;
}

//the size and full used to be packed into an int16 hence the helper functions
__device__  uint32_t get_node_size(BlockHeader* currentHeader){
    return ((uint32_t)(currentHeader->fullSize & 0x7FFF) * 8);
}

__device__  int32_t get_node_prevAdjOffset(BlockHeader* currentHeader){
    return((int32_t)currentHeader->prevAdjOffset * 8);
}

__device__  int32_t get_node_nextListOffset(BlockHeader* currentHeader){
    return ((int32_t)currentHeader->nextListOffset * 8);
}

__device__  int32_t get_node_prevListOffset(BlockHeader* currentHeader){
    return ((int32_t)currentHeader->prevListOffset * 8);
}
__device__  void set_node_size(BlockHeader* currentHeader, uint16_t incomingSize){
    currentHeader->fullSize = ((incomingSize / 8) & 0x7FFF) | (currentHeader->fullSize & 0x8000);
}

__device__  void set_node_nextListOffset(BlockHeader* currentHeader, int32_t incomingOffset){
    currentHeader->nextListOffset = (int16_t)(incomingOffset / 8);
}

__device__  void set_node_prevListOffset(BlockHeader* currentHeader, int32_t incomingOffset){
    currentHeader->prevListOffset = (int16_t)(incomingOffset / 8);
}

__device__  void set_node_prevAdjOffset(BlockHeader* currentHeader, int32_t incomingOffset){
    currentHeader->prevAdjOffset = (int16_t)(incomingOffset / 8);
}
//these are all relatively simple helper functions to make the pointer math easier
__device__  int32_t get_offset(unsigned char* from, unsigned char* to){
    return (unsigned char*)to - (unsigned char*)from;
}

//gets the previous header from the total memPool
__device__ BlockHeader* get_prev_header(BlockHeader* currentHeader, uint sharedMemSize){

    uint threadIndex = get_linear_thread_index();


    if((void*)currentHeader == (void*)get_mem_buffer_start(threadIndex, sharedMemSize)){
        return NULL;
    }

    BlockHeader* prevHeader = (BlockHeader*)((unsigned char*)currentHeader + get_node_prevAdjOffset(currentHeader));
    // BlockHeader* prevHeader = (BlockHeader*)((unsigned char*)currentHeader + currentHeader->prevAdjOffset);
    return prevHeader;
}

//gets the next header from the total memPool
__device__ BlockHeader* get_next_header(BlockHeader* currentHeader, uint sharedMemSize){

    uint threadIndex = get_linear_thread_index();

    uint memBufferSize = get_mem_buffer_size(sharedMemSize);

    int currentHeaderOffset = get_offset(get_mem_buffer_start(threadIndex, sharedMemSize), (unsigned char*)currentHeader);
    uint64_t temp_size = get_node_size(currentHeader);

    if((currentHeaderOffset + temp_size + (2*(sizeof(BlockHeader)))) >= memBufferSize){
        return NULL;
    }

    BlockHeader* nextHeader = (BlockHeader*)((unsigned char*)currentHeader + temp_size +  sizeof(BlockHeader));
    return nextHeader;
}

//gets the next header from the list it current resides in
__device__  BlockHeader* get_next_list_header(BlockHeader* currentHeader){
    uint32_t offset = get_node_nextListOffset(currentHeader);
    if(offset == 0){
        return NULL;
    }

    return (BlockHeader*)((unsigned char*)currentHeader + offset);
}

//gets the previous header from the list it current resides in
__device__  BlockHeader* get_prev_list_header(BlockHeader* currentHeader){
    uint32_t offset = get_node_prevListOffset(currentHeader);

    if(offset == 0){
        return NULL;
    }

    return (BlockHeader*)((unsigned char*)currentHeader + offset);
}

__device__ void list_push_ordered(BlockHeader* insertionNode, uint sharedMemSize){

    uint threadIndex = get_linear_thread_index();

    if(get_free_list(threadIndex, sharedMemSize) == NULL){
        set_free_list(threadIndex, sharedMemSize, insertionNode);
        set_node_nextListOffset(insertionNode, 0);
        set_node_prevListOffset(insertionNode, 0);

        return;
    }


    BlockHeader* current = get_free_list(threadIndex, sharedMemSize);


    while(get_next_list_header(current) != NULL){
        if(get_node_size(current) < get_node_size(insertionNode)){
            break; // insert before current
        }
        current = get_next_list_header(current);
    }



    if(get_node_size(current) < get_node_size(insertionNode)){
        BlockHeader* prev = get_prev_list_header(current);
        set_node_nextListOffset(insertionNode, get_offset((unsigned char*)insertionNode, (unsigned char*)current));
        set_node_prevListOffset(insertionNode, (prev != NULL) ? get_offset((unsigned char*)insertionNode, (unsigned char*)prev) : 0);

        set_node_prevListOffset(current, get_offset((unsigned char*)current, (unsigned char*)insertionNode));

        if(prev != NULL)
            set_node_nextListOffset(prev, get_offset((unsigned char*)prev, (unsigned char*)insertionNode));

        else

            set_free_list(threadIndex, sharedMemSize, insertionNode); // new head
    } else {
        set_node_prevListOffset(insertionNode, get_offset((unsigned char*)insertionNode, (unsigned char*)current));
        set_node_nextListOffset(insertionNode, 0);

        set_node_nextListOffset(current, get_offset((unsigned char*)current, (unsigned char*)insertionNode));

    }




}

__device__ void list_remove(BlockHeader* deletionNode, uint sharedMemSize){
    uint threadIndex = get_linear_thread_index();

    BlockHeader* currentNodeNext = NULL;
    BlockHeader* currentNode_prev = NULL;

    int32_t nextListOffset = get_node_nextListOffset(deletionNode);
    int32_t prevListOffset = get_node_prevListOffset(deletionNode);

    if(nextListOffset != 0){
        currentNodeNext = (BlockHeader*)((unsigned char*)deletionNode + nextListOffset);
    }

    if(prevListOffset != 0){
        currentNode_prev = (BlockHeader*)((unsigned char*)deletionNode + prevListOffset);
    }

    // Fix the previous node's next pointer
    if(currentNode_prev != NULL){
        if(currentNodeNext != NULL){
            set_node_nextListOffset(currentNode_prev, get_offset((unsigned char*)currentNode_prev, (unsigned char*)currentNodeNext));
        } else {
            set_node_nextListOffset(currentNode_prev, 0);
        }
    } else {
        // deletionNode was the head of the list
        set_free_list(threadIndex, sharedMemSize, currentNodeNext);
    }

    // Fix the next node's prev pointer
    if(currentNodeNext != NULL){
        if(currentNode_prev != NULL){
            set_node_prevListOffset(currentNodeNext, get_offset((unsigned char*)currentNodeNext, (unsigned char*)currentNode_prev));

        } else {
            set_node_prevListOffset(currentNodeNext, 0);

        }
    }

    // Clear the deleted node's pointers
    set_node_prevListOffset(deletionNode, 0);
    set_node_nextListOffset(deletionNode, 0);
}


//------------------------------------------------------------------------------------DEBUG FUNCTIONS--------------------------------------------------------------------------------------------


//used solely for debugging help and correctness checking
__device__ void debug_print_buffer(){

    uint threadIndex = get_linear_thread_index();


    printf("\n|");
    BlockHeader* currentNode = (BlockHeader*)get_mem_buffer_start(threadIndex, get_shared_mem_size());
    while (currentNode != NULL){
        printf(" %4u |", get_node_size(currentNode));
        currentNode = get_next_header(currentNode, get_shared_mem_size());
    }
    printf("\n|");
    currentNode = (BlockHeader*)get_mem_buffer_start(threadIndex, get_shared_mem_size());

    while (currentNode != NULL){
        if(get_full(currentNode)){
            printf(" FULL |");
        } else {
            printf(" FREE |");
        }
        currentNode = get_next_header(currentNode, get_shared_mem_size());
    }
    printf(" \n\n");
}


__device__ void debug_print_free_list(){

    uint threadIndex = get_linear_thread_index();
    uint sharedMemSize = get_shared_mem_size();
    printf("\n|");
    BlockHeader* currentNode = get_free_list(threadIndex, sharedMemSize);
    while (currentNode != NULL){
        printf(" %4u |", get_node_size(currentNode));
        currentNode = get_next_list_header(currentNode);
    }
    printf("\n|");
    currentNode = get_free_list(threadIndex, sharedMemSize);

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




__device__ void pool_init(uint incomingMemSize){



    extern __shared__ unsigned char smem[];

    uint threadIndex = get_linear_thread_index();

    /*

    Because all operations use an offset based scheme to find neighbors,
    it is important to be very careful about the size of these the pool per thread.

    At most it is acceptable to have 2^15 bytes,
    however past that the bit packing will be upset and the size of the offset variables will need to be upgraded

    This is particularly important if a kernel is launched with few threads but large shared memory

    */


    uint threadPoolSize = get_thread_pool_size(incomingMemSize);

    poolMetaData* tempMetaData = (poolMetaData*)get_thread_pool_start(threadIndex, incomingMemSize);
    tempMetaData->sharedMemSize = incomingMemSize;

    set_free_list(threadIndex, incomingMemSize, NULL);


    BlockHeader *header = (BlockHeader*)get_mem_buffer_start(threadIndex, incomingMemSize);
    set_node_size(header, (get_mem_buffer_size(incomingMemSize) - (sizeof(BlockHeader))));
    set_empty(header);
    set_node_nextListOffset(header,0);
    set_node_prevListOffset(header,0);

    set_node_prevAdjOffset(header,0);


    list_push_ordered(header, incomingMemSize);

}

__device__ void* pmalloc(unsigned long size){


    uint threadIndex = get_linear_thread_index();
    uint sharedMemSize = get_shared_mem_size();
    bool haveSpaceForNextBlock = false;

    BlockHeader* freeList = get_free_list(threadIndex, sharedMemSize);
    if(freeList == NULL || get_node_size(freeList) < (size)){
        return NULL;
    }


    uint32_t preAllocationSize = get_node_size(freeList);
    BlockHeader* newlyAllocatedHeader = freeList;

    // Align size to 8 bytes
    int offset = (8 - ((sizeof(BlockHeader) + size) % 8)) % 8;
    int sizeToAlloc = size + offset;

    // Create new free block if there's enough space
    int remainingSize = preAllocationSize - (sizeToAlloc + sizeof(BlockHeader));
    if(remainingSize > 0){
        haveSpaceForNextBlock = true;
    }
    else{
        sizeToAlloc = preAllocationSize;
    }

    set_node_size(newlyAllocatedHeader, sizeToAlloc);
    set_full(newlyAllocatedHeader);

    set_node_prevAdjOffset(newlyAllocatedHeader, get_node_prevAdjOffset(freeList));

    // Remove from free list before modifying
    list_remove(newlyAllocatedHeader, sharedMemSize);



    if(haveSpaceForNextBlock){
        BlockHeader* newFreeHeader = (BlockHeader*)((unsigned char*)(newlyAllocatedHeader) + sizeof(BlockHeader) + sizeToAlloc);
        set_node_size(newFreeHeader, remainingSize);
        set_empty(newFreeHeader);
        set_node_nextListOffset(newFreeHeader, 0);

        set_node_prevListOffset(newFreeHeader, 0);

        set_node_prevAdjOffset(newFreeHeader, get_offset((unsigned char*)newFreeHeader, (unsigned char*)newlyAllocatedHeader));



        list_push_ordered(newFreeHeader, sharedMemSize);

    }



    void* return_index = (unsigned char*)newlyAllocatedHeader + sizeof(BlockHeader);
    return return_index;
}

__device__ void pfree(void* addressForDeletion){
    uint threadIndex = get_linear_thread_index();
    uint sharedMemSize = get_shared_mem_size();


    if(addressForDeletion == NULL){
        return;
    }


    BlockHeader* deletion_target = (BlockHeader*)((unsigned char*)addressForDeletion - sizeof(BlockHeader));

    set_empty(deletion_target);

    bool forward_coalesce_valid = false;
    bool backward_coalesce_valid = false;

    BlockHeader* prevHeader = get_prev_header(deletion_target, sharedMemSize);
    BlockHeader* nextHeader = get_next_header(deletion_target, sharedMemSize);


    if(prevHeader != NULL && !get_full(prevHeader)){
        backward_coalesce_valid = true;
    }

    if(nextHeader != NULL && !get_full(nextHeader)){
        forward_coalesce_valid = true;
    }

    // Forward coalesce
    if(forward_coalesce_valid){

        list_remove(nextHeader, sharedMemSize);

        uint32_t tempSize = get_node_size(deletion_target);
        tempSize += sizeof(BlockHeader) + get_node_size(nextHeader);

        set_node_size(deletion_target, tempSize);
        set_empty(deletion_target);

        BlockHeader* tempNextHeader = get_next_header(deletion_target, sharedMemSize);
        if(tempNextHeader != NULL){
            set_node_prevAdjOffset(tempNextHeader, get_offset((unsigned char*)tempNextHeader, (unsigned char*)deletion_target));
        }

    }


    // Backward coalesce
    if(backward_coalesce_valid){

        list_remove(prevHeader, sharedMemSize);

        uint32_t tempSize = get_node_size(prevHeader);

        tempSize += sizeof(BlockHeader) + get_node_size(deletion_target);
        set_node_size(prevHeader, tempSize);
        set_empty(prevHeader);

        BlockHeader* tempNextHeader = get_next_header(deletion_target, sharedMemSize);
        if(tempNextHeader != NULL){
            set_node_prevAdjOffset(tempNextHeader, get_offset((unsigned char*)tempNextHeader, (unsigned char*)prevHeader));
        }

        list_push_ordered(prevHeader, sharedMemSize);
    } else {

        list_push_ordered(deletion_target, sharedMemSize);
    }


}

} // namespace pmalloc_freelist

//#endif
