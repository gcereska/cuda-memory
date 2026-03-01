//#ifndef POOLALLOC_CU
//#define POOLALLOC_CU

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "poolAlloc.cuh"

namespace pmalloc_freelist {

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


__device__  unsigned char* get_mem_buffer_start(uint threadIndex){
    return get_thread_pool_start(threadIndex, get_shared_mem_size()) + sizeof(poolMetaData);
}

__device__ poolMetaData* get_pool_metadata(uint threadIndex){
    return (poolMetaData*)get_thread_pool_start(threadIndex, get_shared_mem_size());
}



__device__  BlockHeader** get_free_list_ptr(uint threadIndex){
    return (BlockHeader**)get_thread_pool_start(threadIndex, get_shared_mem_size());
}

__device__  BlockHeader* get_free_list(uint threadIndex){
    uint16_t offset = get_pool_metadata(threadIndex)->freeListOffset;
    if(offset == 0xFFFF) return NULL;

    return (BlockHeader*)(get_mem_buffer_start(threadIndex) + offset);
}

__device__  void set_free_list(uint threadIndex, BlockHeader* header){
    if(header == NULL){
        get_pool_metadata(threadIndex)->freeListOffset = 0xFFFF;
        return;
    }
    // dont want to mess with sign extension so dont use get_offset() here
    get_pool_metadata(threadIndex)->freeListOffset = (uint16_t)((unsigned char*)header - get_mem_buffer_start(threadIndex));
}


//------------------------------------------------------------------------------------HELPER FUNCTIONS TO DEAL WITH THE STRUCT--------------------------------------------------------------------------------------------


__device__ int8_t get_full(BlockHeader* currentHeader){
    return currentHeader->fullSize < 0;
}

//the size and full used to be packed into an int16 hence the helper functions
__device__  int16_t get_node_size(BlockHeader* currentHeader){
    return (int16_t)(currentHeader->fullSize & 0x7FFF);
}

//these are all relatively simple helper functions to make the pointer math easier
__device__  int16_t get_offset(unsigned char* from, unsigned char* to){
    return (unsigned char*)to - (unsigned char*)from;
}

//gets the previous header from the total memPool
__device__ BlockHeader* get_prev_header(BlockHeader* currentHeader){

    uint threadIndex = get_linear_thread_index();


    if((void*)currentHeader == (void*)get_mem_buffer_start(threadIndex)){
        return NULL; 
    }

    
    BlockHeader* prevHeader = (BlockHeader*)((unsigned char*)currentHeader + currentHeader->prevAdjOffset);
    return prevHeader;
}

//gets the next header from the total memPool
__device__ BlockHeader* get_next_header(BlockHeader* currentHeader){

    uint threadIndex = get_linear_thread_index();

    
    uint threadPoolSize = get_thread_pool_size(get_shared_mem_size()); 
    
    int currentHeaderOffset = get_offset(get_mem_buffer_start(threadIndex), (unsigned char*)currentHeader);
    uint64_t temp_size = get_node_size(currentHeader);
    
    if((currentHeaderOffset + temp_size + (2*(sizeof(BlockHeader)))) >= threadPoolSize){
        return NULL;
    }
    
    BlockHeader* nextHeader = (BlockHeader*)((unsigned char*)currentHeader + temp_size +  sizeof(BlockHeader));
    return nextHeader;
}

//gets the next header from the list it current resides in
__device__  BlockHeader* get_next_list_header(BlockHeader* currentHeader){
    if(currentHeader->nextListOffset == 0){
        return NULL;
    }
    
    return (BlockHeader*)((unsigned char*)currentHeader + currentHeader->nextListOffset);
}

//gets the previous header from the list it current resides in
__device__  BlockHeader* get_prev_list_header(BlockHeader* currentHeader){
    if(currentHeader->prevListOffset == 0){
        return NULL;
    }
    
    return (BlockHeader*)((unsigned char*)currentHeader + currentHeader->prevListOffset);
}

__device__ void list_push_ordered(BlockHeader* insertionNode){
    
    uint threadIndex = get_linear_thread_index();


    if(get_free_list(threadIndex) == NULL){
        set_free_list(threadIndex, insertionNode);
        insertionNode->nextListOffset = 0;
        insertionNode->prevListOffset = 0;
        return;
    }
    
    
    BlockHeader* current = get_free_list(threadIndex);
    
 
    while(get_next_list_header(current) != NULL){
        if(get_node_size(current) < get_node_size(insertionNode)){
            break; // insert before current
        }
        current = get_next_list_header(current);
    }

   
    
    if(get_node_size(current) < get_node_size(insertionNode)){
        BlockHeader* prev = get_prev_list_header(current);
        insertionNode->nextListOffset = get_offset((unsigned char*)insertionNode, (unsigned char*)current);
        insertionNode->prevListOffset = (prev != NULL) ? get_offset((unsigned char*)insertionNode, (unsigned char*)prev) : 0;
        current->prevListOffset = get_offset((unsigned char*)current, (unsigned char*)insertionNode);
        if(prev != NULL)
            prev->nextListOffset = get_offset((unsigned char*)prev, (unsigned char*)insertionNode);
        else
        
            set_free_list(threadIndex, insertionNode); // new head
    } else {
        insertionNode->prevListOffset = get_offset((unsigned char*)insertionNode, (unsigned char*)current);
        insertionNode->nextListOffset = 0;
        current->nextListOffset = get_offset((unsigned char*)current, (unsigned char*)insertionNode);
    }

   
    

}

__device__ void list_remove(BlockHeader* deletionNode){
    uint threadIndex = get_linear_thread_index();

    BlockHeader* currentNodeNext = NULL;
    BlockHeader* currentNode_prev = NULL;
    
    if(deletionNode->nextListOffset != 0){
        currentNodeNext = (BlockHeader*)((unsigned char*)deletionNode + deletionNode->nextListOffset);
    }
    
    if(deletionNode->prevListOffset != 0){
        currentNode_prev = (BlockHeader*)((unsigned char*)deletionNode + deletionNode->prevListOffset);
    }
    
    // Fix the previous node's next pointer
    if(currentNode_prev != NULL){
        if(currentNodeNext != NULL){
            currentNode_prev->nextListOffset = get_offset((unsigned char*)currentNode_prev, (unsigned char*)currentNodeNext);
        } else {
            currentNode_prev->nextListOffset = 0;
        }
    } else {
        // deletionNode was the head of the list
        set_free_list(threadIndex, currentNodeNext);
    }
    
    // Fix the next node's prev pointer
    if(currentNodeNext != NULL){
        if(currentNode_prev != NULL){
            currentNodeNext->prevListOffset = get_offset((unsigned char*)currentNodeNext, (unsigned char*)currentNode_prev);
        } else {
            currentNodeNext->prevListOffset = 0;
        }
    }
    
    // Clear the deleted node's pointers
    deletionNode->nextListOffset = 0;
    deletionNode->prevListOffset = 0;
}


//------------------------------------------------------------------------------------DEBUG FUNCTIONS--------------------------------------------------------------------------------------------


//used solely for debugging help and correctness checking
__device__ void debug_print_buffer(){

    uint threadIndex = get_linear_thread_index();
    

    printf("\n|");
    BlockHeader* currentNode = (BlockHeader*)get_mem_buffer_start(threadIndex);
    while (currentNode != NULL){
        printf(" %4u |", get_node_size(currentNode));
        currentNode = get_next_header(currentNode);
    }
    printf("\n|");
    currentNode = (BlockHeader*)get_mem_buffer_start(threadIndex);

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


__device__ void debug_print_free_list(){

    uint threadIndex = get_linear_thread_index();


    printf("\n|");
    BlockHeader* currentNode = get_free_list(threadIndex);
    while (currentNode != NULL){
        printf(" %4u |", get_node_size(currentNode));
        currentNode = get_next_list_header(currentNode);
    }
    printf("\n|");
    currentNode = get_free_list(threadIndex);

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




__device__ void init_gpu_buffer(uint incomingMemSize){

    

    extern __shared__ unsigned char smem[];

    uint threadIndex = get_linear_thread_index();
    // printf("linear thread index: %d\n",threadIndex);
    
    /*
    
    Because all operations use an offset based scheme to find neighbors,
    it is important to be very careful about the size of these the pool per thread. 

    At most it is acceptable to have 2^15 bytes,
    however past that the bit packing will be upset and the size of the offset variables will need to be upgraded

    This is particularly important if a kernel is launched with few threads but large shared memory
    
    */

   
    uint threadPoolSize = get_thread_pool_size(incomingMemSize);

    // memPools[threadIndex].memBuffer = smem + (threadPoolSize * threadIndex);
    // memPools[threadIndex].freeList = NULL;

    // if(threadIndex == 0){
    //     // printf("thread index %d's memory pool pointer: %p\n",threadIndex,memPools[threadIndex].memBuffer);
    //     printf("thread index %d's new memory pool pointer: %p\n", threadIndex, get_thread_pool_start());
    //     printf("thread index %d's new buffer pool pointer: %p\n", threadIndex, get_mem_buffer_start());
    //     printf("freelist pointer: %p\n", get_free_list());

    // }
   

    // printf("mempool start pointer: %p\n", memPool.memBuffer);
    poolMetaData* tempMetaData = (poolMetaData*)get_thread_pool_start(threadIndex, incomingMemSize);
    tempMetaData->sharedMemSize = incomingMemSize;

    set_free_list(threadIndex, NULL);


    BlockHeader *header = (BlockHeader*)get_mem_buffer_start(threadIndex);
    header->fullSize = (get_mem_buffer_size(incomingMemSize) - (sizeof(BlockHeader) )) & 0x7FFF;
    header->nextListOffset = 0;
    header->prevListOffset = 0;
    header->prevAdjOffset = 0;    

    list_push_ordered(header);
    
}

__device__ void* cmalloc(unsigned long size){


    uint threadIndex = get_linear_thread_index();
    bool haveSpaceForNextBlock = false;
   

    if(get_free_list(threadIndex) == NULL || get_node_size(get_free_list(threadIndex)) < (size)){
        // printf("malloc eval null exit\n");
        return NULL;
    }

    
    BlockHeader* freeList = get_free_list(threadIndex);
    int preAllocationSize = get_node_size(freeList);
    BlockHeader* newlyAllocatedHeader = get_free_list(threadIndex);

    // Align size to 8 bytes
    int offset = (8 - ((sizeof(BlockHeader) + size) % 8)) % 8;
    int sizeToAlloc = size + offset;

    // Create new free block if there's enough space
    int remainingSize = preAllocationSize - (sizeToAlloc + sizeof(BlockHeader) );
    if(remainingSize > 0){
        haveSpaceForNextBlock = true;
    }
    else{
        sizeToAlloc = preAllocationSize;
    }
    
    newlyAllocatedHeader->fullSize = sizeToAlloc & 0x7FFF;
    newlyAllocatedHeader->fullSize |= 0x8000; //sets full

    newlyAllocatedHeader->prevAdjOffset = freeList->prevAdjOffset;

    // Remove from free list before modifying
    list_remove(newlyAllocatedHeader);
  
    
    
    if(haveSpaceForNextBlock){
        BlockHeader* newFreeHeader = (BlockHeader*)((unsigned char*)(newlyAllocatedHeader) + sizeof(BlockHeader) + sizeToAlloc);
       
        newFreeHeader->fullSize = remainingSize & 0x7FFF; //both sets size and sets empty
        newFreeHeader->nextListOffset = 0;
        newFreeHeader->prevListOffset = 0;
       
        newFreeHeader->prevAdjOffset = get_offset((unsigned char*)newFreeHeader, (unsigned char*)newlyAllocatedHeader);


        list_push_ordered(newFreeHeader);
        
    }

   

    void* return_index = (unsigned char*)newlyAllocatedHeader + sizeof(BlockHeader);
    return return_index;
}

__device__ void cfree(void* addressForDeletion){
    uint threadIndex = get_linear_thread_index();

 

    if(addressForDeletion == NULL){
        return;
    }

   
    BlockHeader* deletion_target = (BlockHeader*)((unsigned char*)addressForDeletion - sizeof(BlockHeader));
    // printf("\nTarget for deletion %p\n", deletion_target);
    
    deletion_target->fullSize &= 0x7FFF;

    bool forward_coalesce_valid = false;
    bool backward_coalesce_valid = false;

    BlockHeader* prevHeader = get_prev_header(deletion_target);
    BlockHeader* nextHeader = get_next_header(deletion_target);
    
    
    if(prevHeader != NULL && !get_full(prevHeader)){
        backward_coalesce_valid = true;
    }
    
    if(nextHeader != NULL && !get_full(nextHeader)){
        forward_coalesce_valid = true;
    }

    // Forward coalesce
    if(forward_coalesce_valid){
        
        list_remove(nextHeader);

        int16_t tempSize = get_node_size(deletion_target);
        tempSize += sizeof(BlockHeader) + get_node_size(nextHeader);
        
        deletion_target->fullSize = tempSize & 0x7FFF;

        BlockHeader* tempNextHeader = get_next_header(deletion_target);
        tempNextHeader->prevAdjOffset = get_offset((unsigned char*)tempNextHeader, (unsigned char*)deletion_target);
        // printf("\nTarget for deletion %p\n", nextHeader);
        
    }

   
    // Backward coalesce
    if(backward_coalesce_valid){
        
        list_remove(prevHeader);
        
        int16_t tempSize = get_node_size(prevHeader);

        tempSize += sizeof(BlockHeader) + get_node_size(deletion_target);
        prevHeader->fullSize = tempSize & 0x7FFF;
        
        BlockHeader* tempNextHeader = get_next_header(deletion_target);
        tempNextHeader->prevAdjOffset = get_offset((unsigned char*)tempNextHeader, (unsigned char*)prevHeader);
        // printf("\nTarget for deletion %p\n", prevHeader);
        
        list_push_ordered(prevHeader);
    } else {
        
        list_push_ordered(deletion_target);
    }


}

} // namespace pmalloc_freelist

//#endif
