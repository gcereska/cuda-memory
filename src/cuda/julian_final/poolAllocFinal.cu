#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "RedBlackTree.cuh"
#include "RBTreeBlockHeader.cuh"

#define MEM_MAX_THREADS 128

// Forward declarations
class BlockHeader;
class BlockFooter;
class RedBlackTree;
class OrderedLinkedList;

class MemBufferStorage{
    private:
        __device__ unsigned char* memBuffer;

    public:
        __device__ RedBlackTree freeList;
        __device__ OrderedLinkedList fullList;
        // Constructor
        // Default Constructor
        __device__ MemBufferStorage(): memBuffer(nullptr), freeList(), fullList(){}

        // Constructor with buffer parameter
        __device__ MemBufferStorage(unsigned char* buffer) 
            : memBuffer(buffer), 
              freeList(),//these need to be initialized properly, namely the free list needs to start with the base of the mem buffer as well
              fullList()
        {
        }

        __device__ unsigned char* base() {
            return memBuffer;
        }
        
        __device__ void setMemBuffer(unsigned char* input){
            memBuffer = input;
        }

        


};

__device__ MemBufferStorage memPools[MEM_MAX_THREADS];

__device__ uint sharedMemSize; 

__device__ uint get_thread_pool_size(){
    return (sharedMemSize);
}

__device__ uint get_linear_thread_index(){
    return (((blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x) + blockIdx.x)
    * (blockDim.x * blockDim.y * blockDim.z)
    +
    (threadIdx.z * blockDim.y * blockDim.x +
     threadIdx.y * blockDim.x +
     threadIdx.x);
}

class BlockHeader {
    /*
    first bit will be full status, second bit will be color 
    typedef struct{
        int16_t fullStatus : 1
        int16_t color : 1
        int16_t size : 14
    } fullSize
    */
    private:
        int16_t fullSize;
        int16_t nextOffset;
        int16_t prevOffset;
        int16_t padding;

    public:
        BlockHeader(int16_t fullSize,int16_t nextOffset, int16_t prevOffset)
            :fullSize(fullSize), nextOffset(nextOffset),prevOffset(prevOffset),padding(0){}  


        __device__ int16_t size() {
            return fullSize & 0x7FFF;
        }

        __device__ bool full() {
            return fullSize < 0;
        }

        __device__ int16_t calculateOffset(unsigned char* to) { 
            return to - (unsigned char*)this;
        }

        __device__ BlockFooter* footer() {
            return (BlockFooter*)((unsigned char*)this + size() + sizeof(BlockHeader));
        }

        __device__ BlockFooter* prevFooter() {
            return (BlockFooter*)((unsigned char*)this - sizeof(BlockFooter));
        }

        __device__ BlockHeader* debugPrevHeader() {
            uint threadIndex = get_linear_thread_index();

            if((void*)this == (void*)memPools[threadIndex].base() ){
                return NULL; 
            }

            BlockFooter* previousFooter = prevFooter();
            BlockHeader* prevHeader = previousFooter->ListHeader();

            return prevHeader;
        }

        __device__ BlockHeader* debugNextHeader(){
            uint threadIndex = get_linear_thread_index();
            uint threadPoolSize = get_thread_pool_size(); 
            
            int currentHeaderOffset = calculateOffset(memPools[threadIndex].base());

            if((currentHeaderOffset + size() + sizeof(BlockHeader) + sizeof(BlockFooter)) >= threadPoolSize){
                return NULL;
            }
            
            return (BlockHeader*)((unsigned char*)this + size() + sizeof(BlockFooter) + sizeof(BlockHeader));
        }

        __device__ BlockHeader* nextHeader(){
            if(nextOffset == 0){
                return NULL;
            }
            return (BlockHeader*)((unsigned char*)this + nextOffset);
        }

        __device__ BlockHeader* prevHeader(){
            if(prevOffset == 0){
                return NULL;
            }
            return (BlockHeader*)((unsigned char*)this + prevOffset);
        }    

        __device__ void setNextOffset(BlockHeader* input) {
            nextOffset = input ? calculateOffset((unsigned char*)input) : 0;
        }

        __device__ void setPrevOffset(BlockHeader* input) {
            prevOffset = input ? calculateOffset((unsigned char*)input) : 0;
        }

};

class BlockFooter {
    private:    
        int16_t headerOffset;

    public: 

        __device__ BlockFooter(RBTreeBlockHeader* header)
            :headerOffset((char*)this - (char*)header){}

        int16_t headerOffset(){
            return headerOffset;
        } 

        __device__ void setOffset(RBTreeBlockHeader* header){
            headerOffset = (char*)this - (char*)header;
        }

        __device__ RBTreeBlockHeader* RBTreeheader(){
            return (RBTreeBlockHeader*)((unsigned char*)this + headerOffset);
        }

        __device__ BlockHeader* ListHeader(){
            return (BlockHeader*)((unsigned char*)this + headerOffset);
        }
};



// Ordered Linked List - Largest blocks at the head
class OrderedLinkedList {
private:
    __device__ BlockHeader* head; // Head of the list (largest block)
    __device__ BlockHeader* tail; // Tail of the list (smallest block)

public:
    // Constructor
    __device__ OrderedLinkedList()
        : head(nullptr), tail(nullptr)
    {
    }

    // Get the head of the list
    __device__ BlockHeader* getHead() {
        return head;
    }

    // Get the tail of the list
    __device__ BlockHeader* getTail() {
        return tail;
    }

   

    // Check if list is empty
    __device__ bool isEmpty() {
        return head == nullptr;
    }

    // Insert a block in descending order (largest to smallest)
    __device__ void insert(BlockHeader* node) {
        if (node == nullptr) return;

        // Initialize the node's list pointers
        node->setNextOffset(nullptr);
        node->setPrevOffset(nullptr);

        // If list is empty
        if (head == nullptr) {
            head = node;
            tail = node;
            return;
        }

        int16_t nodeSize = node->size();

        // If new node is larger than or equal to head, insert at front
        if (nodeSize >= head->size()) {
            node->setNextOffset(head);
            head->setPrevOffset(node);
            head = node;
            return;
        }

        // If new node is smaller than tail, insert at end
        if (nodeSize < tail->size()) {
            tail->setNextOffset(node);
            node->setPrevOffset(tail);
            tail = node;
            return;
        }

        // Insert in the middle - find the correct position
        BlockHeader* current = head;
        while (current != nullptr && current->size() > nodeSize) {
            current = current->nextHeader();
        }

        // Insert before current
        BlockHeader* prev = current->prevHeader();
        node->setNextOffset(current);
        node->setPrevOffset(prev);
        
        if (prev != nullptr) {
            prev->setNextOffset(node);
        }
        current->setPrevOffset(node);
        
    }

    // Remove a specific node from the list
    __device__ void remove(BlockHeader* node) {
        if (node == nullptr) return;

        BlockHeader* prev = node->prevHeader();
        BlockHeader* next = node->nextHeader();

        // Update the previous node's next pointer
        if (prev != nullptr) {
            prev->setNextOffset(next);
        } else {
            // This was the head
            head = next;
        }

        // Update the next node's prev pointer
        if (next != nullptr) {
            next->setPrevOffset(prev);
        } else {
            // This was the tail
            tail = prev;
        }

        // Clear the removed node's pointers
        node->setNextOffset(nullptr);
        node->setPrevOffset(nullptr);

    }

    // Remove and return the head (largest block)
    __device__ BlockHeader* removeHead() {
        if (head == nullptr) return nullptr;

        BlockHeader* removed = head;
        head = head->nextHeader();

        if (head != nullptr) {
            head->setPrevOffset(nullptr);
        } else {
            // List is now empty
            tail = nullptr;
        }

        removed->setNextOffset(nullptr);
        removed->setPrevOffset(nullptr);

        return removed;
    }

    // Remove and return the tail (smallest block)
    __device__ BlockHeader* removeTail() {
        if (tail == nullptr) return nullptr;

        BlockHeader* removed = tail;
        tail = tail->prevHeader();

        if (tail != nullptr) {
            tail->setNextOffset(nullptr);
        } else {
            // List is now empty
            head = nullptr;
        }

        removed->setNextOffset(nullptr);
        removed->setPrevOffset(nullptr);

        return removed;
    }

    // Find the first block that is at least the requested size (best fit from largest)
    __device__ BlockHeader* findFirstFit(int16_t requestedSize) {
        BlockHeader* current = head;
        
        while (current != nullptr) {
            if (current->size() >= requestedSize) {
                return current;
            }
            current = current->nextHeader();
        }
        
        return nullptr; // No suitable block found
    }

    // Find the smallest block that fits the requested size (best fit)
    __device__ BlockHeader* findBestFit(int16_t requestedSize) {
        BlockHeader* current = head;
        BlockHeader* bestFit = nullptr;
        
        while (current != nullptr) {
            if (current->size() >= requestedSize) {
                // Since list is ordered largest to smallest,
                // keep going to find the smallest that fits
                bestFit = current;
            } else {
                // Once we hit blocks that are too small, we can stop
                break;
            }
            current = current->nextHeader();
        }
        
        return bestFit;
    }

    // Find a block by exact size
    __device__ BlockHeader* findBySize(int16_t targetSize) {
        BlockHeader* current = head;
        
        while (current != nullptr) {
            if (current->size() == targetSize) {
                return current;
            }
            // Since list is ordered, we can stop early
            if (current->size() < targetSize) {
                break;
            }
            current = current->nextHeader();
        }
        
        return nullptr;
    }

    // Check if a specific node is in the list
    __device__ bool contains(BlockHeader* node) {
        BlockHeader* current = head;
        
        while (current != nullptr) {
            if (current == node) {
                return true;
            }
            current = current->nextHeader();
        }
        
        return false;
    }

    // Clear the entire list (just resets pointers, doesn't free memory)
    __device__ void clear() {
        head = nullptr;
        tail = nullptr;
    }

    // Debug: Print the list (sizes only)
    __device__ void printSizes() {
        #ifdef __CUDA_ARCH__
        BlockHeader* current = head;
        while (current != nullptr) {
            printf("%d -> ", current->size());
            current = current->nextHeader();
        }
        printf("NULL\n");
        #endif
    }

    // Verify list integrity (for debugging)
    __device__ bool verifyIntegrity() {
        if (head == nullptr) {
            return (tail == nullptr);
        }

        // Check head has no prev
        if (head->prevHeader() != nullptr) {
            return false;
        }

        // Check tail has no next
        if (tail != nullptr && tail->nextHeader() != nullptr) {
            return false;
        }

        // Count nodes and verify ordering
        int nodeCount = 0;
        BlockHeader* current = head;
        BlockHeader* prev = nullptr;
        
        while (current != nullptr) {
            // Verify prev pointer
            if (current->prevHeader() != prev) {
                return false;
            }

            // Verify ordering (descending)
            if (prev != nullptr && prev->size() < current->size()) {
                return false;
            }

            prev = current;
            current = current->nextHeader();
            nodeCount++;
        }

        // Verify tail pointer
        if (prev != tail) {
            return false;
        }

        return true;
    }
};

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



    // uint threadPoolSize = get_thread_pool_size();

    printf("thread pool size: %d\n",incomingMemSize);


    memPools[threadIndex].setMemBuffer(smem + (incomingMemSize * threadIndex));

    // printf("thread index %d's memory pool pointer: %p\n",threadIndex,memPools[threadIndex].memBuffer);


    // printf("mempool start pointer: %p\n", memPools[threadIndex].memBuffer);



    int16_t tempSize = (incomingMemSize - (sizeof(RBTreeBlockHeader) + sizeof(BlockFooter)));

    RBTreeBlockHeader *header = new (memPools[threadIndex].base()) RBTreeBlockHeader(tempSize & 0x7FFF, 0,0,0);


    memPools[threadIndex].freeList.insert(header);

    
    BlockFooter *footer = new (header->footer()) BlockFooter(header);


}

__device__ void* cmalloc(unsigned long size){

    uint threadIndex = get_linear_thread_index();

    if(&memPools[threadIndex].freeList == NULL || memPools[threadIndex].freeList.getRoot().size() < (size + (int16_t)sizeof(RBTreeBlockHeader) + (int16_t)sizeof(BlockFooter))){
        printf("malloc eval null exit\n");
        return NULL;
    }

    uint16_t preAllocationSize = memPools[threadIndex].freeList.getRoot().size();

    // Align size to 8 bytes, leaving the last 2 bytes for the block footer
    uint8_t offset = (8 - ((size + sizeof(BlockFooter)) % 8));
    uint16_t sizeToAlloc = size + offset;

    RBTreeBlockHeader* newlyAllocatedHeader = new (memPools[threadIndex].freeList.getRoot()) RBTreeBlockHeader(sizeToAlloc & 0xCFFF, 0,0,0);

    BlockFooter *newlyAllocatedFooter = new (newlyAllocatedHeader->footer()) BlockFooter(newlyAllocatedHeader);

    memPools[threadIndex].fullList.insert(newlyAllocatedHeader);

    // Create new free block if there's enough space
    uint16_t remainingSize = preAllocationSize - (sizeToAlloc + sizeof(RBTreeBlockHeader) + sizeof(BlockFooter));
    
    if(remainingSize > 0){
        RBTreeBlockHeader* newFreeHeader = new ((unsigned char*)(newlyAllocatedFooter) + sizeof(BlockFooter)) RBTreeBlockHeader(remainingSize & 0x7FFF, 0,0,0);
        
        BlockFooter *newFreeFooter = new (newFreeHeader->footer()) BlockFooter(newFreeHeader);
        // list_push_front(&memPools[threadIndex].freeList, newFreeHeader);

        memPools[threadIndex].freeList.insert(newFreeHeader);
    }


    void* return_index = (unsigned char*)newlyAllocatedHeader + sizeof(RBTreeBlockHeader);
    return return_index;
}

__device__ void cfree(void* addressForDeletion){
    uint threadIndex = get_linear_thread_index();


    if(addressForDeletion == NULL){
        return;
    }

    if(memPools[threadIndex].fullList.getRoot() == nullptr){
        return;
    }

    RBTreeBlockHeader* deletionTarget = (RBTreeBlockHeader*)((unsigned char*)addressForDeletion - sizeof(RBTreeBlockHeader));
    deletionTarget->setFull(false);

    int8_t forward_coalesce_valid = 0;
    int8_t backward_coalesce_valid = 0;

    RBTreeBlockHeader* prevHeader = deletionTarget->debugPrevHeader();
    RBTreeBlockHeader* nextHeader = deletionTarget->debugNextHeader();

    if(prevHeader != NULL && !prevHeader->isFull()){
        backward_coalesce_valid = 1;
    }
    
    if(nextHeader != NULL && !nextHeader->isFull()){
        forward_coalesce_valid = 1;
    }

    // Forward coalesce
    if(forward_coalesce_valid){
        int16_t tempSize = deletionTarget->size();
        tempSize += sizeof(BlockFooter) + sizeof(RBTreeBlockHeader) + nextHeader->size();
        
        deletionTarget->setSize(tempSize & 0x7FFF);

        BlockFooter *nextBlockFooter = new (deletionTarget->footer()) BlockFooter(deletionTarget);

        memPools[threadIndex].freeList.remove(nextHeader);
    }

    // Backward coalesce
    if(backward_coalesce_valid){
        memPools[threadIndex].freeList.remove(prevHeader);

        int16_t tempSize = prevHeader->size();

        tempSize += sizeof(BlockFooter) + sizeof(RBTreeBlockHeader) + deletionTarget->size();
        prevHeader->setSize(tempSize & 0x7FFF);

        BlockFooter *current_block_footer = new (prevHeader->footer()) BlockFooter(prevHeader);
        memPools[threadIndex].freeList.insert(prevHeader);
    } else {
        memPools[threadIndex].freeList.insert(deletionTarget);
    }

    // sort_free_list(threadIndex);
}