#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>


#include "CPUMemBuffer.h"
#include "cpu_custom_malloc.h"

#define MEM_POOL_SIZE 3072

char* memBuffer;
BlockHeader *freeList = NULL;
BlockHeader *fullList = NULL;  

void sort_free_list(){
    if(freeList == NULL) return;
    
    int changed = 1;
    while(changed == 1){
        //printf("outer loop\n");
        changed = 0;
        BlockHeader* current = (BlockHeader*)freeList;
        
        while(current != NULL){
            
            BlockHeader* next = get_next_list_header(current);
            
            // If next exists and current is smaller, swap
            if(next != NULL && current->size < next->size){
                //printf("inner loop\n");

                BlockHeader* prev = get_prev_list_header(current);
                BlockHeader* afterNext = get_next_list_header(next);
                
                next->prevOffset = 0;
                // Fix prev node's next pointer
                if(prev != NULL){
                    prev->nextOffset = get_offset((char*)prev, (char*)next);
                    next->prevOffset = get_offset((char*)next, (char*)prev);
                } else {
                    freeList = next;
                    
                }
                
                current->nextOffset = 0;
                // Fix after_next node's prev pointer  
                if(afterNext != NULL){
                    afterNext->prevOffset = get_offset((char*)afterNext, (char*)current);
                    current->nextOffset = get_offset((char*)current, (char*)afterNext);
                }
                
                // Make next point to prev and current
                
                next->nextOffset = get_offset((char*)next, (char*)current);
                
                // Make current point to next and after_next
                current->prevOffset = get_offset((char*)current, (char*)next);
                
                
                changed = 1;
                break; 
            }
            
            if(!changed){
                current = next;  
                
            }
            
            
        }
    }
}

//used solely for debugging help and correctness checking
void debug_print_buffer(){
    printf("\n|");
    BlockHeader* currentNode = (BlockHeader*)memBuffer;
    while (currentNode != NULL){
        printf(" %4u |", currentNode->size);
        currentNode = get_next_header(currentNode);
    }
    printf("\n|");
    currentNode = (BlockHeader*)memBuffer;

    while (currentNode != NULL){
        if(currentNode->full){
            printf(" FULL |");
        } else {
            printf(" FREE |");
        }
        currentNode = get_next_header(currentNode);
    }
    printf(" \n\n");
}


void debug_print_free_list(){
    printf("\n|");
    BlockHeader* currentNode = (BlockHeader*)freeList;
    while (currentNode != NULL){
        printf(" %4u |", currentNode->size);
        currentNode = get_next_list_header(currentNode);
    }
    printf("\n|");
    currentNode = (BlockHeader*)freeList;

    while (currentNode != NULL){
        if(currentNode->full){
            printf(" FULL |");
        } else {
            printf(" FREE |");
        }
        currentNode = get_next_list_header(currentNode);
    }
    printf(" \n\n");
}


void debug_print_full_list(){
    printf("\n|");
    BlockHeader* currentNode = (BlockHeader*)fullList;
    while (currentNode != NULL){
        printf(" %4u |", currentNode->size);
        currentNode = get_next_list_header(currentNode);
    }
    printf("\n|");
    currentNode = (BlockHeader*)fullList;

    while (currentNode != NULL){
        if(currentNode->full){
            printf(" FULL |");
        } else {
            printf(" FREE |");
        }
        currentNode = get_next_list_header(currentNode);
    }
    printf(" \n\n");
}

//this is kind of useles since I added the bitmapped BlockHeader
int8_t is_full(BlockHeader* currentHeader){
    return currentHeader->full != 0;
}

//this is kind of useles since I added the bitmapped BlockHeader
//the size and full used to be packed into an int16 hence the helper functions
int16_t get_node_size(BlockHeader* currentHeader){
    return currentHeader->size;
}

//these are all relatively simple helper functions to make the pointer math easier
int16_t get_offset(char* from, char* to){
    return (char*)to - (char*)from;
}

BlockFooter* get_footer(BlockHeader* header){
    return (BlockFooter*)((char*)header + header->size + sizeof(BlockHeader));
}

//gets the previous header from the total memPool
BlockHeader* get_prev_header(BlockHeader* currentHeader){
    if((void*)currentHeader == (void*)memBuffer){
        return NULL; 
    }

    BlockFooter* prev_footer = (BlockFooter*)((char*)currentHeader - sizeof(BlockFooter));
    BlockHeader* prevHeader = (BlockHeader*)((char*)prev_footer + prev_footer->headerOffset);
    return prevHeader;
}

//gets the next header from the total memPool
BlockHeader* get_next_header(BlockHeader* currentHeader){
    int currentHeaderOffset = get_offset(memBuffer, (char*)currentHeader);
    uint64_t temp_size = currentHeader->size;
    
    if((currentHeaderOffset + temp_size + sizeof(BlockHeader) + sizeof(BlockFooter)) >= MEM_POOL_SIZE){
        return NULL;
    }
    
    BlockHeader* nextHeader = (BlockHeader*)((char*)currentHeader + temp_size + sizeof(BlockFooter) + sizeof(BlockHeader));
    return nextHeader;
}

//gets the next header from the list it current resides in
BlockHeader* get_next_list_header(BlockHeader* currentHeader){
    if(currentHeader->nextOffset == 0){
        return NULL;
    }
    return (BlockHeader*)((char*)currentHeader + currentHeader->nextOffset);
}

//gets the previous header from the list it current resides in
BlockHeader* get_prev_list_header(BlockHeader* currentHeader){
    if(currentHeader->prevOffset == 0){
        return NULL;
    }
    return (BlockHeader*)((char*)currentHeader + currentHeader->prevOffset);
}

void list_push_front(BlockHeader** list, BlockHeader* insertionNode){
    if(*list == NULL){
        *list = insertionNode;
        insertionNode->nextOffset = 0;
        insertionNode->prevOffset = 0;
        return;
    }
    
    (*list)->prevOffset = get_offset((char*)*list, (char*)insertionNode);
    insertionNode->nextOffset = get_offset((char*)insertionNode, (char*)*list);
    insertionNode->prevOffset = 0;
    *list = insertionNode;
}

void list_remove(BlockHeader** list, BlockHeader* deletionNode){
    BlockHeader* currentNodeNext = NULL;
    BlockHeader* currentNode_prev = NULL;
    
    if(deletionNode->nextOffset != 0){
        currentNodeNext = (BlockHeader*)((char*)deletionNode + deletionNode->nextOffset);
    }
    
    if(deletionNode->prevOffset != 0){
        currentNode_prev = (BlockHeader*)((char*)deletionNode + deletionNode->prevOffset);
    }
    
    // Fix the previous node's next pointer
    if(currentNode_prev != NULL){
        if(currentNodeNext != NULL){
            currentNode_prev->nextOffset = get_offset((char*)currentNode_prev, (char*)currentNodeNext);
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
            currentNodeNext->prevOffset = get_offset((char*)currentNodeNext, (char*)currentNode_prev);
        } else {
            currentNodeNext->prevOffset = 0;
        }
    }
    
    // Clear the deleted node's pointers
    deletionNode->nextOffset = 0;
    deletionNode->prevOffset = 0;
}

void init_cpu_buffer(){
    memBuffer = malloc(MEM_POOL_SIZE * sizeof(char));
    // printf("mempool start pointer: %p\n", memBuffer);

    BlockHeader *header = (BlockHeader*)memBuffer;
    header->size = MEM_POOL_SIZE - (sizeof(BlockHeader) + sizeof(BlockFooter));
    header->nextOffset = 0;
    header->prevOffset = 0;
    header->full = 0;

    list_push_front(&freeList, header);

    BlockFooter *footer = get_footer(header);
    footer->headerOffset = get_offset((char*)footer, (char*)header);
}

//frees the memBuffer storage allocation
void free_buffer(){
    free(memBuffer);
}

void* cmalloc(uint16_t size){
    if(freeList == NULL || freeList->size < size + sizeof(BlockHeader) + sizeof(BlockFooter)){
        // printf("malloc eval null exit\n");
        return NULL;
    }

    uint16_t preAllocationSize = freeList->size;
    BlockHeader* newlyAllocatedHeader = freeList;

    // Align size to 8 bytes, leaving the last 2 bytes for the block footer
    uint8_t offset = (8 - ((size + sizeof(BlockFooter)) % 8));
    uint16_t sizeToAlloc = size + offset;

    newlyAllocatedHeader->size = sizeToAlloc;
    newlyAllocatedHeader->full = 1;

    BlockFooter* newlyAllocatedFooter = get_footer(newlyAllocatedHeader);
    newlyAllocatedFooter->headerOffset = get_offset((char*)newlyAllocatedFooter, (char*)newlyAllocatedHeader);

    // Remove from free list before modifying
    list_remove(&freeList, newlyAllocatedHeader);
    list_push_front(&fullList, newlyAllocatedHeader);

    // Create new free block if there's enough space
    uint16_t remainingSize = preAllocationSize - (sizeToAlloc + sizeof(BlockHeader) + sizeof(BlockFooter));
    
    if(remainingSize > 0){
        BlockHeader* newFreeHeader = (BlockHeader*)((char*)(newlyAllocatedFooter) + sizeof(BlockFooter));
        newFreeHeader->size = remainingSize;
        newFreeHeader->full = 0;
        newFreeHeader->nextOffset = 0;
        newFreeHeader->prevOffset = 0;

        BlockFooter* newFreeFooter = get_footer(newFreeHeader);
        newFreeFooter->headerOffset = get_offset((char*)newFreeFooter, (char*)newFreeHeader);

        list_push_front(&freeList, newFreeHeader);
    }

    sort_free_list();

    void* return_index = (char*)newlyAllocatedHeader + sizeof(BlockHeader);
    return return_index;
}

void cfree(void* addressForDeletion){
    if(addressForDeletion == NULL){
        return;
    }

    if(fullList == NULL){
        return;
    }

    BlockHeader* deletion_target = (BlockHeader*)((char*)addressForDeletion - sizeof(BlockHeader));
    list_remove(&fullList, deletion_target);

    deletion_target->full = 0;

    int8_t forward_coalesce_valid = 0;
    int8_t backward_coalesce_valid = 0;

    BlockHeader* prevHeader = get_prev_header(deletion_target);
    BlockHeader* nextHeader = get_next_header(deletion_target);

    if(prevHeader != NULL && !is_full(prevHeader)){
        backward_coalesce_valid = 1;
    }
    
    if(nextHeader != NULL && !is_full(nextHeader)){
        forward_coalesce_valid = 1;
    }

    // Forward coalesce
    if(forward_coalesce_valid){
        deletion_target->size += sizeof(BlockFooter) + sizeof(BlockHeader) + nextHeader->size;
        
        BlockFooter* next_block_footer = get_footer(deletion_target);
        next_block_footer->headerOffset = get_offset((char*)next_block_footer, (char*)deletion_target);

        list_remove(&freeList, nextHeader);
    }

    // Backward coalesce
    if(backward_coalesce_valid){
        list_remove(&freeList, prevHeader);
        
        prevHeader->size += sizeof(BlockFooter) + sizeof(BlockHeader) + deletion_target->size;
        
        BlockFooter* current_block_footer = get_footer(prevHeader);
        current_block_footer->headerOffset = get_offset((char*)current_block_footer, (char*)prevHeader);
        
        list_push_front(&freeList, prevHeader);
    } else {
        list_push_front(&freeList, deletion_target);
    }

    sort_free_list();
}