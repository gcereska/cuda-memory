#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>

#include "rb.h"
#include "CPUMemBuffer.h"

#define MEM_POOL_SIZE 3072

//this is the node for the AVL tree
//this is the most efficient means I could think of to store the meta data required, reducing the size from roughly 32 bytes down to just 8 bytes, assuming padding

//if the color were packed into the size, it would be possible to make a RBtree as well with 8 byte nodes. it only takes 12 bits to represent the entire space (in the size variable), so you could use the upper 1 for color and the lower 12 for size...

/*
For starters I will just implement a linked list implementation, there is going to be a lot of weird math to deal with with just a simple linked list so I am going to get the first implementation down.
If the linked list goes well and proves valuable I will move on to either the AVL or RBTree, RB would be quite tough I believe, we would need to write it from scratch nearly to account for all the weird pointer operations and bit mapping

*/


extern char* memBuffer;
extern BlockHeader *free_list;
extern BlockHeader *full_list;  

// typedef struct {


    
//     int16_t next_offset;
//     int16_t prev_offset;

//     //because our space is bounded to roughly 3kb, it should be possible to use a uint16 instead of a size_t to represent our space(4kb can be represented in 12 bits...)
//     //update: because I am starting with linked list this will have a bool packed into the first bit

//     int16_t size;
//     uint16_t padding; //6 byte accesses are bad, 8 are good

//     /*
//     for later implementations

//     //because of the bounded space, instead of holding full 64 bit pointers it should be possible to get away with a relative offset from the start of the BlockHeader*, thus reducing our space
//     int16_t left_offset;
//     int16_t right_offset;


//     int16_t avl_meta_data; //this will hold a boolean(free) and a 15 bit integer(AVL height)
//     //if the bool is on the most significant bit, i can just check the signedness of the integer to check full or free

//     int16_t parent_offset; //for a RBTree impl

//     */
    
// } BlockHeader;



typedef struct {
    char* buffer;
     
} CpuMemBuffer;


int8_t is_full(BlockHeader* currentHeader);

//helper function to clean up pointer math
int16_t get_offset(char* lhs, char* rhs);

BlockFooter* get_footer(BlockHeader* header);

BlockHeader* get_prev_header(BlockHeader* currentHeader);

BlockHeader* get_next_header(BlockHeader* currentHeader);

BlockHeader* get_next_list_header(BlockHeader* currentHeader);

BlockHeader* get_prev_list_header(BlockHeader* currentHeader);

void list_push_front(BlockHeader** list, BlockHeader* insertionNode);

void list_remove(BlockHeader** list, BlockHeader* deletionNode);

void debug_print_buffer();
void debug_print_free_list();
void debug_print_full_list();

int16_t get_node_size(BlockHeader* currentHeader);


//#TODO deal with alignment later
void init_cpu_buffer();
void free_buffer();

void* cmalloc(uint16_t size);

void cfree(void * address_for_deletion);



