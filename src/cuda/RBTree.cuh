#ifndef RBTREE_CUH
#define RBTREE_CUH

#include "typeDefs.cuh"

typedef enum Color: int {
    BLACK,
    RED
} Color;

typedef enum Direction: int {
    LEFT,
    RIGHT
} Direction;

__device__ static Direction direction(RBTreeBlockHeader* N);

__device__ void set_color(RBTreeBlockHeader* node, Color color);

__device__ RBTreeBlockHeader* get_parent(RBTreeBlockHeader* node);

__device__ RBTreeBlockHeader* get_left_child(RBTreeBlockHeader* node);

__device__ RBTreeBlockHeader* get_right_child(RBTreeBlockHeader* node);

__device__ Color get_color(RBTreeBlockHeader* currentHeader);

static __device__ int16_t get_offset(unsigned char* from, unsigned char* to);

__device__ RBTreeBlockHeader* rotate_subtree(RBTreeBlockHeader** root, RBTreeBlockHeader* sub, Direction dir);

__device__ void insert_node(RBTreeBlockHeader** root, RBTreeBlockHeader* new_node, int16_t size);
__device__ void insert(RBTreeBlockHeader** root, RBTreeBlockHeader* node, RBTreeBlockHeader* parent, Direction dir);
__device__ void remove(RBTreeBlockHeader** root, RBTreeBlockHeader* node);
#endif
