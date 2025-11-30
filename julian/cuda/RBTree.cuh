#include "typeDefs.cuh"

typedef enum Color: int { 
    BLACK, 
    RED 
} Color;

typedef enum Direction: int { 
    LEFT, 
    RIGHT 
} Direction;

typedef struct {
	RBTreeBlockHeader* root;
} Tree;

static Direction direction(RBTreeBlockHeader* N);

void set_color(RBTreeBlockHeader* node, Color color);

RBTreeBlockHeader* get_parent(RBTreeBlockHeader* node);

RBTreeBlockHeader* get_left_child(RBTreeBlockHeader* node);

RBTreeBlockHeader* get_right_child(RBTreeBlockHeader* node);

__device__ Color get_color(RBTreeBlockHeader* currentHeader);

__device__ int16_t get_offset(unsigned char* from, unsigned char* to);

RBTreeBlockHeader* rotate_subtree(Tree* tree, RBTreeBlockHeader* sub, Direction dir);

void insert(Tree* tree, RBTreeBlockHeader* node, RBTreeBlockHeader* parent, Direction dir);
void remove(Tree* tree, RBTreeBlockHeader* node);
