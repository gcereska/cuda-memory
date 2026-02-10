#include <stdio.h>

#include "RBTree.cuh"


//-------------------------------------------------------HELPER FUNCTIONS--------------------------------------------------------------


__device__ static Direction direction(RBTreeBlockHeader* N) {

    RBTreeBlockHeader* parent = get_parent(N);
    
    if (!parent) {
        return LEFT;  
    }

    RBTreeBlockHeader* parentRight = get_right_child(parent);

    return N == parentRight ? RIGHT : LEFT;
}

__device__ void set_color(RBTreeBlockHeader* node, Color color){
    if(color == BLACK){
        node->fullSize &= 0xCFFF;
    }
    else{
        node->fullSize |= 0x4000;
    }
}

__device__ RBTreeBlockHeader* get_parent(RBTreeBlockHeader* node){
    if (node->parentOffset == 0) {
        return NULL;
    }
    return (RBTreeBlockHeader*)(node + node->parentOffset);
}

__device__ RBTreeBlockHeader* get_left_child(RBTreeBlockHeader* node){
    if (node->leftOffset == 0) {
        return NULL;
    }
    return (RBTreeBlockHeader*)(node + node->leftOffset);
}

__device__ RBTreeBlockHeader* get_right_child(RBTreeBlockHeader* node){
    if (node->rightOffset == 0) {
        return NULL;
    }
    return (RBTreeBlockHeader*)(node + node->rightOffset);
}

__device__ Color get_color(RBTreeBlockHeader* currentHeader){
    int temp = currentHeader->fullSize >> 14;
    temp &= 0x0001;
    
    return (Color)temp;
}

static __device__ int16_t get_offset(unsigned char* from, unsigned char* to){
    return (unsigned char*)to - (unsigned char*)from;
}

// Helper function to extract the size from fullSize field
__device__ int16_t get_size(RBTreeBlockHeader* node) {
    // Mask out the two flag bits (bits 15 and 14) and keep only the size (bits 13-0)
    return node->fullSize & 0x3FFF;
}

__device__ bool has_left_child(RBTreeBlockHeader* node) {
    return node->leftOffset != 0;
}

__device__ bool has_right_child(RBTreeBlockHeader* node) {
    return node->rightOffset != 0;
}

//------------------------------------------------START OF REAL FUNCTIONS-------------------------------------------------------------------

// Search function - returns pointer to node with matching size, or NULL if not found
__device__ RBTreeBlockHeader* search(RBTreeBlockHeader** root, int16_t target_size) {
    RBTreeBlockHeader* current = *root;
    
    while (current != NULL) {
        int16_t current_size = get_size(current);
        
        if (target_size == current_size) {
            // Found the target
            return current;
        } else if (target_size < current_size) {
            // Search left subtree
            if (current->leftOffset == 0) {
                return NULL;  // No left child
            }
            current = get_left_child(current);
        } else {
            // Search right subtree
            if (current->rightOffset == 0) {
                return NULL;  // No right child
            }
            current = get_right_child(current);
        }
    }
    
    return NULL;  // Tree is empty or element not found
}

// Alternative: Search for closest fit (smallest block >= target_size)
// Useful for memory allocation scenarios
__device__ RBTreeBlockHeader* search_best_fit(RBTreeBlockHeader** root, int16_t target_size) {
    RBTreeBlockHeader* current = *root;
    RBTreeBlockHeader* best_fit = NULL;
    
    while (current != NULL) {
        int16_t current_size = get_size(current);
        
        if (current_size >= target_size) {
            // This could be a fit - update best_fit if it's smaller than current best
            if (best_fit == NULL || current_size < get_size(best_fit)) {
                best_fit = current;
            }
            
            // Try to find a smaller fit in the left subtree
            if (current->leftOffset == 0) {
                break;
            }
            current = get_left_child(current);
        } else {
            // Current is too small, search right subtree
            if (current->rightOffset == 0) {
                break;
            }
            current = get_right_child(current);
        }
    }
    
    return best_fit;
}

// Find insertion point - returns parent node and direction for insertion
// Pass pointers to parent and dir, which will be set by this function
__device__ void find_insert_position(RBTreeBlockHeader** root, int16_t new_size, 
                          RBTreeBlockHeader** out_parent, Direction* out_dir) {
    if (root == NULL || *root == NULL) {
        *out_parent = NULL;
        *out_dir = LEFT;
        return;
    }
    
    RBTreeBlockHeader* current = *root;
    RBTreeBlockHeader* parent = NULL;
    Direction dir = LEFT;
    
    int iterations = 0;
    while (current != NULL && iterations < 100) {  // Safety limit
        iterations++;
        parent = current;
        int16_t current_size = get_size(current);
                
        if (new_size < current_size) {
            dir = LEFT;
            if (current->leftOffset == 0) {
                current = NULL;
            } else {
                current = get_left_child(current);
            }
        } else {
            dir = RIGHT;
            if (current->rightOffset == 0) {
                current = NULL;
            } else {
                current = get_right_child(current);
            }
        }
    }
    
    *out_parent = parent;
    *out_dir = dir;
}

// Example usage of find_insert_position with your existing insert function:
__device__ void insert_node(RBTreeBlockHeader** root, RBTreeBlockHeader* new_node, int16_t size) {
    // Set the size in the node (preserving any existing flags)
    new_node->fullSize = (new_node->fullSize & 0xC000) | (size & 0x3FFF);
    
    // Initialize offsets to 0 (no children)
    new_node->leftOffset = 0;
    new_node->rightOffset = 0;
    
    // Find where to insert
    RBTreeBlockHeader* parent;
    Direction dir;



    find_insert_position(root, size, &parent, &dir);
    
    // Use your existing insert function to insert and rebalance
    insert(root, new_node, parent, dir);
}

__device__ RBTreeBlockHeader* rotate_subtree(RBTreeBlockHeader** root, RBTreeBlockHeader* sub, Direction dir) {

	RBTreeBlockHeader* sub_parent = get_parent(sub);
	RBTreeBlockHeader* new_root; 
	RBTreeBlockHeader* new_child;

    if(dir == LEFT){
        new_root = get_right_child(sub);
        new_child = get_left_child(sub);

        sub->rightOffset = get_offset((unsigned char*)sub, (unsigned char*)new_child);
    }
    else{
        new_root = get_left_child(sub);
        new_child = get_right_child(sub);

        sub->leftOffset = get_offset((unsigned char*)sub, (unsigned char*)new_child);
    }

	if (new_child) {
        new_child->parentOffset = get_offset((unsigned char*)new_child,(unsigned char*)sub);
    }

    if(dir == LEFT){
	    new_root->leftOffset = get_offset((unsigned char*)new_root,(unsigned char*)sub);
    }
    else{
        new_root->rightOffset = get_offset((unsigned char*)new_root,(unsigned char*)sub);
    }


	new_root->parentOffset = get_offset((unsigned char*)new_root,(unsigned char*)sub_parent);
	sub->parentOffset = get_offset((unsigned char*)sub, (unsigned char*)new_root);
	if (sub_parent) {
        if(sub == get_right_child(sub_parent)){
            sub_parent->rightOffset = get_offset((unsigned char*)sub_parent,(unsigned char*)new_root);
        }
        else{
            sub_parent->leftOffset = get_offset((unsigned char*)sub_parent,(unsigned char*)new_root);
        }

		
	} else {
		*root = new_root;
    }

	return new_root;
}

__device__ void insert(RBTreeBlockHeader** root, RBTreeBlockHeader* node, RBTreeBlockHeader* parent, Direction dir) {
    
    node->leftOffset = 0;
    node->rightOffset = 0;
    
    if (!parent) {
        node->parentOffset = 0;
        *root = node;
        set_color(node, BLACK);
        return;
    }
    
    set_color(node, RED);
    node->parentOffset = get_offset((unsigned char*)node, (unsigned char*)parent);
    
    if(dir == LEFT){
        parent->leftOffset = get_offset((unsigned char*)parent, (unsigned char*)node);
    } else {
        parent->rightOffset = get_offset((unsigned char*)parent, (unsigned char*)node);
    }
    
    int iterations = 0;
    
    do {
        iterations++;
        
 
        
        // Case #1
        if (get_color(parent) == BLACK) {
            return; 
        }
        
        RBTreeBlockHeader* grandparent = get_parent(parent);
        
        if (!grandparent) {
            set_color(parent, BLACK);
            return;
        }
        
        dir = direction(parent);
        
        RBTreeBlockHeader* uncle = NULL;
        
        set_color(parent, BLACK);
        set_color(uncle, BLACK);
        set_color(grandparent, RED);
        
        node = grandparent;
        
        parent = get_parent(node);
        
    } while (parent);
    
    set_color(*root, BLACK);
}

__device__ void remove(RBTreeBlockHeader** root, RBTreeBlockHeader* node) {


	RBTreeBlockHeader* parent = get_parent(node);
    if (!parent) {
        *root = NULL;
        return;
    }

	RBTreeBlockHeader* sibling;
	RBTreeBlockHeader* close_nephew;
	RBTreeBlockHeader* distant_nephew;


	Direction dir = direction(node);

    if(dir == LEFT){
        parent->leftOffset = 0;
    }
    else{
        parent->rightOffset = 0;
    }
	goto start_balance;

	do {
		dir = direction(node);

start_balance:
        if(dir == LEFT){
            sibling = get_right_child(parent);
            distant_nephew = get_right_child(sibling);
            close_nephew =get_left_child(sibling);
            if (get_color(sibling) == RED) {
                // Case #3
                rotate_subtree(root, parent, dir);
                set_color(parent, RED);
                set_color(sibling, BLACK);
                
                sibling = close_nephew;

                distant_nephew = get_right_child(sibling);

                if (distant_nephew && get_color(distant_nephew) == RED) {
                    goto case_6;
                }
                close_nephew = get_left_child(sibling);
                if (close_nephew && get_color(close_nephew) == RED) {
                    goto case_5;
                }

                // Case #4
                set_color(sibling, RED);
                set_color(parent, BLACK);
                return;
            }
        }
        else{
            sibling = get_left_child(parent);
            distant_nephew = get_left_child(sibling);
            close_nephew =get_right_child(sibling);
            if (get_color(sibling) == RED) {
                // Case #3
                rotate_subtree(root, parent, dir);
                set_color(parent, RED);
                set_color(sibling, BLACK);
                
                sibling = close_nephew;

                distant_nephew = get_left_child(sibling);

                if (distant_nephew && get_color(distant_nephew) == RED) {
                    goto case_6;
                }
                close_nephew = get_right_child(sibling);
                if (close_nephew && get_color(close_nephew) == RED) {
                    goto case_5;
                }

                // Case #4
                set_color(sibling, RED);
                set_color(parent, BLACK);
                return;
            }
        }

		if (distant_nephew && get_color(distant_nephew) == RED) {
			goto case_6;
        }

		if (close_nephew && get_color(close_nephew) == RED) {
			goto case_5;
        }

		if (!parent) {
		    // Case #1
            return;
        }

		if (get_color(parent) == RED) {
			// Case #4
            set_color(sibling, RED);
            set_color(parent, BLACK);
			return;
		}

		// Case #2
        set_color(sibling, RED);
		node = parent;

	} while (parent = get_parent(node));

case_5:

    if(dir == LEFT){
        rotate_subtree(root, sibling, RIGHT);
    }
    else{
        rotate_subtree(root, sibling, LEFT);
    }
    set_color(sibling, RED);
    set_color(close_nephew, BLACK);
    
	distant_nephew = sibling;
	sibling = close_nephew;

case_6:

	rotate_subtree(root, parent, dir);
    set_color(sibling, get_color(parent));
    set_color(parent,BLACK);
    set_color(distant_nephew, BLACK);
	return;
}

