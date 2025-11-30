typedef struct {
    /*
    first bit will be full status, second bit will be color 
    typedef struct{
        int16_t fullStatus : 1
        int16_t color : 1
        int16_t size : 14
    } fullSize

    */
    int16_t fullSize;

    int16_t leftOffset;
    int16_t rightOffset;

    int16_t parentOffset;

} RBTreeBlockHeader;

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

static Direction direction(RBTreeBlockHeader* N) {

    RBTreeBlockHeader* parent = get_parent(N);
    RBTreeBlockHeader* parentRight = get_right_child(parent);

    return N == parentRight ? RIGHT : LEFT;
}

void set_color(RBTreeBlockHeader* node, Color color){
    if(color == BLACK){
        node->fullSize &= 0xCFFF;
    }
    else{
        node->fullSize |= 0x4000;
    }
}

RBTreeBlockHeader* get_parent(RBTreeBlockHeader* node){
    return (RBTreeBlockHeader*)(node + node->parentOffset);
}

RBTreeBlockHeader* get_left_child(RBTreeBlockHeader* node){
    return (RBTreeBlockHeader*)(node + node->leftOffset);
}

RBTreeBlockHeader* get_right_child(RBTreeBlockHeader* node){
    return (RBTreeBlockHeader*)(node + node->rightOffset);
}

__device__ Color get_color(RBTreeBlockHeader* currentHeader){
    int temp = currentHeader->fullSize >> 14;
    temp &= 0x0001;
    
    return (Color)temp;
}

__device__ int16_t get_offset(unsigned char* from, unsigned char* to){
    return (unsigned char*)to - (unsigned char*)from;
}

RBTreeBlockHeader* rotate_subtree(Tree* tree, RBTreeBlockHeader* sub, Direction dir) {

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
		tree->root = new_root;
    }

	return new_root;
}

void insert(Tree* tree, RBTreeBlockHeader* node, RBTreeBlockHeader* parent, Direction dir) {
	set_color(node, RED);
	node->parentOffset = get_offset((unsigned char*)node,(unsigned char*)parent);

	if (!parent) {
		tree->root = node;
		return;
	}

    if(dir == LEFT){
        parent->leftOffset = get_offset((unsigned char*)parent,(unsigned char*)node);
    }
    else{
        parent->rightOffset = get_offset((unsigned char*)parent,(unsigned char*)node);
        
    }

	// rebalance the tree 
	do {
		// Case #1
		if (get_color(parent) == BLACK) {
            return; 
        }

	    RBTreeBlockHeader* grandparent = get_parent(parent);

		if (!grandparent) {
			// Case #4
			set_color(parent, BLACK);
			return;
		}

		dir = direction(parent);
        RBTreeBlockHeader* uncle;

        if(dir == LEFT){
            uncle = get_right_child(grandparent);
           
            if (!uncle || get_color(uncle) == BLACK) {
                if (node == get_right_child(parent)) {
                    // Case #5
                    rotate_subtree(tree, parent, dir);
                    node = parent;
                    parent = get_left_child(grandparent);
                }

                // Case #6
                rotate_subtree(tree, grandparent, RIGHT);
                set_color(parent, BLACK);
                set_color(grandparent,RED);
                return;
            }

            
        }
        else{
            uncle = get_left_child(grandparent);
           
            if (!uncle || get_color(uncle) == BLACK) {
                if (node == get_left_child(parent)) {
                    // Case #5
                    rotate_subtree(tree, parent, dir);
                    node = parent;
                    parent = get_right_child(grandparent);
                }

                // Case #6
                rotate_subtree(tree, grandparent, LEFT);
                set_color(parent, BLACK);
                set_color(grandparent,RED);
                return;
            }
        }
	
		// Case #2
		set_color(parent, BLACK);
        set_color(uncle, BLACK);
        set_color(grandparent, RED);
        node = grandparent;

	} while ((parent = get_parent(node)));

	// Case #3
	return;

}

void remove(Tree* tree, RBTreeBlockHeader* node) {
	RBTreeBlockHeader* parent = get_parent(node);

	RBTreeBlockHeader* sibling;
	RBTreeBlockHeader* close_nephew;
	RBTreeBlockHeader* distant_nephew;

	Direction dir = direction(node);

    if(dir == LEFT){
        RBTreeBlockHeader* parentLeftChild = get_left_child(parent);
        parentLeftChild = NULL;
    }
    else{
        RBTreeBlockHeader* parentRightChild = get_left_child(parent);
        parentRightChild = NULL;
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
                rotate_subtree(tree, parent, dir);
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
                rotate_subtree(tree, parent, dir);
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
        rotate_subtree(tree, sibling, RIGHT);
    }
    else{
        rotate_subtree(tree, sibling, LEFT);
    }
    set_color(sibling, RED);
    set_color(close_nephew, BLACK);
    
	distant_nephew = sibling;
	sibling = close_nephew;

case_6:

	rotate_subtree(tree, parent, dir);
    set_color(sibling, get_color(parent));
    set_color(parent,BLACK);
    set_color(distant_nephew, BLACK);
	return;
}

