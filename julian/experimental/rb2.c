// #include <stddef.h> 
// #include "CPUMemBuffer.h"

// typedef enum Color{ 
//     BLACK, 
//     RED 
// } Color;

// typedef enum Direction{ 
//     LEFT, 
//     RIGHT 
// } Direction;

// // red-black tree node
// typedef struct Node {
// 	struct Node* parent; // null for the root node
// 	union {
// 		// Union so we can use ->left/->right or ->child[0]/->child[1]
// 		struct {
// 			struct Node* left;
// 			struct Node* right;
// 		};
// 		struct Node* child[2];
// 	};
// 	Color color;
// 	BlockHeader* key;
// } Node;

// typedef struct {
// 	struct Node* root;
// } Tree;

// static Direction direction(const Node* N) {
//     return N == N->parent->right ? RIGHT : LEFT;
// };

// static int compare_key(const BlockHeader* a, const BlockHeader* b) {
//     if (a->size < b->size) return -1;  // e.g., based on address or field
//     if (a->size > b->size) return 1;
//     return 0;
// }

// Node* rotate_subtree(Tree* tree, Node* sub, Direction dir) {
// 	Node* sub_parent = sub->parent;
// 	Node* new_root = sub->child[1 - dir]; // 1 - dir is the opposite direction
// 	Node* new_child = new_root->child[dir];

// 	sub->child[1 - dir] = new_child;

// 	if (new_child) {
//         new_child->parent = sub;
//     }

// 	new_root->child[dir] = sub;

// 	new_root->parent = sub_parent;
// 	sub->parent = new_root;
// 	if (sub_parent) {
// 		sub_parent->child[sub == sub_parent->right] = new_root;
// 	} else {
// 		tree->root = new_root;
//     }

// 	return new_root;
// }

// void insert(Tree* tree, Node* node, Node* parent, Direction dir) {
// 	node->color = RED;
// 	node->parent = parent;

// 	if (!parent) {
// 		tree->root = node;
// 		return;
// 	}

// 	parent->child[dir] = node;

// 	// rebalance the tree 
// 	do {
// 		// Case #1
// 		if (parent->color == BLACK) {
//             return; 
//         }

// 	    Node* grandparent = parent->parent;

// 		if (!grandparent) {
// 			// Case #4
// 			parent->color = BLACK;
// 			return;
// 		}

// 		dir = direction(parent);
// 		Node* uncle = grandparent->child[1 - dir];
// 		if (!uncle || uncle->color == BLACK) {
// 			if (node == parent->child[1 - dir]) {
// 				// Case #5
// 				rotate_subtree(tree, parent, dir);
// 				node = parent;
// 				parent = grandparent->child[dir];
// 			}

// 			// Case #6
// 			rotate_subtree(tree, grandparent, 1 - dir);
// 			parent->color = BLACK;
// 			grandparent->color = RED;
// 			return;
// 		}
	
// 		// Case #2
// 		parent->color = BLACK;
// 		uncle->color = BLACK;
// 		grandparent->color = RED;
// 		node = grandparent;

// 	} while (parent = node->parent);

// 	// Case #3
// 	return;

// }

// void remove(Tree* tree, Node* node) {
// 	Node* parent = node->parent;

// 	Node* sibling;
// 	Node* close_nephew;
// 	Node* distant_nephew;

// 	Direction dir = direction(node);

// 	parent->child[dir] = NULL;
// 	goto start_balance;

// 	do {
// 		dir = direction(node);
// start_balance:
// 		sibling = parent->child[1 - dir];
// 		distant_nephew = sibling->child[1 - dir];
// 		close_nephew = sibling->child[dir];
// 		if (sibling->color == RED) {
// 			// Case #3
// 			rotate_subtree(tree, parent, dir);
// 			parent->color = RED;
// 			sibling->color = BLACK;
// 			sibling = close_nephew;

// 			distant_nephew = sibling->child[1 - dir];
// 			if (distant_nephew && distant_nephew->color == RED) {
// 				goto case_6;
//             }
// 			close_nephew = sibling->child[dir];
// 			if (close_nephew && close_nephew->color == RED) {
// 				goto case_5;
//             }

// 			// Case #4
// 			sibling->color = RED;
// 			parent->color = BLACK;
// 			return;
// 		}

// 		if (distant_nephew && distant_nephew->color == RED) {
// 			goto case_6;
//         }

// 		if (close_nephew && close_nephew->color == RED) {
// 			goto case_5;
//         }

// 		if (parent->color == RED) {
// 			// Case #4
// 			sibling->color = RED;
// 			parent->color = BLACK;
// 			return;
// 		}

// 		// Case #1
// 		if (!parent) {
//             return;
//         }

// 		// Case #2
// 		sibling->color = RED;
// 		node = parent;

// 	} while (parent = node->parent);

// case_5:

// 	rotate_subtree(tree, sibling, 1 - dir);
// 	sibling->color = RED;
// 	close_nephew->color = BLACK;
// 	distant_nephew = sibling;
// 	sibling = close_nephew;

// case_6:

// 	rotate_subtree(tree, parent, dir);
// 	sibling->color = parent->color;
// 	parent->color = BLACK;
// 	distant_nephew->color = BLACK;
// 	return;
// }

// void rb_insert(Tree* tree, Node* node) {
//     Node* parent = NULL;
//     Node* curr = tree->root;
//     Direction dir = LEFT;

//     // Walk down the tree until we find a NULL slot
//     while (curr != NULL) {
//         parent = curr;
//         int cmp = compare_key(node->key, curr->key);
//         if (cmp < 0) {
//             curr = curr->left;
//             dir = LEFT;
//         } else if (cmp > 0) {
//             curr = curr->right;
//             dir = RIGHT;
//         } else {
//             // Key already exists — depends on your policy
//             return; // or overwrite, or ignore
//         }
//     }

//     // Now we have parent and dir — call the balancing insert
//     insert(tree, node, parent, dir);
// }