#include "RBTreeBlockHeader.cuh"
//https://www.geeksforgeeks.org/cpp/red-black-tree-in-cpp/
class RedBlackTree {
private:
    __device__ RBTreeBlockHeader* root; // Root of the Red-Black Tree

    // Utility function: Left Rotation
    __device__ void rotateLeft(RBTreeBlockHeader* node)
    {
        RBTreeBlockHeader* child = node->rightHeader();
        node->set_right_offset(child->leftHeader());
        if (node->rightHeader() != nullptr){
            node->rightHeader()->set_parent_offset(node);
        }
        child->set_parent_offset(node->parentHeader());
        if (node->parentHeader() == nullptr)
            root = child;
        else if (node == node->parentHeader()->leftHeader())
            node->parentHeader()->set_left_offset(child);
        else
            node->parentHeader()->set_right_offset(child);
        child->set_left_offset(node);
        node->set_parent_offset(child);
    }

    // Utility function: Right Rotation
    __device__ void rotateRight(RBTreeBlockHeader* node)
    {
        RBTreeBlockHeader* child = node->leftHeader();
        node->set_left_offset(child->rightHeader());
        if (node->leftHeader() != nullptr)
            node->leftHeader()->set_parent_offset(node);
        child->set_parent_offset(node->parentHeader());
        if (node->parentHeader() == nullptr)
            root = child;
        else if (node == node->parentHeader()->leftHeader())
            node->parentHeader()->set_left_offset(child);
        else
            node->parentHeader()->set_right_offset(child);
        child->set_right_offset(node);
        node->set_parent_offset(child);
    }

    // Utility function: Swap colors
    __device__ void swap(Color& a, Color& b)
    {
        Color temp = a;
        a = b;
        b = temp;
    }

    // Utility function: Fixing Insertion Violation
    __device__ void fixInsert(RBTreeBlockHeader* node)
    {
        RBTreeBlockHeader* parent = nullptr;
        RBTreeBlockHeader* grandparent = nullptr;
        while (node != root && node->color() == RED
               && node->parentHeader()->color() == RED) {
            parent = node->parentHeader();
            grandparent = parent->parentHeader();
            if (parent == grandparent->leftHeader()) {
                RBTreeBlockHeader* uncle = grandparent->rightHeader();
                if (uncle != nullptr && uncle->color() == RED) {
                    grandparent->setColor(RED);
                    parent->setColor(BLACK);
                    uncle->setColor(BLACK);
                    node = grandparent;
                }
                else {
                    if (node == parent->rightHeader()) {
                        rotateLeft(parent);
                        node = parent;
                        parent = node->parentHeader();
                    }
                    rotateRight(grandparent);
                    Color pColor = parent->color();
                    Color gColor = grandparent->color();
                    parent->setColor(gColor);
                    grandparent->setColor(pColor);
                    node = parent;
                }
            }
            else {
                RBTreeBlockHeader* uncle = grandparent->leftHeader();
                if (uncle != nullptr && uncle->color() == RED) {
                    grandparent->setColor(RED);
                    parent->setColor(BLACK);
                    uncle->setColor(BLACK);
                    node = grandparent;
                }
                else {
                    if (node == parent->leftHeader()) {
                        rotateRight(parent);
                        node = parent;
                        parent = node->parentHeader();
                    }
                    rotateLeft(grandparent);
                    Color pColor = parent->color();
                    Color gColor = grandparent->color();
                    parent->setColor(gColor);
                    grandparent->setColor(pColor);
                    node = parent;
                }
            }
        }
        root->setColor(BLACK);
    }

    // Utility function: Fixing Deletion Violation
    __device__ void fixDelete(RBTreeBlockHeader* node)
    {
        if (node == nullptr) return;
        
        while (node != root && node->color() == BLACK) {
            if (node == node->parentHeader()->leftHeader()) {
                RBTreeBlockHeader* sibling = node->parentHeader()->rightHeader();
                if (sibling->color() == RED) {
                    sibling->setColor(BLACK);
                    node->parentHeader()->setColor(RED);
                    rotateLeft(node->parentHeader());
                    sibling = node->parentHeader()->rightHeader();
                }
                if ((sibling->leftHeader() == nullptr
                     || sibling->leftHeader()->color() == BLACK)
                    && (sibling->rightHeader() == nullptr
                        || sibling->rightHeader()->color() == BLACK)) {
                    sibling->setColor(RED);
                    node = node->parentHeader();
                }
                else {
                    if (sibling->rightHeader() == nullptr
                        || sibling->rightHeader()->color() == BLACK) {
                        if (sibling->leftHeader() != nullptr)
                            sibling->leftHeader()->setColor(BLACK);
                        sibling->setColor(RED);
                        rotateRight(sibling);
                        sibling = node->parentHeader()->rightHeader();
                    }
                    sibling->setColor(node->parentHeader()->color());
                    node->parentHeader()->setColor(BLACK);
                    if (sibling->rightHeader() != nullptr)
                        sibling->rightHeader()->setColor(BLACK);
                    rotateLeft(node->parentHeader());
                    node = root;
                }
            }
            else {
                RBTreeBlockHeader* sibling = node->parentHeader()->leftHeader();
                if (sibling->color() == RED) {
                    sibling->setColor(BLACK);
                    node->parentHeader()->setColor(RED);
                    rotateRight(node->parentHeader());
                    sibling = node->parentHeader()->leftHeader();
                }
                if ((sibling->leftHeader() == nullptr
                     || sibling->leftHeader()->color() == BLACK)
                    && (sibling->rightHeader() == nullptr
                        || sibling->rightHeader()->color() == BLACK)) {
                    sibling->setColor(RED);
                    node = node->parentHeader();
                }
                else {
                    if (sibling->leftHeader() == nullptr
                        || sibling->leftHeader()->color() == BLACK) {
                        if (sibling->rightHeader() != nullptr)
                            sibling->rightHeader()->setColor(BLACK);
                        sibling->setColor(RED);
                        rotateLeft(sibling);
                        sibling = node->parentHeader()->leftHeader();
                    }
                    sibling->setColor(node->parentHeader()->color());
                    node->parentHeader()->setColor(BLACK);
                    if (sibling->leftHeader() != nullptr)
                        sibling->leftHeader()->setColor(BLACK);
                    rotateRight(node->parentHeader());
                    node = root;
                }
            }
        }
        node->setColor(BLACK);
    }

    // Utility function: Find Node with Minimum Value
    __device__ RBTreeBlockHeader* minValueNode(RBTreeBlockHeader* node)
    {
        RBTreeBlockHeader* current = node;
        while (current->leftHeader() != nullptr)
            current = current->leftHeader();
        return current;
    }

    // Utility function: Transplant nodes in Red-Black Tree
    __device__ void transplant(RBTreeBlockHeader* u, RBTreeBlockHeader* v)
    {
        if (u->parentHeader() == nullptr)
            root = v;
        else if (u == u->parentHeader()->leftHeader())
            u->parentHeader()->set_left_offset(v);
        else
            u->parentHeader()->set_right_offset(v);
        if (v != nullptr)
            v->set_parent_offset(u->parentHeader());
    }

public:
    // Constructor: Initialize Red-Black Tree
    __device__ RedBlackTree()
        : root(nullptr)
    {
    }

    // Public function: Insert a node into Red-Black Tree
    // Note: The node should already be allocated in the memory pool
    __device__ void insert(RBTreeBlockHeader* node)
    {
        // Initialize the node's tree pointers
        node->set_left_offset(nullptr);
        node->set_right_offset(nullptr);
        node->setColor(RED);

        RBTreeBlockHeader* parent = nullptr;
        RBTreeBlockHeader* current = root;
        
        // Find the insertion point
        while (current != nullptr) {
            parent = current;
            if (node->size() < current->size())
                current = current->leftHeader();
            else
                current = current->rightHeader();
        }
        
        node->set_parent_offset(parent);
        if (parent == nullptr)
            root = node;
        else if (node->size() < parent->size())
            parent->set_left_offset(node);
        else
            parent->set_right_offset(node);
        
        fixInsert(node);
    }

    // Public function: Remove a node from Red-Black Tree by size
    __device__ RBTreeBlockHeader* remove(int16_t targetSize)
    {
        RBTreeBlockHeader* node = root;
        RBTreeBlockHeader* z = nullptr;
        
        // Find the node with the target size
        while (node != nullptr) {
            if (node->size() == targetSize) {
                z = node;
                break; // Found exact match
            }
            if (node->size() <= targetSize) {
                node = node->rightHeader();
            }
            else {
                node = node->leftHeader();
            }
        }

        if (z == nullptr) {
            return nullptr; // Node not found
        }

        RBTreeBlockHeader* y = z;
        Color yOriginalColor = y->color();
        RBTreeBlockHeader* x = nullptr;
        
        if (z->leftHeader() == nullptr) {
            x = z->rightHeader();
            transplant(z, z->rightHeader());
        }
        else if (z->rightHeader() == nullptr) {
            x = z->leftHeader();
            transplant(z, z->leftHeader());
        }
        else {
            y = minValueNode(z->rightHeader());
            yOriginalColor = y->color();
            x = y->rightHeader();
            if (y->parentHeader() == z) {
                if (x != nullptr)
                    x->set_parent_offset(y);
            }
            else {
                transplant(y, y->rightHeader());
                y->set_right_offset(z->rightHeader());
                y->rightHeader()->set_parent_offset(y);
            }
            transplant(z, y);
            y->set_left_offset(z->leftHeader());
            y->leftHeader()->set_parent_offset(y);
            y->setColor(z->color());
        }
        
        if (yOriginalColor == BLACK) {
            fixDelete(x);
        }
        
        return z; // Return the removed node
    }

    // Public function: Find best fit (smallest block >= requested size)
    __device__ RBTreeBlockHeader* findBestFit(int16_t requestedSize)
    {
        RBTreeBlockHeader* current = root;
        RBTreeBlockHeader* bestFit = nullptr;
        
        while (current != nullptr) {
            if (current->size() >= requestedSize) {
                // This node could work, but check if there's a better fit
                bestFit = current;
                current = current->leftHeader(); // Look for smaller blocks
            }
            else {
                current = current->rightHeader(); // Need larger blocks
            }
        }
        
        return bestFit;
    }

    // Get root for debugging
    __device__ RBTreeBlockHeader* getRoot() {
        return root;
    }
};