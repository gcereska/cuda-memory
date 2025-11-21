// #include <stdio.h>
// #include <stdlib.h>

// #include "CPUMemBuffer.h"

// #define INITIAL_CAPACITY 16

// int pq_compare(BlockHeader *lhs,BlockHeader *rhs){ //its a max heap PQ, return 1 if bigger to maintain max
//     if(lhs->size > rhs->size){
//         return 1;
//     }
//     return 0;
// };

// typedef struct {
//     void **data;      // array of pointers to your items
//     size_t size;      // number of elements
//     size_t capacity;  // allocated space
// } PriorityQueue;

// PriorityQueue *pq_create() {
//     PriorityQueue *pq = malloc(sizeof(PriorityQueue));
//     pq->data = malloc(INITIAL_CAPACITY * sizeof(void *));
//     pq->size = 0;
//     pq->capacity = INITIAL_CAPACITY;
//     return pq;
// }

// void pq_free(PriorityQueue *pq) {
//     free(pq->data);
//     free(pq);
// }

// static void swap(void **a, void **b) {
//     void *tmp = *a;
//     *a = *b;
//     *b = tmp;
// }

// static void bubble_up(PriorityQueue *pq, size_t idx) {
//     while (idx > 0) {
//         size_t parent = (idx - 1) / 2;
//         if (pq_compare(pq->data[idx], pq->data[parent]) <= 0) break;
//         swap(&pq->data[idx], &pq->data[parent]);
//         idx = parent;
//     }
// }

// static void bubble_down(PriorityQueue *pq, size_t idx) {
//     while (1) {
//         size_t left = 2 * idx + 1;
//         size_t right = 2 * idx + 2;
//         size_t largest = idx;

//         if (left < pq->size && pq_compare(pq->data[left], pq->data[largest]) > 0)
//             largest = left;
//         if (right < pq->size && pq_compare(pq->data[right], pq->data[largest]) > 0)
//             largest = right;

//         if (largest == idx) break;
//         swap(&pq->data[idx], &pq->data[largest]);
//         idx = largest;
//     }
// }

// void pq_push(PriorityQueue *pq, void *item) {
//     if (pq->size == pq->capacity) {
//         pq->capacity *= 2;
//         pq->data = realloc(pq->data, pq->capacity * sizeof(void *));
//     }
//     pq->data[pq->size] = item;
//     bubble_up(pq, pq->size);
//     pq->size++;
// }

// BlockHeader* pq_top(PriorityQueue *pq) {
//     if (pq->size == 0) return NULL;
//     return pq->data[0];
// }

// BlockHeader* pq_pop(PriorityQueue *pq) { //returns head on pop
    
//     if (pq->size == 0) return NULL;
//     void *result = pq->data[0];
//     pq->data[0] = pq->data[pq->size - 1];
//     pq->size--;
//     bubble_down(pq, 0);
//     return result;
// }
