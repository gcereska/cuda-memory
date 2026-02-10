#include <cstdlib>
#include <memorypool/cpu/poolalloc.h>

void test_alloc_ints() {
    int *ptrs[14];
    for (int i = 0; i < 14; i++) {
        if (i % 5 == 0) {
            int *ptr = (int *)poolmalloc(10 * sizeof(int));
            *ptr = i;
            ptrs[i] = ptr;
        } else {
            int *ptr = (int *)poolmalloc(sizeof(int));
            *ptr = i;
            ptrs[i] = ptr;
        }
        if (i == 7) {
            poolfree(ptrs[6]);
            ptrs[6] = NULL;
        }
    }

    printlayout();

    for (int i = 0; i < 14; i++) {
        if (ptrs[i] != NULL) {
            poolfree(ptrs[i]);
        }
    }
}

void test_alloc_chars() {
    char *ptrs[20];
    for (int i = 0; i < 20; i++) {
        char *ptr = (char *)poolmalloc(sizeof(int));
        *ptr = i;
        ptrs[i] = ptr;
    }
    for (int i = 0; i < 20; i++) {
        poolfree(ptrs[i]);
    }
}

int main(void) {
    poolinit();

    test_alloc_ints();
    test_alloc_chars();
}
