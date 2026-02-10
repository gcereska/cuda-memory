#include <stdio.h>
#include <printf.h>

#include "cpuPoolMalloc.h"

int main()
{

    // printf("program starts\n");

    init_cpu_buffer();

    printf("init complete\n");

    void *small_test = cmalloc(128);

    debug_print_buffer();

    void *med_test = cmalloc(512);

    debug_print_buffer();

    void *large_test = cmalloc(1028);

    debug_print_buffer();

    printf("-----------------------------\n");

    cfree((char*)med_test + 4);

    debug_print_buffer();
    debug_print_free_list();

    cfree(large_test);

    debug_print_buffer();

    cfree(small_test);

    debug_print_buffer();

    free_buffer();
    return 0;
}