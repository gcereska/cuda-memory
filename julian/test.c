#include <stdio.h>
#include <printf.h>

#include "cpuPoolMalloc.h"



int main(){

    //printf("program starts\n");


    init_cpu_buffer();


    printf("init complete\n");


    void* small_test = cmalloc(128);
    printf("alloc complete\n");
    printf("%p\n",small_test);

    debug_print_buffer();

    void* med_test = cmalloc(512);
    printf("alloc complete\n");
    printf("%p\n",med_test);
    
    debug_print_buffer();
    
    
    void* large_test = cmalloc(1028);
    printf("alloc complete\n");
    printf("%p\n",large_test);
    printf("all allocations complete\n");
    
    debug_print_buffer();
    
    printf("-----------------------------\n");

    cfree(med_test);
    printf("medium free complete\n");
    
    debug_print_buffer();
    debug_print_free_list();
    
    printf("\n");
    cfree(large_test);
    printf("large free complete\n");
    
    debug_print_buffer();
    
    printf("\n");
    cfree(small_test);
    printf("small free complete\n");
    
    debug_print_buffer();
    
    printf("\n");

    free_buffer();
    return 0;
}