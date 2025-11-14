#include <stdio.h>
#include <printf.h>

#include "cpu_custom_malloc.h"

const size_t SIZE = 15;

int main()
{

    // printf("program starts\n");

    init_cpu_buffer();

    printf("init complete\n");

    void *pointer_storage[SIZE];

    for (size_t i = 0; i < SIZE; i++)
    {
        void *temp_pointer = cmalloc(9);
        pointer_storage[i] = temp_pointer;
    }



    printf("--------------------ENTIRE BUFFER---------------------\n");
    debug_print_buffer();
    printf("----------------------FREE LIST---------------------\n");
    debug_print_free_list();
    printf("----------------------FULL LIST---------------------\n");
    debug_print_full_list();
    printf("\n");

    for (size_t i = 0; i < SIZE; i += 5)
    {
        cfree(pointer_storage[i]);
    }


    printf("--------------------ENTIRE BUFFER---------------------\n");
    debug_print_buffer();
    printf("----------------------FREE LIST---------------------\n");
    debug_print_free_list();
    printf("----------------------FULL LIST---------------------\n");
    debug_print_full_list();


    printf("\n");

    // printf("completing step 1\n");

    for (size_t i = 0; i < SIZE; i += 2)
    {
        if (i % 5 != 0)
        {
            cfree(pointer_storage[i]);
        }
    }

    // printf("starting step 2\n");

    printf("--------------------ENTIRE BUFFER---------------------\n");
    debug_print_buffer();
    printf("----------------------FREE LIST---------------------\n");
    debug_print_free_list();
    printf("----------------------FULL LIST---------------------\n");
    debug_print_full_list();
    // printf("completing step 2\n");

    printf("\n");


    for (size_t i = 0; i < SIZE; i++)
    {
        if (i % 2 != 0 && i % 5 != 0)
        {
            cfree(pointer_storage[i]);
        }
    }
    printf("--------------------ENTIRE BUFFER---------------------\n");
    debug_print_buffer();
    printf("----------------------FREE LIST---------------------\n");
    debug_print_free_list();
    printf("----------------------FULL LIST---------------------\n");
    debug_print_full_list();
    // printf("completing step 3\n");

    

    printf("\n");

    free_buffer();
    return 0;
}