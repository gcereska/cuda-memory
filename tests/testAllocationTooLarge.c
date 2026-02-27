#include <printf.h>
#include <stdio.h>

#include "cpuPoolMalloc.h"

int main() {
  // printf("program starts\n");

  init_cpu_buffer();

  printf("init complete\n");

  void* small_test = cmalloc(4000);
  printf("alloc complete\n");
  printf("%p\n", small_test);

  debug_print_buffer();

  free_buffer();
  return 0;
}
