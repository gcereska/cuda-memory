# define default parameters

# whether to use memory pool
if(NOT MEMPOOL OR NOT DEFINED MEMPOOL)
  set(MEMORY_POOL_OPTION "NOT_USE_MEMORY_POOL")
else()
  set(MEMORY_POOL_OPTION "USE_MEMORY_POOL")
endif()

set(CPU_MEM_POOL_SIZE 64000)
set(GPU_MEM_POOL_SIZE 800)

set(MEM_MAX_BLOCKS 100)
set(MEM_MAX_THREADS 1100000)
