#pragma once

#include <cumem/cuda/allocator.cuh>
#include <cumem/cuda/poolAlloc.cuh>
#include <cumem/cuda/poolAllocBST.cuh>

#if defined(USE_THREAD_LOCAL_BEST_FIT)
#define ALLOCATOR_NAME "Thread Local (Best Fit)"
#define pool_type thread_pool
#define pool_init thread_pool::pool_init
#define pmalloc thread_pool::pmalloc_best_fit
#define pfree thread_pool::pfree

#elif defined(USE_WARP_LOCAL_BEST_FIT)
#define ALLOCATOR_NAME "Warp Local (Best Fit)"
#define pool_type warp_pool
#define pool_init warp_pool::pool_init
#define pmalloc warp_pool::pmalloc_best_fit
#define pfree warp_pool::pfree

#elif defined(USE_THREAD_LOCAL)
#define ALLOCATOR_NAME "Thread Local (First Fit)"
#define pool_type thread_pool
#define pool_init thread_pool::pool_init
#define pmalloc thread_pool::pmalloc
#define pfree thread_pool::pfree

#elif defined(USE_WARP_LOCAL)
#define ALLOCATOR_NAME "Warp Local (First Fit)"
#define pool_type warp_pool
#define pool_init warp_pool::pool_init
#define pmalloc warp_pool::pmalloc
#define pfree warp_pool::pfree

#elif defined(USE_FREELIST_ALLOCATOR)
#define ALLOCATOR_NAME "Julian (Standard Freelist)"
#define pool_type list_pool
#define pool_init list_pool::pool_init
#define pmalloc list_pool::pmalloc
#define pfree list_pool::pfree

#elif defined(USE_BST_ALLOCATOR)
#define ALLOCATOR_NAME "Julian (BST)"
#define pool_type bst_pool
#define pool_init bst_pool::pool_init
#define pmalloc bst_pool::pmalloc
#define pfree bst_pool::pfree

#else
#define ALLOCATOR_NAME "Native Device Malloc"
#define pool_init /* no pool init */
#define pmalloc malloc
#define pfree free
#endif
