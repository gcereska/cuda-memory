//allocator.cuh
#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

// Initializes the shared memory heap. 
// Must be called by ALL threads in the block for my implementation
namespace thread_pool {
    __device__ void  pool_init(std::size_t total_bytes);

    __device__ void* pmalloc(std::size_t size);

    __device__ void* pmalloc_best_fit(std::size_t size);

    __device__ void  pfree(void* p);
}

namespace warp_pool {
    __device__ void  pool_init(std::size_t total_bytes);

    __device__ void* pmalloc(std::size_t size);

    __device__ void* pmalloc_best_fit(std::size_t size);

    __device__ void  pfree(void* p);
}
