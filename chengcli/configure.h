#pragma once

// C/C++
#include <cstddef>

// CUDA FLAG (ENABLE_CUDA or DISABLE_CUDA)
#define DISABLE_CUDA

#ifdef __CUDACC__
  #define DISPATCH_MACRO __host__ __device__
#else
  #define DISPATCH_MACRO
#endif
