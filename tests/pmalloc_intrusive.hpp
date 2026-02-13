#pragma once

#include <cstddef>
namespace pmem_intrusive {


inline constexpr std::size_t KiB = 1024ULL;
inline constexpr std::size_t MiB = 1024ULL * KiB;
inline constexpr std::size_t GiB = 1024ULL * MiB;


inline constexpr std::size_t DEFAULT_POOL_SIZE = 8 * GiB;


void init_pool();                              // initialize with DEFAULT_POOL_SIZE
void init_pool_with_bytes(std::size_t bytes);  // initialize with explicit size (required)
void destroy_pool();                           // delete pool + reset globals

// Allocator
void* pmalloc(std::size_t size);              
void  pfree(void* p);                         
void* pmemcpy(void* dst, const void* src, std::size_t n);

// other
void* memfill(void* s, int c, std::size_t n);
std::size_t total_free_bytes(); // sum of sizes of all free blocks
std::size_t largest_free_block(); // size in bytes of largest free chunk

} 
