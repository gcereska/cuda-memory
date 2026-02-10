#pragma once

#include <cstddef> 

namespace pmem_binheap {


inline constexpr std::size_t KiB = 1024ULL;
inline constexpr std::size_t MiB = 1024ULL * KiB;
inline constexpr std::size_t GiB = 1024ULL * MiB;
inline constexpr std::size_t DEFAULT_POOL_SIZE = 8 * GiB;


void init_pool();                              // uses DEFAULT_POOL_SIZE
void init_pool_with_bytes(std::size_t bytes);  // required: explicit size init
void destroy_pool();                           // delete pool + reset globals


void* pmalloc(std::size_t size);              
void  pfree(void* p);                  
void* pmemcpy(void* dst, const void* src, std::size_t n);

//others
void* memfill(void* s, int c, std::size_t n);
std::size_t total_free_bytes();   // total bytes across all free blocks
std::size_t largest_free_block(); // largest single free block size

} // namespace pmem_binheap

