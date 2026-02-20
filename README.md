# cuda_memory — GPU Shared-Memory Pool Allocators

## Overview

This repository implements multiple CUDA shared-memory allocators for fast, block-local dynamic allocation inside kernels. All allocators operate on **dynamic shared memory** (`extern __shared__`) and do not touch global memory (except native device malloc, included as a benchmark baseline).

The project provides:

- A common allocator API (`pool_init`, `pmalloc`, `pfree`)
- Six allocator implementations behind a unified macro interface
- Choosing which Allocator(s) can be done Run-time or Compile-time (via macros)
- Example usage patterns (see `blueprint.cu`) for Compile-time
- see (`benchmark_all`) for dynamic Runtime switching

---

## Requirements

- CMake ≥ 3.25
- CUDA Toolkit
- C++17 compiler
- An NVIDIA GPU supporting your chosen SM target
- (Optional) PyTorch/LibTorch for Torch integration

---

## Quick Start

```bash
# Configure and build
cmake -B build 
cmake --build build -j

# Run the benchmark (1000 iterations per allocator)
./build/tests/benchmark_all -i 1000

# Run with a specific random seed for fuzzy tests
./build/tests/benchmark_all -s 42 -i 500

# Make builds for specific allocators
cmake -B build -DUSE_THREAD_LOCAL_BEST_FIT
cmake -B build -DUSE_WARP_LOCAL_BEST_FIT
cmake -B build -DUSE_THREAD_LOCAL
cmake -B build -DUSE_WARP_LOCAL
cmake -B build -DUSE_FREELIST_ALLOCATOR
cmake -B build -DUSE_BST_ALLOCATOR
```

---

## Build Configuration

### CMake Options

| Option | Default | Description |
|---|---|---|
| `CMAKE_BUILD_TYPE` | `Release` | `Release` or `Debug` |
| `CMAKE_CUDA_ARCHITECTURES` | `"75 80 86 89"` | Target GPU architectures |
| `USE_CUDA` | `ON` | Build with CUDA support |
| `USE_TORCH` | `ON` | Build with PyTorch/LibTorch support |
| `BUILD_TESTS` | `ON` | Build test and benchmark executables |
| `BUILD_EXAMPLES` | `OFF` | Build example programs |

```bash
# Target a specific GPU architecture
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=89

# Debug build with verbose allocator prints
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CUDA_FLAGS="-DBENCH_VERBOSE=1"
```

### Compile-Time Allocator Selection

Define one of these to control which allocator the macros route to:

| Define | Allocator | Namespace |
|---|---|---|
| `USE_THREAD_LOCAL` | Thread-local first-fit | `thread_pool::pmalloc` |
| `USE_THREAD_LOCAL_BEST_FIT` | Thread-local best-fit | `thread_pool::pmalloc_best_fit` |
| `USE_WARP_LOCAL` | Warp-local first-fit | `warp_pool::pmalloc` |
| `USE_WARP_LOCAL_BEST_FIT` | Warp-local best-fit | `warp_pool::pmalloc_best_fit` |
| `USE_FREELIST_ALLOCATOR` | Global sorted freelist | `pmalloc_freelist::cmalloc` |
| `USE_BST_ALLOCATOR` | Global Red-Black Tree | `pmalloc_bst::cmalloc` |
| (none) | Native device `malloc`/`free` | — |

In CMake:
```cmake
target_compile_definitions(my_target PRIVATE USE_WARP_LOCAL_BEST_FIT)
```

---

## Allocator Implementations

## Universal Rules

-- All Threads must call pool_init
-- Dont double free
-- Dont free memory thats allocated from another thread (if the threads are in a different pools)

### Thread-Local Pool (`thread_pool`)

**Files:** `src/cuda/threadlocal.cu`, `src/include/allocator.cuh`

Each thread gets its own private memory pool carved from shared memory. No synchronization needed because no two threads ever touch the same pool.

**How it works:**
- Shared memory is divided equally among 32 threads (1 pool per thread)
- Each pool maintains a sorted descending free list using 16-bit offsets
- Blocks have an 8-byte header (bitfield: free/size/next/prev) and an 8-byte footer (back-pointer to header)
- Splitting creates a remainder block when the found block is significantly larger than requested
- Coalescing merges adjacent free blocks on free, using footer back-pointers for O(1) neighbor lookup
- Two strategies: first-fit (`pmalloc`) takes the largest available block, best-fit (`pmalloc_best_fit`) walks the sorted list to find the smallest block that fits

**API:**
```cpp
thread_pool::pool_init(total_shared_bytes);
void* p = thread_pool::pmalloc(size);
void* p = thread_pool::pmalloc_best_fit(size);
thread_pool::pfree(p);
```

**When to use:** You can structure the kernel so that allocation and free are thread-private, you want zero synchronization overhead, and you only need up to 32 allocator-participating threads.

**Ownership:** Strict. Only the owning thread can free what it allocated. Freeing a pointer from another thread's pool will corrupt that pool.

---

### Warp-Local Pool (`warp_pool`)

**Files:** `src/cuda/warplocal.cu`, `src/include/allocator.cuh`

Multiple threads share a pool, protected by per-pool spinlocks. Configurable `threads_per_pool` allows tuning the tradeoff between pool size and contention.

**How it works:**
- Shared memory is divided into `num_pools = ceil(blockDim.x / threads_per_pool)` pools
- Each pool has a spinlock (`atomicCAS`-based) so multiple threads can safely allocate/free from the same pool
- Thread-to-pool mapping: `pool_id = threadIdx.x / threads_per_pool`
- Same block layout, free list, splitting, and coalescing as `thread_pool`
- Auto-adjusts `threads_per_pool` upward if `num_pools` would exceed MAX_POOLS (32)
- With `threads_per_pool = 1`, behaves like `thread_pool` but with spinlock overhead — useful as a baseline to measure locking cost

**API:**
```cpp
warp_pool::pool_init(total_shared_bytes);         // 1 thread per pool (default)
warp_pool::pool_init(total_shared_bytes, 4);      // 4 threads share each pool
warp_pool::pool_init(total_shared_bytes, 32);     // full warp shares one pool
void* p = warp_pool::pmalloc(size);
void* p = warp_pool::pmalloc_best_fit(size);
warp_pool::pfree(p);
```

**When to use:** Many allocations with high parallelism, and you can keep allocation/free ownership within the same pool group. Larger pools mean less fragmentation but more contention; smaller pools mean less contention but more fragmentation.

**Warp boundary note:** If you want pools to never span hardware warps, choose `threads_per_pool` ∈ {1, 2, 4, 8, 16, 32}.

**SM requirement:** `threads_per_pool > 1` requires SM ≥ 7.0 (Volta+) for independent thread scheduling. On pre-Volta GPUs, threads in the same warp execute in lockstep, causing deadlock when one thread holds the lock while another in the same warp spins. The allocator automatically forces `threads_per_pool = 1` on SM < 7.0 via `#if __CUDA_ARCH__ < 700`.

---

### Global Freelist (`pmalloc_freelist`)

**Files:** `src/cuda/poolAlloc.cu`, `src/cuda/poolAlloc.cuh`

A sorted freelist allocator using relative offsets for all pointers. Each thread gets its own pool managed with a doubly-linked free list sorted by block size (largest first). Includes a bubble-sort pass after each allocation/free to maintain ordering.

**API:**
```cpp
pmalloc_freelist::init_gpu_buffer(pool_size_per_thread);
void* p = pmalloc_freelist::cmalloc(size);
pmalloc_freelist::cfree(p);
```

---

### Global BST (`pmalloc_bst`)

**Files:** `src/cuda/poolAllocBST.cu`, `src/cuda/poolAllocBST.cuh`

A Red-Black Tree based allocator for best-fit allocation. Each thread maintains a self-balancing BST of free blocks, enabling O(log n) best-fit search instead of O(n) list traversal.

**API:**
```cpp
pmalloc_bst::init_gpu_buffer(pool_size_per_thread);
void* p = pmalloc_bst::cmalloc(size);
pmalloc_bst::cfree(p);
```

---

### Native Device Malloc (Baseline)

CUDA's built-in `malloc`/`free` on the device heap. Uses global memory (not shared memory), so it's significantly slower but has much larger capacity. Included as a benchmark baseline only.

---

## Critical Usage Rules

These apply to **all** allocators in this project.

### Rule 1: `pool_init` must be reached by ALL threads in the block

Both `thread_pool` and `warp_pool` call `__syncthreads()` inside `pool_init`. If any thread in the block skips it, the block deadlocks permanently.

```cpp
__global__ void kernel(size_t shared_bytes) {
    extern __shared__ std::byte heap[];

    // ALL threads execute this — no conditional branching around it
    pool_init(shared_bytes);

    // Now any subset of threads may allocate/free
    void* p = pmalloc(64);
    if (p) {
        // use memory...
        pfree(p);
    }
}
```

### Rule 2: Dynamic shared memory size must include allocator metadata

The allocator stores metadata (pool_managers struct, free list heads, pool bounds) inside shared memory. The usable allocation capacity is the total shared memory minus the manager struct size minus per-pool overhead (headers + footers + alignment waste).

### Rule 3: 16-bit offsets cap each pool to ~64 KB

Block headers store next/prev pointers as 16-bit offsets, limiting each individual pool to 65,535 bytes. If total shared memory exceeds what your pool count can address, the excess is unused. To use larger shared memory efficiently, increase the pool count.

| Shared Memory | Minimum Pools Needed |
|---|---|
| 48 KB | 1 |
| 100 KB | 2 |
| 164 KB | 3 |
| 228 KB | 4 |

### Rule 4: Respect ownership

- **`thread_pool`**: Only the owning thread can free what it allocated. Cross-thread frees corrupt the pool.
- **`warp_pool`**: Any thread mapped to the same `pool_id` can free a pointer from that pool. Cross-pool frees corrupt the target pool.
- **`pmalloc_freelist` / `pmalloc_bst`**: Same as thread_pool — per-thread ownership.

### Rule 5: Only free exact user pointers

`pfree` / `cfree` expect the exact pointer returned by `pmalloc` / `cmalloc`. Interior pointers, offset pointers, or arbitrary addresses will corrupt the allocator state.

---

## Shared Memory Limits by Architecture

| Architecture | SM | Max Shared Memory Per Block |
|---|---|---|
| Kepler/Maxwell/Pascal | 3.x–6.x | 48 KB |
| Volta | 7.0 | 96 KB |
| Turing | 7.5 | 64 KB |
| Ampere | 8.0/8.6 | 100–164 KB |
| Ada Lovelace | 8.9 | 100 KB |
| Hopper | 9.0 | up to 228 KB |

Hardware shared memory may exceed the 64 KB per-pool cap. A single pool is still limited by the 16-bit offset format. To use the full shared memory, distribute it across multiple pools.

---

## Usage Patterns

### Macro Interface (Recommended)

The macro interface lets you write allocator-agnostic kernel code. Select an allocator at compile time and the macros route `pool_init`, `pmalloc`, and `pfree` to the right implementation:

```cpp
#include "allocator.cuh"
#include "poolAlloc.cuh"
#include "poolAllocBST.cuh"

// After setting a define (via CMake or #define), these macros are available:
//   pool_init(size)         — or pool_init(size, threads_per_pool) for warp variants
//   pmalloc(size)           — allocate
//   pfree(ptr)              — free
//   ALLOCATOR_NAME          — string name of the selected allocator
```

The warp-local variants use variadic macros, so the optional second parameter works:
```cpp
pool_init(total_bytes);        // default: 1 thread per pool
pool_init(total_bytes, 4);     // 4 threads share each pool
```

### Direct Namespace API

Call allocators directly without macros:

```cpp
#include "allocator.cuh"

__global__ void my_kernel(size_t shared_mem_size) {
    extern __shared__ std::byte heap[];

    thread_pool::pool_init(shared_mem_size);
    void* p = thread_pool::pmalloc(64);
    thread_pool::pfree(p);
}
```

```cpp
__global__ void my_kernel(size_t shared_mem_size) {
    extern __shared__ std::byte heap[];

    warp_pool::pool_init(shared_mem_size, 4);
    void* q = warp_pool::pmalloc(64);
    warp_pool::pfree(q);
}
```

### Runtime Switching (Benchmarking)

`benchmark_all.cu` uses `__managed__` global variables to switch allocators at runtime without recompiling. Set `g_mode` from the host before each kernel launch:

```cpp
__device__ __managed__ int g_mode = MODE_THREAD_FIRST_FIT;

// Host side:
g_mode = MODE_WARP_BEST_FIT;
cudaDeviceSynchronize();
my_kernel<<<blocks, threads, shared_mem>>>();
```

See `tests/benchmark_all.cu` for the full implementation including timing harness and fuzzy testing across all modes.

---

## Writing Your Own Kernel

See `tests/blueprint.cu` for a complete minimal example. Here is the general pattern:

```cpp
#include "allocator.cuh"

__global__ void my_kernel(size_t shared_mem_size) {
    extern __shared__ std::byte heap[];

    // Step 1: ALL threads must call pool_init (contains __syncthreads)
    pool_init(shared_mem_size);

    // Step 2: Allocate
    int* data = (int*)pmalloc(4 * sizeof(int));
    if (data == nullptr) {
        // Pool exhausted — handle gracefully
        return;
    }

    // Step 3: Use the memory
    for (int i = 0; i < 4; i++) {
        data[i] = threadIdx.x * 100 + i;
    }

    // Step 4: Free when done
    pfree(data);
}

int main() {
    constexpr int threads = 32;
    constexpr size_t shared_mem = 48 * 1024;

    my_kernel<<<1, threads, shared_mem>>>(shared_mem);
    cudaDeviceSynchronize();
}
```

### Implementing Your Own Allocator

If you want to add a new allocator to this project:

1. Create a new `.cu` / `.cuh` pair under `src/cuda/`
2. Wrap everything in a unique namespace (e.g., `namespace my_allocator { }`)
3. Implement at minimum: `pool_init`, `pmalloc`, `pfree`
4. Add the `.cu` file to the glob in `src/CMakeLists.txt`
5. Add a macro define and switch case to the macro header and `benchmark_all.cu`
6. Decide and document your ownership model (thread-private, group-private, block-global)

**Debug recommendations:** Add checks for pointer range validation, header/footer consistency, double-free detection, and free list ordering invariants. Gate these behind a `#ifdef ALLOCATOR_DEBUG` so they compile out in release.

---

## Troubleshooting

**Deadlock during initialization**
Some threads did not execute `pool_init` but others reached the `__syncthreads()` inside it. Ensure all threads in the block call `pool_init` unconditionally with no early returns or conditional branches before it.

**`pmalloc` returns `nullptr` unexpectedly**
- Insufficient shared memory passed at kernel launch
- Pool size capped at ~64 KB per pool due to 16-bit offsets
- Fragmentation — try best-fit or adjust allocation patterns
- Calling from threads not supported by the allocator (`thread_pool` only supports threadIdx.x 0–31)

**Memory corruption**
- Freeing pointers from a different pool than they were allocated from
- Freeing non-allocator pointers or interior pointers
- Double free
- Enable debug checks and validate your ownership policy

**nvlink: Multiple definition errors**
Each allocator's `.cu` file must be wrapped in its namespace. If you see duplicate symbol errors, check that the `namespace { }` wraps the entire file contents including all `__device__` variables and helper functions.

**nvlink: Undefined reference errors**
- Check that the `.cu` file is listed in the `file(GLOB ...)` in `src/CMakeLists.txt`
- Check that the namespace in the `.cu` matches the namespace in the `.cuh`
- Remove include guards (`#ifndef` / `#endif`) from `.cu` source files — they belong on headers only
- Do a clean rebuild: `cmake --build build --clean-first -j`

---

