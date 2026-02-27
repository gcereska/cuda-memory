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

### Build & Run Scripts

Three helper scripts in `scripts/` automate multi-configuration builds:

**`build_all.sh`** — Clean builds every allocator variant into its own build directory:
```bash
./scripts/build_all.sh
```
Creates `build_USE_THREAD_LOCAL/`, `build_USE_WARP_LOCAL/`, `build_native/`, etc. Each directory has the full project compiled with that allocator selected. Deletes and recreates each build directory from scratch.

**`rebuild_all.sh`** — Incremental rebuild of all existing build directories (no clean):
```bash
./scripts/rebuild_all.sh
```
Use this after editing source files. Much faster than `build_all.sh` since it only recompiles changed files.

**`run_all.sh`** — Runs a test executable with our macro setup across all build directories.

```bash
./scripts/run_all.sh
```
Edit the `TARGETS` array at the top of the script to choose which test to run (e.g., `blueprint`, `benchmark_all`). It looks for the executable in each `build_*/tests/` directory and skips any that don't exist. However, tests like `benchmark_all` don't benefit from this since they use runtime switching via unified memory globals and ignore the compile-time macros entirely. Running it from any single build directory does the same thing.

**Run `benchmark_all`**
```bash
./build_native/tests/benchmark_all
```

---

## Build Configuration

### CMake Options

| Option | Default | Description |
|---|---|---|
| `CMAKE_BUILD_TYPE` | `Release` | `Release` or `Debug` |
| `CMAKE_CUDA_ARCHITECTURES` | `"75 80 86 89"` | Target GPU architectures |
| `CUDA` | `ON` | Build with CUDA support |
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
### Super Basic Rules For Context

- All Threads must call pool_init(dyn_smem_size)
- Pass in shared memory size into kernel ex:
- simple_test_kernel<<<1, 32, dyn_smem_size>>>(dyn_smem_size);
- Dont double free
- Dont free memory thats allocated from another thread

## Allocator Implementations

### Thread-Local Pool (`thread_pool`)

**Files:** `src/cuda/threadlocal.cu`, `src/include/allocator.cuh`

Each thread gets its own private memory pool carved from shared memory.

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

**When to use:** You can structure the kernel so that allocation and free are thread-private, you only need up to 32 allocator-participating threads.

**Note** If you want 16 threads each with their individual pools, use warp_local with 16 threads and 1 thread per pool. There is some overhead with atomicCAS still always being used in warp_local, and I think in the future I can set it so that thread_local can take a dynamic (within 1-32) number of threads, splitting memory equally between each thread.



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

**NOTE** Pool mapping uses `threadIdx.x` directly, so 2D/3D block configurations will not map threads to pools correctly. To support multi-dimensional blocks, `get_pool_id()` would need to compute a linear thread index (`threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y`) before dividing by `threads_per_pool`. The same limitation applies to `thread_pool`. I will look into this and update it soon.

**When to use:** Many allocations with high parallelism. Larger pools mean less fragmentation but more contention. smaller pools mean less contention but more fragmentation.

**Warp boundary note:** If you want pools to never span hardware warps, choose `threads_per_pool` ∈ {1, 2, 4, 8, 16, 32}.

**SM requirement:** `threads_per_pool > 1` requires SM ≥ 7.0 (Volta+) for independent thread scheduling. On pre-Volta GPUs, threads in the same warp execute in lockstep, causing deadlock when one thread holds the lock while another in the same warp spins. I think I can add something where the allocator automatically forces `threads_per_pool = 1` on SM < 7.0 via `#if __CUDA_ARCH__ < 700`. For now just don't use warp_local with those architectures.

---

### Freelist (`pmalloc_freelist`)

**Files:** `src/cuda/poolAlloc.cu`, `src/cuda/poolAlloc.cuh`, `typeDefs.cuh`

A sorted freelist allocator using relative offsets for all pointers. Each thread gets its own pool managed with a doubly-linked free list sorted by block size largest first (just like thread_local). Includes a bubble-sort pass after each allocation/free to maintain ordering. Ask Julian for more specific details if needed.

**API:**
```cpp
pmalloc_freelist::init_gpu_buffer(pool_size_per_thread);
void* p = pmalloc_freelist::cmalloc(size);
pmalloc_freelist::cfree(p);
```

---

### BST (`pmalloc_bst`)

**Files:** `src/cuda/poolAllocBST.cu`, `src/cuda/poolAllocBST.cuh`, `typeDefs.cuh`

A Red-Black Tree based allocator for best-fit allocation. Each thread maintains a self-balancing BST of free blocks. Ask Julian for more specific details if needed.

**API:**
```cpp
pmalloc_bst::init_gpu_buffer(pool_size_per_thread);
void* p = pmalloc_bst::cmalloc(size);
pmalloc_bst::cfree(p);
```

---

### Native Device Malloc (Baseline)

CUDA's built-in `malloc`/`free` on the device heap. Uses global memory, so its slow but has much larger capacity. Included as a benchmark baseline only.

- To set custom heap size
- `CUDA_CHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 256ULL * 1024ULL * 1024ULL));`

---

## Critical Usage Rules

These apply to **all** allocators in this project.

### Rule 1: `pool_init` must be reached by ALL threads in the block

Both `thread_pool` and `warp_pool` call `__syncthreads()` inside `pool_init`. If any thread in the block skips it, the block deadlocks permanently.

```cpp
__global__ void kernel(size_t shared_bytes) {
    extern __shared__ std::byte heap[];

    // ALL threads execute this -> no conditional branching around it
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

The allocator stores metadata (pool_managers struct, free list heads, pool bounds) inside shared memory. The usable allocation capacity is the total shared memory minus the manager struct size minus per-pool overhead (headers + footers + alignment waste). Julian's allocators have something similar.

### Rule 3: 16-bit offsets cap each pool to ~64 KB

Block headers store next/prev pointers as 16-bit offsets, limiting each individual pool to 65,535 bytes. If total shared memory exceeds what your pool count can address, the excess is unused. To use larger shared memory efficiently, increase the pool count. This really only matters with warp_local, the other allocators default to 32 equally sized pools.
| Shared Memory | Minimum Pools Needed |
|---|---|
| 48 KB | 1 |
| 100 KB | 2 |
| 164 KB | 3 |
| 228 KB | 4 |

### Rule 4: Respect ownership

- **`thread_pool`**: Only the owning thread can free what it allocated. Cross-thread frees corrupt the pool.
- **`warp_pool`**: Any thread mapped to the same `pool_id` can free a pointer from that pool. Cross-pool frees corrupt the target pool.
- **`pmalloc_freelist` / `pmalloc_bst`**: Same as thread_pool -> per-thread ownership.

### Rule 5: Only free exact user pointers

`pfree` / `cfree` expect the exact pointer returned by `pmalloc` / `cmalloc`. arbitrary addresses will break everything.

---

## Shared Memory Limits by Architecture

| Architecture | SM | Example GPUs | Max Shared Memory Per Block |
|---|---|---|---|
| Kepler | 3.5/3.7 | GTX 780, Tesla K40/K80 | 48 KB |
| Maxwell | 5.0/5.2 | GTX 970/980, Tesla M40 | 48 KB |
| Pascal | 6.0/6.1 | GTX 1080, Tesla P100/V100 | 48 KB |
| Volta | 7.0 | Tesla V100, Titan V | 96 KB |
| Turing | 7.5 | RTX 2080, Tesla T4 | 64 KB |
| Ampere | 8.0/8.6 | RTX 3090, A100, A6000 | 100–164 KB |
| Ada Lovelace | 8.9 | RTX 4090, L40, RTX 4070 | 100 KB |
| Hopper | 9.0 | H100, H200 | up to 228 KB |

---

## Querying and Opting Into Maximum Shared Memory

The default shared memory limit per block is typically 48 KB. To use more, you must query the device and opt in per kernel:
```cpp
cudaDeviceProp prop{};
cudaGetDeviceProperties(&prop, 0);

// Default limit (usually 48 KB)
size_t default_smem = prop.sharedMemPerBlock;

// Maximum available after opt-in (e.g., 100 KB on Ada Lovelace)
size_t max_smem = prop.sharedMemPerBlockOptin;

// Reserve 16 bytes for CUDA internal overhead (kernel args, driver)
// My tests return errors if I don't subtract 16 bytes
// That might not be the case on other machines
size_t usable_smem = max_smem - 16;

// Opt in for each kernel that needs more than 48 KB
cudaFuncSetAttribute(my_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, usable_smem);

// Launch with the full amount
my_kernel<<<1, 32, usable_smem>>>(usable_smem);
```

Without the `cudaFuncSetAttribute` call, launching with more than `sharedMemPerBlock` bytes will return a CUDA error. The opt-in is required because using more shared memory reduces occupancy — fewer blocks can run concurrently per SM, so CUDA makes you explicitly request it.

## Usage Patterns

### Macro Interface

The macro interface lets you select an allocator at compile time and the macros route `pool_init`, `pmalloc`, and `pfree` to the right implementation:

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

The warp-local variants use __VA_ARGS__ so the second parameter works:
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

    // ALL threads must call pool_init (contains __syncthreads)
    pool_init(shared_mem_size);

    // Allocate
    int* data = (int*)pmalloc(4 * sizeof(int));
    if (data == nullptr) {
        // Pool exhausted
        return;
    }

    // Use the memory
    for (int i = 0; i < 4; i++) {
        data[i] = threadIdx.x * 100 + i;
    }

    // Free when done
    pfree(data);
}

int main() {
    constexpr int threads = 32;
    constexpr size_t shared_mem = 48 * 1024;

    my_kernel<<<1, threads, shared_mem>>>(shared_mem);
    cudaDeviceSynchronize();
}
```

### Steps if we want to add even more allocators

If you want to add a new allocator to this project:

1. Create a new `.cu` / `.cuh` pair under `src/cuda/`
2. Wrap everything in a unique namespace (e.g., `namespace my_allocator { }`)
3. Implement at minimum: `pool_init`, `pmalloc`, `pfree`
4. Add the `.cu` file to the glob in `src/CMakeLists.txt`
5. Add a macro define and switch case to the macro header and `benchmark_all.cu`

- Cmake creates one static library with the object files for all allocator versions so the namespaces are required.

**Future Debug recommendations:** Add checks for pointer range validation, header/footer consistency, double-free detection, and free list ordering invariants. Gate these behind a `#ifdef ALLOCATOR_DEBUG` so they compile out in release.

---

## Troubleshooting

**`pmalloc` returns `nullptr` unexpectedly**
- Insufficient shared memory passed at kernel launch
- Pool size capped at ~64 KB per pool due to 16-bit offsets
- Fragmentation — try best-fit or adjust allocation patterns
- Calling from threads not supported by the allocator (`thread_pool` only supports threadIdx.x 0–31)

## Additional Notes I couldn't figure out where to add

- Cmake creates one static library with the object files for all allocator versions so the namespaces are required.
- To add new tests you have to manually add them to the Cmake GLOBs.

---
