cd "...\cuda-memory"

// debug build setup
cmake -S . -B build -LAH
cmake -S . -B build --log-level=STATUS


// before build
cmake -S . -B build

// debug build setup
cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug

// to build/rebuild
cmake --build build

// debug build/rebuild
cmake --build debug

// Cmake Flags
// -DBUILD_TESTS=OFF
// -DCMAKE_BUILD_TYPE=Debug
// cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES="89"          for rtx 4090 architecture


// to run tests on windows // cmake is inherently different and decides to make seperate folder for release and debug on windows and only windows
.\build\tests\Release\allocator_test.exe
.\build\tests\Release\fuzzy-test-gpu.exe -s 67
// -s is to indicate rng seed

// on Linux
./build/tests/allocator_test
./build/tests/fuzzy-test-gpu -s 67
./build/tests/fuzzy-test-gpu-julian-impl
./build/tests/test-cuda-julian-impl

./build/tests/test
./build/tests/testAllocationTooLarge
./build/tests/testInvalidDeletion
./build/tests/testLightFragmentation
./build/tests/testRigorousFragmentation

./build/tests/allocator_bench_all

./build/tests/benchmark_all



// test for python torch
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0))"



//install python envrionmenton wsl
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu130


// RUN SCRIPTS
bash scripts/build_all.sh && bash scripts/run_all.sh > output.txt

./build_native/tests/benchmark_all > benchmark_all_output.txt

//MAKE SPECIFIC BUILDS
cmake -B build -DUSE_THREAD_LOCAL_BEST_FIT
cmake -B build -DUSE_WARP_LOCAL_BEST_FIT
cmake -B build -DUSE_THREAD_LOCAL
cmake -B build -DUSE_WARP_LOCAL
cmake -B build -DUSE_FREELIST_ALLOCATOR
cmake -B build -DUSE_BST_ALLOCATOR

// WARP LOCAL MULTIPLE THREADS PER POOL
// CAPPED TO 32 POOLS
warp_pool::pool_init(bytes)        // 1 thread per pool
warp_pool::pool_init(bytes, 4)     // 4 threads per pool

--------   AFTER MACRO   -----------
           pmalloc(bytes)        // 1 thread per pool
           pmalloc(bytes, 4)     // 4 threads per pool
