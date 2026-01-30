1. The makefile compiles both fuzzy-test-gpu and my custom allocator-test.cu

2. threadlocal is 1 thread per warp. each warp gets own pool. Exactly 32 threads/warps/pools

3. warplocal can allocate any number of pools, 1 warp per pool,  1-32 threads per warp/pool

4. allocator_test.cu is AI generated, fuzzy-test-gpu.cu is adapted from the test files, the rest of the test files assume global memory to be consistent 
across kernel calls so I couldnt figure out a way to use them. Linalg test requires a math.h that wasnt uploaded.

5. threadlocal.cu works with fuzzy-test-gpu.cu and allocator_test.cu

6. I implemented nmake clean and nmake run

7. pmalloc best fit currently doesn't work there is a new bug I recently discovered.
   It crashed on my test files [TEST 9] with an error relating to "illegal memory access", but it passes fuzzy test
