cmake_minimum_required(VERSION 3.18)
project(cuda-memory LANGUAGES CXX CUDA)

# Use C++17 and CUDA 17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Enable "Relocatable Device Code" (RDC)
# So the test files call functions inside threadlocal.cu
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Create Library named "allocator_lib".
add_library(allocator_lib STATIC src/threadlocal.cu)

# Tell it where to find include
target_include_directories(allocator_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Link the CUDA device runtime (needed for RDC)
target_link_libraries(allocator_lib PUBLIC cudadevrt)

# Tells it to build the test files
add_executable(test_thread tests/allocator_test.cu)
target_link_libraries(test_thread PRIVATE allocator_lib)

add_executable(fuzzy_thread tests/fuzzy-test-gpu.cu)
target_link_libraries(fuzzy_thread PRIVATE allocator_lib)

message(STATUS "Setup complete")

message(STATUS "run cmake --build build")