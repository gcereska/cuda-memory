cmake_minimum_required(VERSION 3.25)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

option(BUILD_EXAMPLES "Build examples" OFF)
option(BUILD_TESTS "Build tests" ON)
option(USE_CUDA "Build with CUDA support" ON)


option(USE_THREAD_LOCAL_BEST_FIT "Use Thread Local Best Fit allocator" OFF)
option(USE_WARP_LOCAL_BEST_FIT   "Use Warp Local Best Fit allocator" OFF)
option(USE_THREAD_LOCAL          "Use Thread Local First Fit allocator" OFF)
option(USE_WARP_LOCAL            "Use Warp Local First Fit allocator" OFF)
option(USE_FREELIST_ALLOCATOR    "Use Freelist allocator" OFF)
option(USE_BST_ALLOCATOR         "Use BST allocator" OFF)

if (USE_THREAD_LOCAL_BEST_FIT)
  add_compile_definitions(USE_THREAD_LOCAL_BEST_FIT)
  message(STATUS "Configured to use: Thread Local (Best Fit)")

elseif (USE_WARP_LOCAL_BEST_FIT)
  add_compile_definitions(USE_WARP_LOCAL_BEST_FIT)
  message(STATUS "Configured to use: Warp Local (Best Fit)")

elseif (USE_THREAD_LOCAL)
  add_compile_definitions(USE_THREAD_LOCAL)
  message(STATUS "Configured to use: Thread Local (First Fit)")

elseif (USE_WARP_LOCAL)
  add_compile_definitions(USE_WARP_LOCAL)
  message(STATUS "Configured to use: Warp Local (First Fit)")

elseif (USE_FREELIST_ALLOCATOR)
  add_compile_definitions(USE_FREELIST_ALLOCATOR)
  message(STATUS "Configured to use: Freelist Allocator")

elseif (USE_BST_ALLOCATOR)
  add_compile_definitions(USE_BST_ALLOCATOR)
  message(STATUS "Configured to use: BST Allocator")

else()
  # No flags were turned on. Native Device Malloc for Universal Tests
  message(STATUS "Configured to use: Native Device Malloc")
endif()

project(
  cuda_memory
  LANGUAGES CXX C
  DESCRIPTION "Memory Pool Allocator for C/C++ and CUDA"
)



# search for <Package>_ROOT
cmake_policy(SET CMP0074 NEW)

#find_program(Python3_EXECUTABLE NAMES python)
#find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
#execute_process(
#  COMMAND ${Python3_EXECUTABLE} -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))"
#  RESULT_VARIABLE _ABI_RESULT
#  OUTPUT_VARIABLE _ABI_OUTPUT
#  OUTPUT_STRIP_TRAILING_WHITESPACE
#)

#if(NOT _ABI_RESULT EQUAL 0)
#  message(FATAL_ERROR "Failed to detect Torch GLIBCXX ABI")
#endif()

#add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=${_ABI_OUTPUT})

# Set MacOS deployment target  // ignored in linux or windows
# wont work on MacOS version lower than 15.0
set(CMAKE_OSX_DEPLOYMENT_TARGET "15.0")

# make colorful output
# kept from last student
string(ASCII 27 Esc)
set(ColorReset "${Esc}[m")
set(Green      "${Esc}[32m")
set(Blue       "${Esc}[34m")

message(STATUS "")
message(STATUS "${Green}==== ${PROJECT_NAME} configure begin ====${ColorReset}")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
  message(STATUS "Default Build Type = Release")
  message(STATUS "")
else()
  message(STATUS "Build Type = ${CMAKE_BUILD_TYPE}")
  message(STATUS "")
endif()

# load all modules
# LIST(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules/)
# no modules as of now

# load all macros
file(GLOB _macro_files "${CMAKE_CURRENT_SOURCE_DIR}/cmake/macros/*.cmake")
foreach(_file ${_macro_files})
  message(STATUS "Include ${_file}")
  include(${_file})
endforeach()

# set output directories
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

message(STATUS "${Blue}${PROJECT_NAME}-1. Set up project compiler flags ...${ColorReset}")
message(STATUS "Include ${CMAKE_CURRENT_SOURCE_DIR}/cmake/compilers.cmake")
include(${PROJECT_SOURCE_DIR}/cmake/compilers.cmake)

message(STATUS "${Blue}${PROJECT_NAME}-2. Setting up project parameters ...${ColorReset}")
include(${PROJECT_SOURCE_DIR}/cmake/parameters.cmake)

message(STATUS "${Blue}${PROJECT_NAME}-3. Setting up system libraries ...${ColorReset}")

option(USE_TORCH "Build with PyTorch (LibTorch) support" ON)

if (USE_TORCH)
  # Find Python interpreter
  find_package(Python3 QUIET COMPONENTS Interpreter)
  message(STATUS "Python3_EXECUTABLE=${Python3_EXECUTABLE}")


  # ask torch where its CMake prefix is
  if (Python3_Interpreter_FOUND)
    execute_process(
      COMMAND "${Python3_EXECUTABLE}" -c
              "import torch; print(torch.utils.cmake_prefix_path)"
      RESULT_VARIABLE _TORCH_PREFIX_RC
      OUTPUT_VARIABLE _TORCH_PREFIX
      ERROR_VARIABLE  _TORCH_PREFIX_ERR
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    # If that worked and returned a path, add it so find_package(Torch) can succeed
    if (_TORCH_PREFIX_RC EQUAL 0 AND NOT _TORCH_PREFIX STREQUAL "")
      list(PREPEND CMAKE_PREFIX_PATH "${_TORCH_PREFIX}")
      message(STATUS "Torch CMake prefix from Python: ${_TORCH_PREFIX}")
    else()
      message(STATUS "Python found but could not query torch CMake prefix (will still try find_package(Torch)).")
    endif()
  else()
    message(STATUS "Python3 not found; cannot query torch prefix (will still try find_package(Torch)).")
  endif()

  # CMake package discovery
  find_package(Torch QUIET)

  if (Torch_FOUND)
    message(STATUS "Torch found by CMake.")
    set(TORCH_CUDA_ARCH_LIST "8.9" CACHE STRING "Torch CUDA arch list")
  else()
    message(STATUS "Torch not found -> building without Torch")
    set(USE_TORCH OFF)
  endif()
endif()

if (USE_CUDA)
  enable_language(CUDA OPTIONAL)

  if (CMAKE_CUDA_COMPILER)
    find_package(CUDAToolkit QUIET)
    if (CUDAToolkit_FOUND)
      set(CMAKE_CUDA_ARCHITECTURES "86;89")
      set(CMAKE_CUDA_STANDARD 17)
    else()
      set(USE_CUDA OFF) # no toolkit CPU-only
    endif()
  else()
    set(USE_CUDA OFF)   # no nvcc  CPU-only
  endif()
endif()

message(STATUS "${Blue}${PROJECT_NAME}-4. Setting up project libraries ...${ColorReset}")
configure_file(${PROJECT_SOURCE_DIR}/cmake_configure.h.in cmake_configure.h @ONLY)
add_subdirectory(src)

if (BUILD_TESTS)
  message(STATUS "${Blue}${PROJECT_NAME}-5. Setting up unit tests ...${ColorReset}")
  add_subdirectory(tests)
endif()
