cmake_minimum_required(VERSION 3.25)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

option(BUILD_EXAMPLES "Build examples" OFF)
option(BUILD_TESTS "Build tests" ON)

project(
  cuda-memory
  LANGUAGES CXX C
  DESCRIPTION "Memory Pool Allocator for C/C++ and CUDA")

# Try to enable CUDA
enable_language(CUDA OPTIONAL)

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

# Set MacOS deployment target
set(CMAKE_OSX_DEPLOYMENT_TARGET "15.0")

# make colorful output
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
LIST(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules/)

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
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/compilers.cmake)

message(STATUS "${Blue}${PROJECT_NAME}-2. Setting up project parameters ...${ColorReset}")
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/parameters.cmake)

message(STATUS "${Blue}${PROJECT_NAME}-3. Setting up system libraries ...${ColorReset}")
#find_package(Torch REQUIRED)
if (CMAKE_CUDA_COMPILER)
  find_package(CUDAToolkit REQUIRED)
else()
  set(CUDAToolkit_FOUND OFF)
endif()

if (CUDAToolkit_FOUND)
  set(CMAKE_CUDA_ARCHITECTURES 86 89)
  set(CMAKE_CUDA_STANDARD 17)
endif()

message(STATUS "${Blue}${PROJECT_NAME}-4. Setting up project libraries ...${ColorReset}")
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/configure.h.in configure.h @ONLY)
add_subdirectory(src)

#if (BUILD_TESTS)
message(STATUS "${Blue}${PROJECT_NAME}-5. Setting up unit tests ...${ColorReset}")
add_subdirectory(tests)
#endif()
