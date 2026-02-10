# Install ${PROJECT_NAME} library
#
#   library: lib${PROJECT_NAME}_debug.o
#   library: lib${PROJECT_NAME}_release.o
#
# Define the following symbols
#
#   MEMORYPOOL_INCLUDE_DIR
#   ${PROJECT_NAME}::${PROJECT_NAME}
#   ${PROJECT_NAME}::${PROJECT_NAME}_cu
#
# Normal usage would be:
#
#   include_directories(${MEMORYPOOL_INCLUDE_DIR})
#   target_link_libraries(${PROJECT_NAME}::${PROJECT_NAME} ${PROJECT_NAME}::${PROJECT_NAME}_cu)


include_directories(
  "${CMAKE_CURRENT_SOURCE_DIR}/.."
  ${CMAKE_BINARY_DIR}
  SYSTEM ${TORCH_API_INCLUDE_DIR}
  SYSTEM ${TORCH_INCLUDE_DIR}
  )

string(TOLOWER ${CMAKE_BUILD_TYPE} buildl)
string(TOUPPER ${CMAKE_BUILD_TYPE} buildu)

set(namel ${PROJECT_NAME})
string(TOUPPER ${namel} nameu)

file(GLOB src_files 
  cpu/pmalloc_*.cpp
  )

add_library(${namel}_${buildl}
    STATIC
    ${src_files}
    )

set_target_properties(${namel}_${buildl}
    PROPERTIES
    COMPILE_FLAGS "${CMAKE_CXX_FLAGS_${buildu}}"
    )

target_link_libraries(${namel}_${buildl}
    PUBLIC
    )

add_library(${namel}::${namel} ALIAS ${namel}_${buildl})

# CUDA source codes
if (CUDAToolkit_FOUND)
  file(GLOB cu_src_files 
    cuda/threadlocal.cu
    )

  add_library(${namel}_cuda_${buildl}
      STATIC
      ${cu_src_files}
      )

  set_target_properties(${namel}_cuda_${buildl}
      PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      )

  target_link_libraries(${namel}_cuda_${buildl}
      PUBLIC
      $<IF:$<BOOL:${CUDAToolkit_FOUND}>,CUDA::cudart,>
      $<IF:$<BOOL:${CUDAToolkit_FOUND}>,CUDA::cudart_static,>
      )

  add_library(${namel}::${namel}_cu ALIAS ${namel}_cuda_${buildl})
endif()

set(${nameu}_INCLUDE_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/.."
  CACHE INTERNAL "${namel} include directory")
