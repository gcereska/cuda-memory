# A small macro used for setting up the build of a test.
#
# Usage: setup_cuda_test(name)

string(TOLOWER ${CMAKE_BUILD_TYPE} buildl)
string(TOUPPER ${CMAKE_BUILD_TYPE} buildu)
string(TOUPPER ${PROJECT_NAME} projectu)

#macro(memory_test namel)

macro(cuda_memory_test namel)
  add_executable(${namel} ${namel}.cu ${ARGN})

  set_target_properties(${namel}
    PROPERTIES CUDA_SEPARABLE_COMPILATION ON
  )

  target_compile_options(${namel} PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=177,550,127>
  )

  target_include_directories(${namel}
    PRIVATE ${CMAKE_BINARY_DIR}
            ${${projectu}_INCLUDE_DIR}
            ${TORCH_INCLUDE_DIR}
            ${TORCH_API_INCLUDE_DIR}
  )

  target_link_libraries(${namel}
    PRIVATE ${PROJECT_NAME}::${PROJECT_NAME}
            ${TORCH_LIBRARY}
            ${TORCH_CPU_LIBRARY}
            ${C10_LIBRARY}
            $<IF:$<BOOL:${CUDA}>,${PROJECT_NAME}::${PROJECT_NAME}_cu,>
            $<IF:$<BOOL:${CUDA}>,${TORCH_CUDA_LIBRARY},>
            $<IF:$<BOOL:${CUDA}>,${C10_CUDA_LIBRARY},>
  )

  add_test(NAME ${namel} COMMAND ${namel})
endmacro()

