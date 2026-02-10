# A small macro used for setting up the build of a test.
#
# Usage: setup_cuda_test(name)

string(TOLOWER ${CMAKE_BUILD_TYPE} buildl)
string(TOUPPER ${CMAKE_BUILD_TYPE} buildu)
string(TOUPPER ${PROJECT_NAME} projectu)

macro(cuda_memory_test namel)
  add_executable(${namel}.${buildl} ${namel}.cu)

  set_target_properties(
    ${namel}.${buildl}
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    )

  target_include_directories(
    ${namel}.${buildl}
    PRIVATE ${CMAKE_BINARY_DIR}
            ${${projectu}_INCLUDE_DIR}
            ${TORCH_INCLUDE_DIR}
            ${TORCH_API_INCLUDE_DIR})

  target_link_libraries(${namel}.${buildl}
    PRIVATE ${PROJECT_NAME}::${PROJECT_NAME}
            ${TORCH_LIBRARY}
            ${TORCH_CPU_LIBRARY}
            ${C10_LIBRARY}
            $<IF:$<BOOL:${CUDAToolkit_FOUND}>,${PROJECT_NAME}::${PROJECT_NAME}_cu,>
            $<IF:$<BOOL:${CUDAToolkit_FOUND}>,${TORCH_CUDA_LIBRARY},>
            $<IF:$<BOOL:${CUDAToolkit_FOUND}>,${C10_CUDA_LIBRARY},>)

  add_test(NAME ${namel}.${buildl} COMMAND ${namel}.${buildl})
endmacro()
