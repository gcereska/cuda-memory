# ============================
# A collection of unit tests
# ============================

enable_testing()


add_executable(allocator_bench_all allocator_bench_all.cpp)

target_link_libraries(allocator_bench_all 
  PRIVATE 
  ${PROJECT_NAME}::${PROJECT_NAME}
)


set(CPU_C_TESTS
  test
  testAllocationTooLarge
  testInvalidDeletion
  testLightFragmentation
  testRigorousFragmentation
)

foreach(t ${CPU_C_TESTS})
  add_executable(${t} ${t}.c)
  
  target_link_libraries(${t} 
    PRIVATE 
    ${PROJECT_NAME}::${PROJECT_NAME}
  )

  # Register with CTest so you can run `make test` or `ctest` later
  add_test(NAME ${t} COMMAND ${t})
endforeach()


if (USE_CUDA)
  file(GLOB CUDA_TESTS "*.cu")


  foreach(test ${CUDA_TESTS})
    get_filename_component(t ${test} NAME_WE)
    cuda_memory_test(${t})
  endforeach()
endif()



# copy data file
file(GLOB txtfiles *.txt)
foreach(txtfile ${txtfiles})
  get_filename_component(name ${txtfile} NAME)
  message(STATUS "Copying ${txtfile} to ${name}")
  configure_file(${txtfile} ${CMAKE_CURRENT_BINARY_DIR}/${name} @ONLY)
endforeach()
