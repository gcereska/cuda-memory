# ============================
# A collection of unit tests
# ============================

enable_testing()

# C++ tests
#memorypool_setup_test(fuzzy-test)
#memorypool_setup_test(linalg-test)
#memorypool_setup_test(volume-test)

# cuda tests
cuda_memory_test(allocator_test)
cuda_memory_test(fuzzy-test-gpu)
#setup_cuda_test(test-cuda)
#setup_cuda_test(linalg-gpu)
#setup_cuda_test(fuzzy-test-gpu)
#setup_cuda_test(performance-test-gpu)

#setup_python_test_script(performance-test-py)
#setup_python_test_script(blocksize-test-py)

# copy data file
file(GLOB txtfiles *.txt)
foreach(txtfile ${txtfiles})
  get_filename_component(name ${txtfile} NAME)
  message(STATUS "Copying ${txtfile} to ${name}")
  configure_file(${txtfile} ${CMAKE_CURRENT_BINARY_DIR}/${name} @ONLY)
endforeach()
