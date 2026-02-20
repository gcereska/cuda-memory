#!/bin/bash

# 1. Add the names of the test executables you want to run
TARGETS=(
  "blueprint"
  #"benchmark_all"
)

# 2. The list of build directories we generated in the compile script
BUILD_DIRS=(
  "build_USE_THREAD_LOCAL_BEST_FIT"
  "build_USE_WARP_LOCAL_BEST_FIT"
  "build_USE_THREAD_LOCAL"
  "build_USE_WARP_LOCAL"
  "build_USE_FREELIST_ALLOCATOR"
  "build_USE_BST_ALLOCATOR"
  "build_native"
)

echo "Starting to run all tests"
echo "[][][][][][][][]][[][][][][][][][][][[][][][][][][][][][][][][][][][][][][]]]"
for TARGET in "${TARGETS[@]}"; do
  echo "#########################################################"
  echo " RUNNING TEST SUITE: $TARGET"
  echo "#########################################################"

  for DIR in "${BUILD_DIRS[@]}"; do
    # Construct the path to executable
    EXE_PATH="./${DIR}/tests/${TARGET}"

    # Check if the file actually exists
    if [ -x "$EXE_PATH" ]; then
      echo "---------------------------------------------------------"
      echo " Executing: $EXE_PATH"
      echo "---------------------------------------------------------"
      
      # Run it
      $EXE_PATH
      
      echo "" # Print a blank line for readability
    else
      echo " [SKIPPED] Could not find executable: $EXE_PATH"
    fi
  done
  
  echo -e "\n"
done

echo "[][][][][][][][]][[][][][][][][][][][[][][][][][][][][][][][][][][][][][][]]]"
echo "All tests complete"