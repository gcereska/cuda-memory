FLAGS=(
  "USE_THREAD_LOCAL_BEST_FIT"
  "USE_WARP_LOCAL_BEST_FIT"
  "USE_THREAD_LOCAL"
  "USE_WARP_LOCAL"
  "USE_FREELIST_ALLOCATOR"
  "USE_BST_ALLOCATOR"
  "NONE" # native cuda malloc
)

for FLAG in "${FLAGS[@]}"; do
  echo "|||||=============================================|||||"
  echo " Re-Building Allocator: $FLAG"
  echo "|||||=============================================|||||"

  # remove build
  # rm -rf build

  # 2. Configure and Build
  if [ "$FLAG" == "NONE" ]; then
    BUILD_DIR="build_native"
    cmake --build ${BUILD_DIR} -j
  else
    BUILD_DIR="build_${FLAG}"
    cmake --build ${BUILD_DIR} -j
    # Save the executable with the flag name
  fi
  
  echo -e "Finished re-building bench_${FLAG}\n"
done

echo "All re-builds complete"