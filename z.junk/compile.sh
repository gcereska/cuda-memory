#!/bin/bash

# Check if a test file was provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <test-file.cu>"
    echo "Example: $0 test-cuda-julian-impl.cu"
    exit 1
fi

TEST_FILE=$1

# Check if the test file exists
if [ ! -f "$TEST_FILE" ]; then
    echo "Error: File '$TEST_FILE' not found!"
    exit 1
fi

# Extract the base name without extension for the output executable
OUTPUT_NAME="${TEST_FILE%.cu}"

echo "Compiling $TEST_FILE..."
echo "Output executable: $OUTPUT_NAME"

# Compile with all required source files
# Note: Only compiling poolAllocBST.cu (not poolAlloc.cu) to avoid duplicate definitions
nvcc -rdc=true "$TEST_FILE" poolAllocBST.cu RBTree.cu -o "$OUTPUT_NAME"

# Check if compilation was successful
if [ $? -eq 0 ]; then
    echo "Compilation successful!"
    echo "Run with: ./$OUTPUT_NAME"
else
    echo "Compilation failed!"
    exit 1
fi
