# NVCC Compiler settings
# -rdc=true: Enables calling device functions across different files
# -lcudadevrt: Links the CUDA device runtime library required for RDC
NVCC = nvcc
NVCC_FLAGS = -std=c++17 -I./include -arch=sm_89 -rdc=true -lcudadevrt
# sm_100 for rtx 5090

# Output directory
BUILD_DIR = build

# --- Targets ---

all: dir thread_tests

# Create build directory if it doesn't exist
dir:
	if not exist $(BUILD_DIR) mkdir $(BUILD_DIR)

# --- Compilation Targets ---

# IMPORTANT: Added 'dir' as a dependency here so it runs before compilation
thread_tests: dir $(BUILD_DIR)\test_thread.exe $(BUILD_DIR)\fuzzy_thread.exe

$(BUILD_DIR)\test_thread.exe: tests\allocator_test.cu src\threadlocal.cu
	$(NVCC) $(NVCC_FLAGS) tests\allocator_test.cu src\threadlocal.cu -o $@

$(BUILD_DIR)\fuzzy_thread.exe: tests\fuzzy-test-gpu.cu src\threadlocal.cu
	$(NVCC) $(NVCC_FLAGS) tests\fuzzy-test-gpu.cu src\threadlocal.cu -o $@

# --- Run Target ---

# This now depends on 'thread_tests', which guarantees 'dir' is created
run: thread_tests
	@echo.
	@echo ---------------------------
	@echo Running Unit Tests...
	@echo ---------------------------
	$(BUILD_DIR)\test_thread.exe
	@echo.
	@echo ---------------------------
	@echo Running Fuzzy Tests...
	@echo ---------------------------
	$(BUILD_DIR)\fuzzy_thread.exe -s 1234
	@echo.
	@echo [SUCCESS] All tests finished.

# --- Clean Target ---

clean:
	if exist $(BUILD_DIR) rmdir /S /Q $(BUILD_DIR)