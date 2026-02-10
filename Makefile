NVCC       := nvcc
NVCC_FLAGS := -std=c++17 -I./include -arch=sm_89 -rdc=true -lcudadevrt
# sm_100 for rtx 5090

CONFIG    := Release
OUT_DIR   := build/tests/$(CONFIG)

.PHONY: all dir thread_tests run clean

all: dir thread_tests

dir:
	mkdir -p $(OUT_DIR)

thread_tests: dir $(OUT_DIR)/allocator_test.release $(OUT_DIR)/fuzzy-test-gpu.release

$(OUT_DIR)/allocator_test.release: tests/allocator_test.cu src/threadlocal.cu
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

$(OUT_DIR)/fuzzy-test-gpu.release: tests/fuzzy-test-gpu.cu src/threadlocal.cu
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

run: thread_tests
	@echo
	@echo "---------------------------"
	@echo "Running Unit Tests..."
	@echo "---------------------------"
	./$(OUT_DIR)/allocator_test.release
	@echo
	@echo "---------------------------"
	@echo "Running Fuzzy Tests..."
	@echo "---------------------------"
	./$(OUT_DIR)/fuzzy-test-gpu.release -s 1234
	@echo
	@echo "[SUCCESS] All tests finished."

clean:
	rm -rf build
