.PHONY: all clean mock-app cpu-minimal help

# Build all test applications
all: mock-app cpu-minimal

# Build CUDA mock applications
mock-app:
	@echo "Building CUDA mock applications..."
	$(MAKE) -C mock-app
	@echo "Mock applications build complete"

# Build CPU minimal test
cpu-minimal:
	@echo "Building CPU minimal test..."
	$(MAKE) -C cpu-minimal
	@echo "CPU minimal test build complete"

# Clean all test builds
clean:
	@echo "Cleaning test builds..."
	$(MAKE) -C mock-app clean
	$(MAKE) -C cpu-minimal clean
	@echo "Test clean complete"

# Show help
help:
	@echo "XPU Performance Profiler - Test Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  make           - Build all test applications"
	@echo "  make mock-app  - Build CUDA mock applications (vectorAdd, llm-inference, cuda_graph_simple)"
	@echo "  make cpu-minimal - Build CPU minimal test (test_malloc)"
	@echo "  make clean     - Clean all test builds"
	@echo "  make help      - Show this help message"
	@echo ""
	@echo "Test applications:"
	@echo "  - mock-app/vectorAdd         : Simple CUDA vector addition"
	@echo "  - mock-app/llm-inference     : Mock LLM inference with multiple kernels"
	@echo "  - mock-app/cuda_graph_simple : CUDA graph example"
	@echo "  - cpu-minimal/test_malloc    : Simple CPU malloc test"
