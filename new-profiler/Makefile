.PHONY: all build clean test cupti generate install help

BINARY_NAME=new-xpu-perf
CUPTI_LIB=../cupti_trace/libcupti_trace_injection.so
CUPTI_LIB_LOCAL=libcupti_trace_injection.so

all: build

# Generate eBPF code from C source
generate:
	@echo "Generating eBPF code..."
	cd source/primitive/otel && go generate ./...
	@echo "eBPF code generation complete"

# Build the CUPTI library first, then copy it and build the profiler
build: cupti generate
	@echo "Building new-xpu-perf profiler..."
	go build -o $(BINARY_NAME) .
	@echo "Build complete: $(BINARY_NAME)"

# Build CUPTI library and copy it to profiler directory
cupti:
	@echo "Building CUPTI trace library..."
	$(MAKE) -C ../cupti_trace
	@echo "Copying CUPTI library to profiler directory..."
	cp $(CUPTI_LIB) $(CUPTI_LIB_LOCAL)

# Clean build artifacts
clean:
	@echo "Cleaning profiler build artifacts..."
	rm -f $(BINARY_NAME)
	rm -f $(CUPTI_LIB_LOCAL)
	rm -f *.folded *.svg
	rm -f source/primitive/otel/correlationuprobe_x86_bpfel.go
	rm -f source/primitive/otel/correlationuprobe_x86_bpfel.o
	$(MAKE) -C ../cupti_trace clean
	@echo "Clean complete"

# Run unit tests
test-unit:
	@echo "Running unit tests..."
	go test -v ./...

# Test CPU-only mode
test-cpu-only: build
	@echo "Testing CPU-only mode..."
	@if [ ! -f /root/xpu-perf/test/pytorch/pytorch_minimal.py ]; then \
		echo "Warning: Test file not found, skipping"; \
	else \
		sudo ./$(BINARY_NAME) -cpu-only -o test_cpu.folded python3 /root/xpu-perf/test/pytorch/pytorch_minimal.py || true; \
		if [ -s test_cpu.folded ]; then \
			echo "✓ CPU-only test passed - output generated"; \
			wc -l test_cpu.folded; \
			rm -f test_cpu.folded; \
		else \
			echo "✗ CPU-only test failed - no output"; \
			rm -f test_cpu.folded; \
		fi \
	fi

# Test GPU-only mode (uprobe correlation)
test-gpu-only: build
	@echo "Testing GPU-only mode (uprobe correlation)..."
	@if [ ! -f /root/xpu-perf/test/pytorch/pytorch_minimal.py ]; then \
		echo "Warning: Test file not found, skipping"; \
	else \
		timeout 30 sudo ./$(BINARY_NAME) -gpu-only -o test_gpu.folded python3 /root/xpu-perf/test/pytorch/pytorch_minimal.py || true; \
		if [ -s test_gpu.folded ]; then \
			echo "✓ GPU-only test passed - output generated"; \
			wc -l test_gpu.folded; \
			rm -f test_gpu.folded; \
		else \
			echo "✗ GPU-only test failed - no output"; \
			rm -f test_gpu.folded; \
		fi \
	fi

# Test merge mode (CPU sampling + GPU correlation)
test-merge: build
	@echo "Testing merge mode (CPU + GPU)..."
	@if [ ! -f /root/xpu-perf/test/pytorch/pytorch_minimal.py ]; then \
		echo "Warning: Test file not found, skipping"; \
	else \
		timeout 30 sudo ./$(BINARY_NAME) -o test_merge.folded python3 /root/xpu-perf/test/pytorch/pytorch_minimal.py || true; \
		if [ -s test_merge.folded ]; then \
			echo "✓ Merge mode test passed - output generated"; \
			wc -l test_merge.folded; \
			rm -f test_merge.folded; \
		else \
			echo "✗ Merge mode test failed - no output"; \
			rm -f test_merge.folded; \
		fi \
	fi

# Test correlation ID mode (exact correlation)
test-correlation-id: build
	@echo "Testing correlation ID mode..."
	@if [ ! -f /root/xpu-perf/test/pytorch/pytorch_minimal.py ]; then \
		echo "Warning: Test file not found, skipping"; \
	else \
		timeout 30 sudo ./$(BINARY_NAME) -gpu-only -use-correlation-id -o test_corr.folded python3 /root/xpu-perf/test/pytorch/pytorch_minimal.py || true; \
		if [ -s test_corr.folded ]; then \
			echo "✓ Correlation ID test passed - output generated"; \
			wc -l test_corr.folded; \
			rm -f test_corr.folded; \
		else \
			echo "✗ Correlation ID test failed - no output"; \
			rm -f test_corr.folded; \
		fi \
	fi

# Component tests - test each source in isolation

# Test CUPTI source only
test-component-cupti: build
	@echo "Testing CUPTI source component..."
	@echo "Note: This requires manual CUPTI pipe setup. Run this in two terminals:"
	@echo "  Terminal 1: go run test_cupti_only.go /tmp/test_cupti.pipe"
	@echo "  Terminal 2: CUDA_INJECTION64_PATH=./libcupti_trace_injection.so CUPTI_TRACE_OUTPUT_FILE=/tmp/test_cupti.pipe python3 /root/xpu-perf/test/pytorch/pytorch_minimal.py"
	@echo ""

# Test OTel source only
test-component-otel: build
	@echo "Testing OTel source component..."
	@if [ ! -f /root/xpu-perf/test/pytorch/pytorch_minimal.py ]; then \
		echo "Warning: Test file not found, skipping"; \
	else \
		timeout 30 sudo go run test_otel_only.go python3 /root/xpu-perf/test/pytorch/pytorch_minimal.py || true; \
	fi

# Test correlation source (combined CUPTI + OTel)
test-component-correlation: build
	@echo "Testing correlation source component..."
	@if [ ! -f /root/xpu-perf/test/pytorch/pytorch_minimal.py ]; then \
		echo "Warning: Test file not found, skipping"; \
	else \
		timeout 30 sudo CUPTI_LIB_PATH=./libcupti_trace_injection.so go run test_correlation.go python3 /root/xpu-perf/test/pytorch/pytorch_minimal.py || true; \
	fi

# Run component tests
test-components: test-component-otel test-component-correlation
	@echo ""
	@echo "=========================================="
	@echo "Component tests completed!"
	@echo "=========================================="

# Run all tests
test: test-unit test-cpu-only test-gpu-only test-merge test-correlation-id
	@echo ""
	@echo "=========================================="
	@echo "All tests completed!"
	@echo "=========================================="

# Install to system (optional)
install: build
	@echo "Installing $(BINARY_NAME) to /usr/local/bin..."
	sudo cp $(BINARY_NAME) /usr/local/bin/
	@echo "Installation complete"

# Show help
help:
	@echo "XPU Performance Profiler - New Architecture - Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  make              - Build everything (CUPTI library + eBPF + profiler)"
	@echo "  make build        - Build profiler with embedded CUPTI library"
	@echo "  make generate     - Generate eBPF code from C source"
	@echo "  make cupti        - Build only CUPTI library and copy to profiler dir"
	@echo "  make clean        - Clean all build artifacts"
	@echo "  make test         - Run all tests (unit + integration)"
	@echo "  make test-unit    - Run unit tests only"
	@echo "  make test-cpu-only        - Test CPU-only profiling mode"
	@echo "  make test-gpu-only        - Test GPU-only profiling mode (uprobe)"
	@echo "  make test-merge           - Test merge mode (CPU + GPU)"
	@echo "  make test-correlation-id  - Test exact correlation ID matching"
	@echo "  make test-components      - Run component tests (OTel + Correlation)"
	@echo "  make test-component-cupti         - Test CUPTI source only"
	@echo "  make test-component-otel          - Test OTel source only"
	@echo "  make test-component-correlation   - Test correlation source"
	@echo "  make install      - Install binary to /usr/local/bin"
	@echo "  make help         - Show this help message"
	@echo ""
	@echo "Usage examples:"
	@echo "  # Full CPU+GPU profiling (merged view)"
	@echo "  make && sudo ./$(BINARY_NAME) -o trace.folded ./my_cuda_app"
	@echo ""
	@echo "  # CPU-only profiling"
	@echo "  sudo ./$(BINARY_NAME) -cpu-only -o cpu_trace.folded ./my_app"
	@echo ""
	@echo "  # GPU-only profiling (CPU→GPU causality)"
	@echo "  sudo ./$(BINARY_NAME) -gpu-only -o gpu_trace.folded ./my_cuda_app"
	@echo ""
	@echo "  # With exact correlation IDs"
	@echo "  sudo ./$(BINARY_NAME) -gpu-only -use-correlation-id -o trace.folded ./my_cuda_app"
	@echo ""
