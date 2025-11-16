.PHONY: all build clean test test-apps cupti generate

BINARY_NAME=xpu-perf
CUPTI_LIB=../cupti_trace/libcupti_trace_injection.so
CUPTI_LIB_LOCAL=libcupti_trace_injection.so
TEST_DIR=../test
TEST_APP=$(TEST_DIR)/mock-app/vectorAdd

all: build

# Generate eBPF Go bindings
generate:
	@echo "Generating eBPF Go bindings..."
	go generate ./...
	@echo "eBPF code generation complete"

# Build the CUPTI library first, then copy it and build the profiler
build: generate cupti
	@echo "Building xpu-perf profiler..."
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
	rm -f correlationuprobe_*.go correlationuprobe_*.o
	$(MAKE) -C ../cupti_trace clean
	@echo "Clean complete"

# Build test applications
test-apps:
	@echo "Building test applications..."
	$(MAKE) -C $(TEST_DIR)
	@echo "Test applications built successfully"

# Test with a sample CUDA application
test: build test-apps
	@echo "Testing profiler with vector addition..."
	@if [ ! -f $(TEST_APP) ]; then \
		echo "✗ Test application not found: $(TEST_APP)"; \
		exit 1; \
	fi
	sudo -E env "PATH=$$PATH" ./$(BINARY_NAME) --gpu-only -o test.folded $(TEST_APP)
	@if [ -s test.folded ]; then \
		echo "✓ Test passed - output generated"; \
		cat test.folded; \
		rm -f test.folded; \
	else \
		echo "✗ Test failed - no output"; \
		rm -f test.folded; \
		exit 1; \
	fi

# Install to system (optional)
install: build
	@echo "Installing $(BINARY_NAME) to /usr/local/bin..."
	sudo cp $(BINARY_NAME) /usr/local/bin/
	@echo "Installation complete"

# Show help
help:
	@echo "XPU Performance Profiler - Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  make           - Build everything (eBPF generation + CUPTI library + profiler)"
	@echo "  make build     - Build profiler with embedded CUPTI library"
	@echo "  make generate  - Generate eBPF Go bindings from eBPF C code"
	@echo "  make cupti     - Build only CUPTI library and copy to profiler dir"
	@echo "  make test-apps - Build test applications in ../test directory"
	@echo "  make test      - Build test apps and run profiler test with vectorAdd"
	@echo "  make clean     - Clean all build artifacts"
	@echo "  make install   - Install binary to /usr/local/bin"
	@echo "  make help      - Show this help message"
	@echo ""
	@echo "Usage example:"
	@echo "  make && sudo ./xpu-perf -o trace.folded ./my_cuda_app"
