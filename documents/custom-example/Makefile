.PHONY: all clean test_program custom-example generate run help

# Default target
all: custom-example test_program

# Generate vmlinux.h if it doesn't exist
vmlinux.h:
	@echo "Generating vmlinux.h..."
	@bpftool btf dump file /sys/kernel/btf/vmlinux format c > vmlinux.h

# Generate Go bindings from eBPF C code
generate: vmlinux.h
	@echo "Generating eBPF Go bindings..."
	@go generate

# Build the custom-example profiler
custom-example: generate
	@echo "Building custom-example..."
	@mkdir -p .build
	@mv -f example_tailcall.c .build/ 2>/dev/null || true
	@go build -o custom-example
	@mv .build/example_tailcall.c . 2>/dev/null || true
	@rmdir .build 2>/dev/null || true
	@echo "✓ custom-example built successfully"

# Build the test program
test_program: test/test_program.c
	@echo "Building test_program..."
	@gcc -o test/test_program test/test_program.c
	@echo "✓ test_program built successfully"

# Run the profiler (requires root)
run: all
	@echo "Starting profiler..."
	@echo "Run './test/test_program' in another terminal to generate traces"
	@sudo ./custom-example ./test/test_program:report_trace

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	@rm -f custom-example
	@rm -f test/test_program
	@rm -f exampletailcall_x86_bpfel.go exampletailcall_x86_bpfel.o
	@rm -f vmlinux.h
	@echo "✓ Cleaned"

# Help target
help:
	@echo "Custom Trace Example Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all            - Build everything (default)"
	@echo "  generate       - Generate eBPF Go bindings"
	@echo "  custom-example - Build the profiler"
	@echo "  test_program   - Build the test program"
	@echo "  run            - Build and run the profiler (requires root)"
	@echo "  clean          - Remove all generated files"
	@echo "  help           - Show this help message"
	@echo ""
	@echo "Usage:"
	@echo "  1. Build everything:  make"
	@echo "  2. Run profiler:      sudo make run"
	@echo "  3. In another terminal: ./test/test_program"
