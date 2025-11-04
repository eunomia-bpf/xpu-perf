.PHONY: all build clean install deps test help

all: build

build:
	@echo "Building xpu-perf profiler..."
	$(MAKE) -C profiler
	@echo "Build complete: profiler/xpu-perf"

install: deps build
	@echo "Installing xpu-perf to /usr/local/bin..."
	sudo $(MAKE) -C profiler install
	@echo "Installation complete."

deps:
	@echo "Installing system dependencies..."
	sudo apt update
	sudo apt-get install -y --no-install-recommends \
		libelf1 libelf-dev zlib1g-dev \
		make clang llvm git perl golang-go

test: build
	@echo "Running profiler tests..."
	$(MAKE) -C profiler test

clean:
	@echo "Cleaning build artifacts..."
	$(MAKE) -C profiler clean
	$(MAKE) -C cupti_trace clean
	@echo "Clean complete."

help:
	@echo "XPU Performance Profiler - Main Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  make           - Build the profiler"
	@echo "  make build     - Build the profiler"
	@echo "  make install   - Install dependencies and profiler to /usr/local/bin"
	@echo "  make deps      - Install system dependencies"
	@echo "  make test      - Run profiler tests"
	@echo "  make clean     - Clean all build artifacts"
	@echo "  make help      - Show this help message"
	@echo ""
	@echo "Usage example:"
	@echo "  make && sudo ./profiler/xpu-perf -o trace.folded ./my_cuda_app"