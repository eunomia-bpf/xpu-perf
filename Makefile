.PHONY: all run clean install deps flamegraph

PID ?= 
DURATION ?= 30
FREQ ?= 49
MIN_BLOCK ?= 1000

all: install run

run:
	@if [ -z "$(PID)" ]; then \
		echo "Error: Please specify PID=<process_id>"; \
		echo "Usage: make run PID=1234 [DURATION=30] [FREQ=49] [MIN_BLOCK=1000]"; \
		exit 1; \
	fi
	@echo "Starting wall-clock profiling for PID $(PID)..."
	sudo python3 cpu-tools/wallclock_profiler.py $(PID) -d $(DURATION) -f $(FREQ) -m $(MIN_BLOCK)

install: deps
	@echo "Checking for required tools..."
	@if [ ! -f cpu-tools/oncputime ] || [ ! -f cpu-tools/offcputime ]; then \
		echo "Building profiling tools..."; \
		$(MAKE) -C cpu-tools; \
	fi
	@echo "Installation complete."

deps:
	@echo "Installing system dependencies..."
	sudo apt update
	sudo apt-get install -y --no-install-recommends \
		libelf1 libelf-dev zlib1g-dev \
		make clang llvm python3 python3-pip git perl

flamegraph:
	@if [ ! -d cpu-tools/FlameGraph ]; then \
		echo "Cloning FlameGraph tools..."; \
		git clone https://github.com/brendangregg/FlameGraph.git cpu-tools/FlameGraph --depth=1; \
	fi

clean:
	rm -rf build
	rm -f combined_profile_*.folded combined_profile_*.svg
	rm -rf multithread_combined_profile_*