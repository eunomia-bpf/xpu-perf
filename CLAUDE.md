# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SystemScope is a high-performance eBPF profiler for Linux systems. It provides zero-instrumentation profiling with minimal overhead, capturing both on-CPU and off-CPU time for comprehensive performance analysis.

## Build System

```bash
# Quick build
make build

# Clean build
make clean

# Run tests
make test

# Install dependencies (Ubuntu)
make install
```

### Build Details
- CMake 3.16+ with Unix Makefiles generator
- C++20 standard required
- AddressSanitizer enabled by default in debug builds
- Binary name: `profiler` (in `build/` directory)

## Source Code Structure

### Main Components (`src/`)

#### 1. Entry Point
- `main.cpp`: Command-line interface and analyzer orchestration
- `args_parser.cpp/hpp`: Argument parsing using argparse library

#### 2. Analyzers (`src/analyzers/`)
Three main analyzer types, each with specific use cases:

- **ProfileAnalyzer** (`profile_analyzer.hpp`)
  - On-CPU profiling using perf events
  - Samples at specified frequency (default 49 Hz)
  - Best for CPU-bound performance analysis

- **OffCPUTimeAnalyzer** (`offcputime_analyzer.hpp`)
  - Tracks time spent off-CPU (blocking, I/O, sleep)
  - Configurable minimum block time threshold
  - Best for I/O-bound and lock contention analysis

- **WallClockAnalyzer** (`wallclock_analyzer.cpp/hpp`)
  - Combines on-CPU and off-CPU data
  - Provides complete wall-clock time view
  - Best for comprehensive performance analysis

Supporting classes:
- `base_analyzer.hpp`: Abstract base class for all analyzers
- `analyzer_config.hpp`: Configuration structures
- `flamegraph_view.cpp/hpp`: Data structure for flamegraph representation
- `symbol_resolver.cpp/hpp`: Stack trace symbolization using blazesym

#### 3. eBPF Collectors (`src/collectors/`)
Kernel-space data collection:

- **On-CPU Collector** (`oncpu/`)
  - `profile.bpf.c`: eBPF program for CPU sampling
  - `profile.cpp/hpp`: User-space handler
  - Uses perf events for sampling

- **Off-CPU Collector** (`offcpu/`)
  - `offcputime.bpf.c`: eBPF program for off-CPU tracking
  - `offcputime.cpp/hpp`: User-space handler
  - Hooks into scheduler events

Common:
- `collector_interface.hpp`: Abstract interface for collectors
- `sampling_data.hpp`: Data structures for samples
- `config.hpp`: Collector configuration
- `bpf_event.h`: Shared kernel/user-space structures

#### 4. Data Processing
- `flamegraph_generator.cpp/hpp`: Converts raw samples to flamegraph format
- Outputs SVG files using folded stack format
- Handles per-thread and aggregate views

### Command-Line Interface

```
Usage: profiler ANALYZER [options]

Analyzers:
  profile      - On-CPU profiling
  offcputime   - Off-CPU time analysis  
  wallclock    - Combined analysis

Key Options:
  -d, --duration SECONDS     # Profiling duration
  -p, --pid PID[,PID...]    # Target processes
  -t, --tid TID[,TID...]    # Target threads
  -f, --frequency HZ        # Sampling rate (default: 49)
  -m, --min-block µs        # Min off-CPU time (default: 1000)
```

### Output Format

SystemScope generates organized output directories:
```
{analyzer}_profile_[pid{PID}]_{timestamp}/
├── flame_cpu_{PID}.svg       # On-CPU flamegraph
├── flame_offcpu_{PID}.svg    # Off-CPU flamegraph
├── flame_wallclock_{PID}.svg # Combined view
└── stacks_{PID}.txt          # Raw stack data
```

## Key Implementation Details

### eBPF Programs
- Located in `src/collectors/{oncpu,offcpu}/*.bpf.c`
- Compiled at build time using clang
- Loaded into kernel via libbpf
- Use BPF ring buffers for data transfer

### Stack Walking
- Uses frame pointers when available
- Falls back to DWARF unwinding via blazesym
- Maximum stack depth configurable (default: 127)

### Symbol Resolution
- Blazesym library for fast symbolization
- Caches symbol maps for efficiency
- Handles kernel and user-space symbols

### Performance Considerations
- Default 49 Hz sampling (avoids 50 Hz timer aliasing)
- Ring buffer size tuned for low overhead
- Minimal kernel-space processing

## Testing

Tests are in `tests/` directory:
- `test_profile_collector.cpp`: On-CPU collector tests
- `test_offcputime_collector.cpp`: Off-CPU collector tests
- `test_flamegraph_view.cpp`: Data structure tests

Run with: `make test` or `ctest --test-dir build`

## Dependencies

### Core Dependencies
- **libbpf**: eBPF library (included)
- **blazesym**: Symbolization (built from source)
- **argparse**: Command-line parsing (included)
- **spdlog**: Logging (included)

### Test Dependencies
- **Catch2**: Testing framework (included)

## Development Notes

### Adding a New Analyzer
1. Create class inheriting from `BaseAnalyzer`
2. Implement `start()` method
3. Add configuration in `analyzer_config.hpp`
4. Register in `main.cpp` and `args_parser.cpp`

### Adding a New Collector
1. Write eBPF program in `src/collectors/`
2. Create user-space handler class
3. Implement `ICollector` interface
4. Add to CMakeLists.txt compilation

### Debugging Tips
- Use `-v` flag for verbose output
- Check `dmesg` for eBPF verifier errors
- Use `bpftool prog list` to see loaded programs
- AddressSanitizer enabled by default in debug builds

## Common Issues and Solutions

### Build Issues
- Ensure clang is installed for BPF compilation
- Check kernel headers are available
- Verify CMake uses Unix Makefiles generator

### Runtime Issues
- Requires root or CAP_SYS_ADMIN capability
- Check kernel has eBPF support (4.9+)
- Verify frame pointers enabled for stack walking

### Performance Tuning
- Adjust sampling frequency with `-f`
- Increase min-block time for less overhead
- Use PID filtering to reduce data volume

## Project Structure

```
systemscope/
├── src/
│   ├── analyzers/      # Analyzer implementations
│   ├── collectors/     # eBPF collectors
│   ├── main.cpp       # Entry point
│   └── args_parser.cpp # CLI argument handling
├── tests/             # Unit tests
├── tools/             # Additional profiling tools
├── vmlinux/          # Pre-generated vmlinux headers
├── libbpf/           # libbpf submodule
├── blazesym/         # Blazesym for symbolization
└── CMakeLists.txt    # Build configuration
```