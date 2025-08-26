# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SystemScope is a high-performance wall-clock eBPF profiler for Linux systems. It provides zero-instrumentation profiling with minimal overhead, outputting standard formats compatible with popular visualization tools. The optional `systemscope-vis` frontend package provides 3D visualization capabilities.

## Build Commands

```bash
# Build the project
make build
# or directly with CMake
cmake -B build
cmake --build build

# Run tests
make test
# or
cmake -B build
cmake --build build --target profiler_tests
ctest --test-dir build

# Clean build artifacts
make clean

# Install dependencies (Ubuntu)
make install
```

## Optional Frontend Development (systemscope-vis)

The frontend is now a separate package called `systemscope-vis` in the `frontend/` directory.

```bash
cd frontend

# Install dependencies
npm install

# Development server
npm run dev

# Build production
npm run build

# Run tests
npm run test
npm run test:run  # Run once without watch

# Type checking
npm run type-check

# Linting and formatting
npm run lint
npm run lint:fix
npm run format
```

## Architecture

### Core Components

1. **eBPF Collectors** (`src/collectors/`)
   - `oncpu/`: On-CPU profiling using performance events
   - `offcpu/`: Off-CPU time tracking
   - BPF programs compile to bytecode loaded into kernel

2. **Profile Server** (`src/server/`)
   - WebSocket server for real-time data streaming
   - Handles frontend connections and profile data delivery

3. **Flamegraph Generator** (`src/flamegraph_generator.cpp`)
   - Converts raw profiling data to standard formats
   - Handles stack trace symbolization via blazesym
   - Supports folded stacks, pprof, and Chrome trace formats

4. **Optional Frontend** (`frontend/` - package `systemscope-vis`)
   - Separate npm package for visualization
   - React + TypeScript + Three.js
   - 3D visualization of flamegraphs
   - Can be used standalone or with SystemScope server

### Key Dependencies

- **libbpf**: eBPF library for kernel instrumentation
- **blazesym**: Symbolization for stack traces
- **CMake**: Build system
- **React Three Fiber**: 3D visualization in frontend

### Testing Strategy

- C++ tests use Google Test framework
- Frontend tests use Vitest
- Tests located in `tests/` for backend, `frontend/src/__tests__/` for frontend

## Development Notes

- The project uses CMake with Unix Makefiles generator
- AddressSanitizer is enabled by default for debug builds
- BPF programs require root privileges to run
- Frontend connects to backend via WebSocket on port 8080 (default)