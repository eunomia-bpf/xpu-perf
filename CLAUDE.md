# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SystemScope is a unified real-time profiling, analysis, and optimization system for modern heterogeneous computing environments. It uses eBPF for zero-instrumentation system profiling and provides real-time visualization through a web frontend.

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

## Frontend Development

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
   - Converts raw profiling data to flamegraph format
   - Handles stack trace symbolization via blazesym

4. **Frontend** (`frontend/`)
   - React + TypeScript + Three.js
   - 3D visualization of flamegraphs
   - Real-time WebSocket connection to backend

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