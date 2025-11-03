# CUPTI Trace Injection

A CUPTI-based tracing injection library to trace CUDA applications without modifying their source code.

## Overview

This library uses NVIDIA's CUPTI (CUDA Profiling Tools Interface) Activity and Callback APIs to inject tracing functionality into any CUDA application. It captures kernel launches, memory operations, driver/runtime API calls, and other CUDA activities.

## Features

- **Zero code modification**: Inject profiling into any CUDA application via environment variables
- **Flexible activity tracking**: Selectively enable different CUDA activities
- **Configurable output**: Write traces to files or stdout
- **Graph node tracking**: Track CUDA graph node creation and execution
- **PC sampling support**: Optional PC sampling for detailed performance analysis

## Supported Activities

### Core Activities (always enabled)
- `CUPTI_ACTIVITY_KIND_RUNTIME`: CUDA runtime API calls
- `CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL`: Kernel executions

### Optional Activities

Enable via environment variables:

**Driver Activities** (`CUPTI_ENABLE_DRIVER=1`):
- `CUPTI_ACTIVITY_KIND_DRIVER`: CUDA driver API calls
- `CUPTI_ACTIVITY_KIND_OVERHEAD`: CUPTI overhead tracking

**Memory Operations** (`CUPTI_ENABLE_MEMORY=1`):
- `CUPTI_ACTIVITY_KIND_MEMCPY`: Memory copy operations
- `CUPTI_ACTIVITY_KIND_MEMCPY2`: Peer-to-peer memory copies
- `CUPTI_ACTIVITY_KIND_MEMSET`: Memory set operations
- `CUPTI_ACTIVITY_KIND_MEMORY2`: Memory allocation/deallocation

**NVTX Annotations** (`CUPTI_ENABLE_NVTX=1`):
- `CUPTI_ACTIVITY_KIND_NAME`: Named entities (streams, kernels, etc.)
- `CUPTI_ACTIVITY_KIND_MARKER`: NVTX markers
- `CUPTI_ACTIVITY_KIND_MARKER_DATA`: NVTX marker data

**CUDA Graph Tracking** (`CUPTI_ENABLE_GRAPH=1`):
- Tracks graph node creation and cloning
- Correlates graph nodes with the API calls that created them
- Shows `graphId` and `graphNodeId` in kernel/memcpy records
- Displays which API created each graph node (e.g., `cudaGraphAddKernelNode`)

**PC Sampling** (`CUPTI_ENABLE_PC_SAMPLING=1`):
- Requires additional configuration for kernel-level sampling

## Usage

### Linux

```bash
# Basic usage (runtime + kernel activities only)
env CUDA_INJECTION64_PATH=/path/to/libcupti_trace_injection.so \
    CUPTI_TRACE_OUTPUT_FILE=/tmp/trace.txt \
    ./your_cuda_app

# Enable all features
env CUDA_INJECTION64_PATH=/path/to/libcupti_trace_injection.so \
    CUPTI_TRACE_OUTPUT_FILE=/tmp/trace.txt \
    CUPTI_ENABLE_DRIVER=1 \
    CUPTI_ENABLE_MEMORY=1 \
    CUPTI_ENABLE_NVTX=1 \
    ./your_cuda_app

# PC Sampling (for detailed kernel profiling)
env CUPTI_ENABLE_PC_SAMPLING=1 \
    CUDA_INJECTION64_PATH=/path/to/libcupti_trace_injection.so \
    CUPTI_TRACE_OUTPUT_FILE=/tmp/trace.txt \
    ./your_cuda_app
```

### Windows

```cmd
set CUDA_INJECTION64_PATH=C:\path\to\cupti_trace_injection.dll
set CUPTI_TRACE_OUTPUT_FILE=C:\trace.txt
your_cuda_app.exe
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CUDA_INJECTION64_PATH` | Path to the injection library | (required) |
| `CUPTI_TRACE_OUTPUT_FILE` | Output file path (use "stdout" for console) | `cupti_trace_output.txt` |
| `CUPTI_ENABLE_DRIVER` | Enable driver API tracing (0/1) | `0` |
| `CUPTI_ENABLE_MEMORY` | Enable memory operation tracing (0/1) | `0` |
| `CUPTI_ENABLE_NVTX` | Enable NVTX annotation tracing (0/1) | `0` |
| `CUPTI_ENABLE_GRAPH` | Enable CUDA graph node tracking (0/1) | `0` |
| `CUPTI_ENABLE_PC_SAMPLING` | Enable PC sampling (0/1) | `0` |

## Build

```bash
# Ensure CUDA_PATH is set
export CUDA_PATH=/usr/local/cuda

# Compile the injection library
g++ -shared -fPIC cupti_trace_injection.cpp \
    -I${CUDA_PATH}/include \
    -I${CUDA_PATH}/extras/CUPTI/include \
    -L${CUDA_PATH}/lib64 \
    -L${CUDA_PATH}/extras/CUPTI/lib64 \
    -lcupti -lcuda \
    -o libcupti_trace_injection.so
```

## Output Format

The trace output includes detailed information for each activity:

### Kernel Execution
```
CONCURRENT_KERNEL [ start_ns, end_ns ] duration ns, "kernel_name", correlationId N
    grid [ X, Y, Z ], block [ X, Y, Z ], cluster [ X, Y, Z ], sharedMemory (static N, dynamic N)
    deviceId N, contextId N, streamId N, graphId N, graphNodeId N, channelId N, channelType TYPE
```

### Memory Copy
```
MEMCPY "HtoD" [ start_ns, end_ns ] duration ns, size N, copyCount N, srcKind PAGEABLE, dstKind DEVICE, correlationId N
    deviceId N, contextId N, streamId N, graphId N, graphNodeId N, channelId N, channelType TYPE
```

### Runtime API
```
RUNTIME [ start_ns, end_ns ] duration ns, "cudaMemcpy", cbid N, processId N, threadId N, correlationId N
```

## Architecture

### Workflow

1. **Initialization** (`InitializeInjection`):
   - Called automatically when the library is injected
   - Sets up CUPTI callbacks and activity tracking
   - Registers exit handler for cleanup

2. **Profiler Control**:
   - `cuProfilerStart`: Enables activity collection
   - `cuProfilerStop`: Flushes and disables activities

3. **Activity Collection**:
   - CUPTI requests buffers via `BufferRequested` callback
   - Activities are recorded to buffers
   - Completed buffers are processed via `BufferCompleted` callback

4. **Exit Handling**:
   - `AtExitHandler`: Force-flushes all pending activities
   - Ensures no data is lost on application exit

### Key Components

- `cupti_trace.cpp`: Main injection logic, callback handlers, activity selection, buffer management
- `cupti_trace_print.h`: Activity record printing functions with CUDA graph support
  - String conversion utilities for activity types, memory kinds, etc.
  - Comprehensive PrintActivity function for all supported activity types
  - Graph node tracking and correlation with API calls

## Example: CUDA Graph Tracing

The repository includes a simple CUDA graph example that demonstrates the performance benefits of CUDA graphs and how to trace them with CUPTI.

### Build and Run

```bash
# Build the example
cd /home/yunwei37/workspace/xpu-perf/test/mock-app
make cuda_graph_simple

# Run without tracing
./cuda_graph_simple [iterations] [vector_size]
# Example: ./cuda_graph_simple 10 1000000

# Run with CUPTI tracing
env CUDA_INJECTION64_PATH=/path/to/libcupti_trace_injection.so \
    CUPTI_TRACE_OUTPUT_FILE=/tmp/cuda_graph_trace.txt \
    ./cuda_graph_simple

# Enable all tracing features including graph tracking
env CUDA_INJECTION64_PATH=/path/to/libcupti_trace_injection.so \
    CUPTI_TRACE_OUTPUT_FILE=/tmp/cuda_graph_trace.txt \
    CUPTI_ENABLE_DRIVER=1 \
    CUPTI_ENABLE_MEMORY=1 \
    CUPTI_ENABLE_GRAPH=1 \
    ./cuda_graph_simple 5 100000
```

### What the Example Shows

The `cuda_graph_simple` example:
- Compares performance with and without CUDA graphs
- Demonstrates CUDA graph capture and instantiation
- Shows significant overhead reduction (typically 8-180x speedup)
- Traces both regular kernel launches and graph executions

### Understanding the Trace Output

When you examine `/tmp/cuda_graph_trace.txt`, you'll see:

1. **Regular kernel launches** (without graph):
   ```
   RUNTIME "cudaLaunchKernel_v7000", correlationId=9
   CONCURRENT_KERNEL name="_Z9vectorAddPKfS0_Pfi", correlationId=9
     grid=[391,1,1], block=[256,1,1], streamId=7
   ```

2. **Graph capture phase**:
   ```
   RUNTIME "cudaStreamBeginCapture_v10000"
   RUNTIME "cudaLaunchKernel_v7000"  (captured, not executed)
   RUNTIME "cudaStreamEndCapture_v10000"
   RUNTIME "cudaGraphInstantiate_v12000"
   ```

3. **Graph execution** (notice `graphId` and `graphNodeId`):
   ```
   RUNTIME "cudaGraphLaunch_v10000", correlationId=29
   CONCURRENT_KERNEL name="_Z9vectorAddPKfS0_Pfi", correlationId=29
     streamId=13, graphId=2, graphNodeId=8589934592
   ```

The key difference: graph executions have `graphId` and `graphNodeId` fields, showing they're part of a pre-compiled graph.

## Use Cases

- **Performance profiling**: Understand kernel execution times and memory transfer patterns
- **Debugging**: Trace API calls and activity sequences
- **Optimization**: Identify performance bottlenecks in CUDA applications
- **Testing**: Verify correct API usage and activity patterns
- **CUDA Graph analysis**: Validate graph structure and execution patterns

## Limitations

- Adds overhead to the traced application (typically <5% for most workloads)
- PC sampling requires additional GPU capabilities
- Some activities may not be supported on older CUDA architectures
- Output files can become large for long-running applications

## References

- [CUPTI Documentation](https://docs.nvidia.com/cupti/)
- [CUDA Documentation](https://docs.nvidia.com/cuda/)
- NVIDIA CUPTI Samples (included with CUDA Toolkit)
