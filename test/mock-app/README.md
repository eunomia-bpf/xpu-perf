# Mock CUDA Applications for Testing

This directory contains simple CUDA applications for testing and demonstrating profiling capabilities.

## Available Examples

### 1. vectorAdd
Simple vector addition benchmark.

```bash
./vectorAdd
```

### 2. llm-inference
Mock LLM inference workload with multiple layers and attention mechanisms.

```bash
./llm-inference
```

### 3. cuda_graph_simple
Demonstrates CUDA graphs with performance comparison.

```bash
# Run with default settings (10 iterations, 1M elements)
./cuda_graph_simple

# Custom iterations and vector size
./cuda_graph_simple 20 500000
```

**Output:**
- Compares performance with/without CUDA graphs
- Shows speedup (typically 8-180x)
- Verifies correctness

## Building

```bash
# Build all examples
make

# Build specific example
make cuda_graph_simple
make vectorAdd
make llm-inference

# Clean
make clean
```

## Profiling with CUPTI

All examples can be profiled using CUPTI trace injection:

```bash
# Basic tracing
env CUDA_INJECTION64_PATH=/path/to/libcupti_trace_injection.so \
    CUPTI_TRACE_OUTPUT_FILE=/tmp/trace.txt \
    ./cuda_graph_simple

# Full tracing with driver and memory operations
env CUDA_INJECTION64_PATH=/path/to/libcupti_trace_injection.so \
    CUPTI_TRACE_OUTPUT_FILE=/tmp/trace.txt \
    CUPTI_ENABLE_DRIVER=1 \
    CUPTI_ENABLE_MEMORY=1 \
    ./cuda_graph_simple
```

## Using with xpu-perf

The examples work with the xpu-perf profiling infrastructure:

```bash
# From cupti_trace directory
cd ../../cupti_trace

# Profile CUDA graph example
env CUDA_INJECTION64_PATH=$PWD/libcupti_trace_injection.so \
    CUPTI_TRACE_OUTPUT_FILE=/tmp/cuda_graph.txt \
    ../test/mock-app/cuda_graph_simple

# View trace
cat /tmp/cuda_graph.txt | grep CONCURRENT_KERNEL
```

## CUDA Graph Tracing Output

When tracing `cuda_graph_simple`, you'll see:

### Regular Kernel Launches
```
RUNTIME "cudaLaunchKernel_v7000", correlationId=9
CONCURRENT_KERNEL name="_Z9vectorAddPKfS0_Pfi", correlationId=9
  grid=[391,1,1], block=[256,1,1], streamId=7
```

### Graph Capture
```
RUNTIME "cudaStreamBeginCapture_v10000"
RUNTIME "cudaStreamEndCapture_v10000"
RUNTIME "cudaGraphInstantiate_v12000"
```

### Graph Execution
```
RUNTIME "cudaGraphLaunch_v10000", correlationId=29
CONCURRENT_KERNEL name="_Z9vectorAddPKfS0_Pfi", correlationId=29
  streamId=13, graphId=2, graphNodeId=8589934592
```

**Key Indicator:** Graph executions include `graphId` and `graphNodeId` fields.

## Requirements

- CUDA Toolkit 11.0+ (tested with 12.9)
- GPU with compute capability 3.5+
- For CUPTI tracing: libcupti.so from CUDA extras

## Performance Notes

### CUDA Graphs Benefits
- Reduced CPU overhead (8-180x in this example)
- Better for workloads with many small kernels
- Most effective when kernel launch overhead dominates

### Typical Speedups by Workload
- Small kernels (<1ms): 50-200x
- Medium kernels (1-10ms): 5-20x
- Large kernels (>10ms): 2-5x

## Troubleshooting

### Build Errors

If you see linking errors, use `--no-device-link` flag:
```bash
nvcc -o cuda_graph_simple cuda_graph_simple.cu --no-device-link
```

### CUPTI Not Found

Set CUDA_PATH:
```bash
export CUDA_PATH=/usr/local/cuda-12.9
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$CUDA_PATH/extras/CUPTI/lib64:$LD_LIBRARY_PATH
```

### No GPU Available

The examples will fail gracefully if no GPU is detected:
```
CUDA error: no CUDA-capable device is detected
```
