# GPU+CPU Performance Profiler

A performance profiler that combines CUPTI GPU tracing with eBPF uprobe CPU profiling to generate unified flamegraphs showing complete CPU→GPU execution flows.

## Features

- **Integrated CPU+GPU profiling**: Correlates CPU stack traces with GPU kernel executions
- **CUPTI integration**: Uses named pipes to collect GPU kernel launches and memory events in real-time
- **eBPF uprobes**: Captures CPU call stacks at `cudaLaunchKernel` invocation points
- **Timestamp correlation**: Matches CPU and GPU events with configurable tolerance windows
- **Three profiling modes**:
  - **CPU-only**: Captures CPU stack traces without GPU overhead
  - **GPU-only**: Records GPU kernel execution times
  - **Merged** (default): Correlates and combines CPU+GPU traces
- **Flamegraph-compatible output**: Generates folded stack format for visualization

## How It Works

1. Creates a named pipe for CUPTI trace output
2. Starts CUPTI parser in background to read GPU events (JSON format)
3. Attaches eBPF uprobe to `cudaLaunchKernel` in CUDA runtime library
4. Launches target CUDA application with CUPTI injection
5. Correlates CPU stacks with GPU kernels using timestamps and correlation IDs
6. Outputs merged stacks in folded format weighted by GPU kernel duration

## Building

The profiler uses a Makefile to build both the CUPTI library and the Go profiler binary with the embedded CUPTI library:

```bash
cd /home/yunwei37/workspace/xpu-perf/profiler
make        # Build everything (CUPTI library + profiler)
```

Available Makefile targets:
- `make` or `make build` - Build the profiler with embedded CUPTI library
- `make cupti` - Build only the CUPTI library
- `make clean` - Clean all build artifacts
- `make test` - Run basic functionality test
- `make install` - Install binary to /usr/local/bin
- `make help` - Show help message

## Usage

```bash
# Full CPU+GPU profiling (default mode)
sudo ./xpu-perf -o merged_trace.folded ./my_cuda_app [args...]

# GPU-only profiling
sudo ./xpu-perf -gpu-only -o gpu_trace.folded ./my_cuda_app [args...]

# CPU-only profiling
sudo ./xpu-perf -cpu-only -o cpu_trace.folded ./my_cuda_app [args...]

# Custom CUDA library path
sudo ./xpu-perf -cuda-lib /usr/local/cuda/lib64/libcudart.so.12 -o trace.folded ./app

# Generate flamegraph
flamegraph.pl merged_trace.folded > flamegraph.svg
```

## Command-Line Options

```
-o string
    Output file for folded stack traces (default "merged_trace.folded")

-cupti-lib string
    Path to CUPTI trace injection library (uses embedded library if not specified)

-cuda-lib string
    Path to CUDA runtime library (auto-detected if not specified)

-cpu-only
    Only collect CPU traces (no GPU profiling)

-gpu-only
    Only collect GPU traces (no CPU profiling)

-merge
    Merge CPU and GPU traces (default: true)
```

## Output Format

The profiler generates folded stack traces compatible with flamegraph.pl:

### CPU-only mode
```
libcudart.so.12+0x7ce80;app+0x4d05;app+0x472c;libc.so.6+0x2a1c9 3781
```

### GPU-only mode
```
[GPU_Kernel]_Z13softmaxKernelPKfPfmm 2616491
[GPU_Kernel]_Z15layerNormKernelPKfPfS0_S0_mmm 621123
```

### Merged CPU+GPU mode
```
libcudart.so.12+0x7ce80;app+0x4d05;app+0x472c;libc.so.6+0x2a1c9;[GPU_Kernel]_Z13softmaxKernelPKfPfmm 2543061
libcudart.so.12+0x7ce80;app+0x4d05;app+0x4929;libc.so.6+0x2a1c9;[GPU_Kernel]_Z15layerNormKernelPKfPfS0_S0_mmm 244659
```

The numbers at the end represent:
- **CPU-only**: Sample count (number of times this stack was captured)
- **GPU-only**: Total GPU kernel duration in microseconds
- **Merged**: Total GPU kernel duration in microseconds for this CPU→GPU call path

## Example

```bash
# Profile an LLM inference workload
sudo ./xpu-perf -o llm_profile.folded ./llm-inference

# Output:
# Final stats - Total: 5580, Uprobe: 5372, Sampling: 208
# Correlation complete: matched=5372, unmatched=0
# Wrote 8 unique stacks (3358594 total samples) to llm_profile.folded

# Generate flamegraph
flamegraph.pl llm_profile.folded > llm_flamegraph.svg
```

## Requirements

- Root privileges (for eBPF and uprobes)
- CUDA Toolkit installed (for building the CUPTI library)
- Linux kernel with eBPF support (kernel 5.10+)
- Go 1.19+ (for building the profiler)
- Make (for build automation)

**Note:** The CUPTI trace library is embedded in the profiler binary, so no additional runtime dependencies are needed beyond CUDA.

## Architecture

### Components

1. **main.go**: Orchestrates the profiler, manages target process lifecycle
2. **cupti_parser.go**: Parses JSON-formatted CUPTI trace events from named pipe
3. **correlation.go**: Correlates CPU stacks with GPU kernels using timestamp matching
4. **simple_reporter.go**: Implements TraceReporter interface to capture uprobe events

### Correlation Strategy

The profiler uses a two-step correlation:

1. **Correlation ID matching**: CUPTI provides correlation IDs linking runtime API calls to kernel executions
2. **Timestamp matching**: Matches CPU uprobe events (nanosecond timestamps) with GPU runtime API calls

If CPU and GPU event counts match, sequential 1:1 matching is used. Otherwise, time-based matching with a configurable tolerance window (default 10ms) is applied.

## Limitations

- Requires longer-running applications for reliable uprobe capture (>1 second)
- Very short CUDA applications may complete before uprobes fully attach
- Stack symbol resolution limited to available symbols and file+offset information
- CUPTI overhead typically <5% for most workloads

## Troubleshooting

**No CPU traces captured:**
- Ensure target application runs long enough for uprobes to attach (add delay if needed)
- Verify CUDA library path is correct (`ldd ./your_app | grep cudart`)
- Check that `cudaLaunchKernel` symbol exists (`nm -D /path/to/libcudart.so.12 | grep LaunchKernel`)

**No GPU traces captured:**
- Verify CUPTI injection library path is correct
- Check that target application is using CUDA
- Ensure CUPTI_TRACE_OUTPUT_FILE environment variable is being set

**Correlation errors:**
- Increase tolerance window in correlation.go if timestamps are misaligned
- Check clock synchronization between CPU and GPU timers

## References

- [CUPTI Documentation](https://docs.nvidia.com/cupti/)
- [eBPF Profiler](https://github.com/elastic/otel-profiling-agent)
- [Flamegraph](https://github.com/brendangregg/FlameGraph)
