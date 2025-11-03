# GPU+CPU Performance Profiler

A performance profiler that combines CUPTI GPU tracing with eBPF CPU profiling to generate unified flamegraphs showing complete CPU→GPU execution flows with accurate time attribution.

## Features

- **Three profiling modes**:
  - **Default (Merge)**: CPU sampling + GPU kernels with CPU call stacks (shows both CPU activity and GPU causality)
  - **GPU-only**: CPU→GPU causality via uprobes (shows which CPU code launched which GPU kernel)
  - **CPU-only**: Pure CPU sampling without GPU overhead
- **CUPTI integration**: Real-time GPU kernel launch and execution tracking via named pipes
- **eBPF uprobes + sampling**: Captures CPU call stacks at `cudaLaunchKernel` and periodic sampling
- **Accurate time attribution**: Converts GPU kernel duration to sample counts using CPU sampling frequency
- **Symbol resolution**: Resolves function names from ELF symbol tables with C++ demangling
- **Flamegraph-compatible output**: Generates folded stack format for visualization

## How It Works

### Default (Merge) Mode
1. Attaches eBPF uprobe to `cudaLaunchKernel` to capture kernel launch call stacks
2. Enables eBPF-based CPU sampling at 50 Hz to capture CPU activity
3. Creates named pipe for CUPTI to output GPU kernel execution events (JSON format)
4. Starts target CUDA application with CUPTI injection library
5. Correlates uprobes with GPU kernels using correlation IDs and timestamps
6. Converts GPU kernel durations to equivalent sample counts: `samples = (durationNs * samplesPerSec) / 1e9`
7. Outputs merged stacks: CPU sampling (pure) + GPU kernels with CPU context (causality)

### GPU-only Mode
- Only collects uprobes at `cudaLaunchKernel` (no CPU sampling)
- Shows full CPU call path leading to each GPU kernel launch
- Sample counts represent GPU kernel execution time

### CPU-only Mode
- Only collects CPU sampling traces
- No GPU overhead, no CUPTI injection

## Building

```bash
cd /home/yunwei37/workspace/xpu-perf/profiler
make        # Build everything (CUPTI library + profiler with embedded library)
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
# Default: Merge CPU sampling + GPU kernels with call stacks
sudo ./profile -o merged_trace.folded ./my_cuda_app [args...]

# GPU-only: CPU→GPU causality (shows which CPU code launched kernels)
sudo ./profile -gpu-only -o gpu_trace.folded ./my_cuda_app [args...]

# CPU-only: Pure CPU sampling
sudo ./profile -cpu-only -o cpu_trace.folded ./my_cuda_app [args...]

# Custom CUDA library path
sudo ./profile -cuda-lib /usr/local/cuda-12.9/lib64/libcudart.so.12 -o trace.folded ./app

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
    Only collect CPU sampling traces (no GPU)

-gpu-only
    CPU→GPU causality via uprobes (shows which CPU code launched which GPU kernel)
```

## Output Format

The profiler generates folded stack traces compatible with flamegraph.pl. Sample counts are scaled to match CPU sampling frequency (50 Hz).

### CPU-only mode
```
main;InferencePipeline::runRequest;TokenEmbedding::embed 42
main;InferencePipeline::runRequest;TransformerLayer::forward 156
```
Sample count = number of times this stack was captured during sampling

### GPU-only mode
```
main;InferencePipeline::runRequest;TransformerLayer::forward;cudaLaunchKernel;[GPU_Kernel]_Z13softmaxKernelPKfPfmm 181
main;InferencePipeline::runRequest;TransformerLayer::forward;cudaLaunchKernel;[GPU_Kernel]_Z15layerNormKernelPKfPfS0_S0_mmm 52
```
Sample count = GPU kernel duration converted to samples at 50 Hz (e.g., 181 samples = 3.62 seconds GPU time)

### Default (Merge) mode
```
# CPU sampling stacks (no GPU kernel)
main;InferencePipeline::runRequest;TokenEmbedding::embed 42
main;cudaSetDevice;cuDevicePrimaryCtxRetain 15

# GPU kernel stacks (with CPU context from uprobe)
main;InferencePipeline::runRequest;TransformerLayer::forward;cudaLaunchKernel;[GPU_Kernel]_Z13softmaxKernelPKfPfmm 181
main;InferencePipeline::runRequest;TransformerLayer::forward;cudaLaunchKernel;[GPU_Kernel]_Z15layerNormKernelPKfPfS0_S0_mmm 52
```
Shows both CPU activity and GPU execution with proper time attribution

## Example

```bash
# Profile an LLM inference workload (10 seconds)
sudo ./profile -o llm_profile.folded ./llm-inference

# Output:
# Final stats - Total: 8529, Uprobe: 7728, Sampling: 801
# Correlation complete: matched=7728, unmatched=0
# Wrote 184 unique stacks (769 total samples) to llm_profile.folded

# Analysis:
# - 769 samples over ~10 seconds at 50 Hz ≈ expected
# - GPU samples: 243 (31.6% of time)
# - CPU samples: 526 (68.4% of time)

# Generate flamegraph
flamegraph.pl llm_profile.folded > llm_flamegraph.svg
```

## Sample Count Calculation

GPU kernel durations are converted to sample counts to match CPU sampling frequency:

```
samples = (kernel_duration_ns * samplesPerSec) / 1e9
```

Example:
- Kernel duration: 36.2 ms = 36,200,000 ns
- Sampling rate: 50 Hz
- Samples: (36,200,000 * 50) / 1,000,000,000 = 1.81 samples ≈ 2 samples

This ensures the flamegraph correctly shows time proportions between CPU and GPU work.

## Requirements

- Root privileges (for eBPF and uprobes)
- CUDA Toolkit installed (for building CUPTI library)
- Linux kernel with eBPF support (kernel 5.10+)
- Go 1.19+ (for building the profiler)
- Make (for build automation)

**Note:** The CUPTI trace library is embedded in the profiler binary, so no additional runtime dependencies are needed beyond CUDA.

## Architecture

### Components

1. **main.go**: Orchestrates profiler, manages target process lifecycle
2. **cupti_parser.go**: Parses JSON-formatted CUPTI trace events from named pipe
3. **correlation.go**: Correlates CPU stacks with GPU kernels; converts GPU time to samples
4. **symbolizer.go**: Resolves symbols from ELF files with C++ demangling
5. **simple_reporter.go**: Implements TraceReporter interface to capture traces

### Correlation Strategy

**Merge mode:**
1. Separates uprobe traces (kernel launches) from CPU sampling traces
2. Matches uprobes with GPU kernels using correlation IDs (1:1 or time-based)
3. Converts each kernel's duration to sample count: `samples = durationNs * 50 / 1e9`
4. Accumulates samples per unique stack using `float64` to avoid rounding errors
5. Rounds to integers only when writing final output

**GPU-only mode:**
- Sequential or time-based matching of uprobes to GPU kernels
- Sample count represents total GPU execution time for that call path

### Symbol Resolution

1. Parses ELF symbol tables (`.symtab` preferred over `.dynsym`)
2. Calculates file offset: `fileOffset = runtimeAddress - mappingStart` (for PIE executables)
3. Looks up symbol by file offset using binary search
4. Demangles C++ symbols using `github.com/ianlancetaylor/demangle`

## Limitations

- Requires root privileges for eBPF
- Symbol resolution requires unstripped binaries or dynamic symbols
- CUPTI overhead typically <5% for most workloads
- Very short applications (<1 second) may not capture enough samples

## Troubleshooting

**No CPU traces captured:**
- Ensure target application runs long enough for uprobes to attach
- Verify CUDA library path: `ldd ./your_app | grep cudart`
- Check uprobe symbol exists: `nm -D /path/to/libcudart.so.12 | grep LaunchKernel`

**No GPU traces captured:**
- Verify target application uses CUDA
- Check CUPTI pipe was created: `ls -l /tmp/cupti_trace_*.pipe`
- Ensure CUPTI library loaded: check app output for "CUPTI trace injection"

**Symbols not resolved:**
- Verify binary is not fully stripped: `file ./your_app` (should show "not stripped")
- For PIE executables, ensure proper address calculation in symbolizer.go
- Check symbol table exists: `nm ./your_app | wc -l` (should show >0)

**Sample counts seem wrong:**
- Verify GPU kernel durations in CUPTI trace are in nanoseconds
- Check sampling frequency matches: default is 50 Hz
- Ensure float64 arithmetic is used (no premature rounding)

## References

- [CUPTI Documentation](https://docs.nvidia.com/cupti/)
- [OpenTelemetry eBPF Profiler](https://github.com/open-telemetry/opentelemetry-ebpf-profiler)
- [Flamegraph](https://github.com/brendangregg/FlameGraph)
- [ELF Symbol Tables](https://refspecs.linuxfoundation.org/elf/elf.pdf)
