# CUPTI Trace Injection Library

## Overview

The CUPTI Trace Injection Library is a powerful profiling tool that enables automatic tracing and performance monitoring of any CUDA application without requiring source code modification. It leverages NVIDIA's CUPTI (CUDA Profiling Tools Interface) to collect detailed runtime information about CUDA kernels, memory operations, and API calls.

## Directory Structure

```
cupti_trace/
├── Core Libraries & Headers
│   ├── cupti_trace_injection.cpp    # Main injection library source
│   ├── helper_cupti.h               # Error handling and utilities
│   ├── helper_cupti_activity.h      # Activity management framework
│   └── libcupti_trace_injection.so  # Compiled injection library
├── Python Tools
│   ├── cupti_trace_parser.py        # CUPTI trace parser module
│   ├── gpuperf.py                   # Automated profiling tool
│   └── merge_gpu_cpu_trace.py       # Trace merger utility
├── Visualization
│   └── combined_flamegraph.pl       # Flamegraph generator
├── Build System
│   └── Makefile                     # Cross-platform build configuration
└── Generated Artifacts (examples)
    ├── gpu_results.txt              # Raw CUPTI trace
    ├── gpu_results.json             # Chrome Trace Format
    ├── cpu_results.txt              # CPU profiling data
    ├── merged_trace.folded          # Merged folded stacks
    └── merged_flamegraph.svg        # Interactive flamegraph
```

## Table of Contents

1. [Features](#features)
2. [Architecture](#architecture)
3. [Components](#components)
4. [Workflow](#workflow)
5. [Call Graph](#call-graph)
6. [Building](#building)
7. [Usage](#usage)
8. [Output Formats](#output-formats)
9. [Technical Details](#technical-details)

## Features

- **Zero Source Code Modification**: Inject profiling capabilities into any CUDA application
- **Comprehensive Activity Tracking**: 
  - CUDA Runtime API calls
  - CUDA Driver API calls
  - Kernel executions (concurrent and sequential)
  - Memory operations (memcpy, memset, memory allocations)
  - NVTX markers and ranges
  - Overhead measurements
- **Multiple Output Formats**:
  - Raw CUPTI trace output
  - Chrome Trace Format (viewable in chrome://tracing)
- **Cross-platform Support**: Works on Linux and Windows
- **Context-aware Profiling**: Support for cudaProfilerStart/Stop APIs

## Architecture

```
┌─────────────────────┐
│   CUDA Application  │
└──────────┬──────────┘
           │
    ┌──────▼──────┐
    │ CUDA Runtime│ ← Injection Point (CUDA_INJECTION64_PATH)
    └──────┬──────┘
           │
┌──────────▼──────────────┐
│ libcupti_trace_injection│
│        Library          │
├─────────────────────────┤
│ • InitializeInjection() │
│ • Callback Handlers     │
│ • Activity Management   │
└──────────┬──────────────┘
           │
    ┌──────▼──────┐
    │    CUPTI    │
    └─────────────┘
```

## Components

### Core Libraries and Headers

#### 1. **cupti_trace_injection.cpp** (Main Library)
The core injection library that implements the CUPTI-based tracing mechanism.

#### Key Data Structures:

```cpp
typedef struct InjectionGlobals_st {
    volatile uint32_t       initialized;      // Initialization flag
    CUpti_SubscriberHandle  subscriberHandle; // CUPTI subscriber handle
    int                     tracingEnabled;   // Tracing state
    uint64_t                profileMode;      // Bitmask of enabled activities
} InjectionGlobals;
```

#### Main Functions:

- **`InitializeInjection()`**: Entry point called by CUDA runtime
- **`SetupCupti()`**: Initialize CUPTI and register callbacks
- **`EnableCuptiActivities()`**: Enable specific activity types
- **`DisableCuptiActivities()`**: Disable activity collection
- **`InjectionCallbackHandler()`**: Main callback dispatcher
- **`AtExitHandler()`**: Cleanup and flush buffers on exit

### 2. **helper_cupti_activity.h**
Comprehensive CUPTI activity management framework providing core functionality for activity tracking.

#### Key Features:
- **Activity Buffer Management**: 
  - 8MB default buffer size (configurable via BUF_SIZE)
  - 8-byte alignment for optimal performance
  - Automatic buffer request/completion callbacks
- **Activity Record Processing**:
  - Supports 40+ activity types (KERNEL, MEMCPY, RUNTIME, DRIVER, etc.)
  - Detailed record printing with all relevant fields
  - Correlation ID tracking for API-to-kernel mapping
- **Global State Management**:
  ```cpp
  typedef struct GlobalState_st {
      CUpti_SubscriberHandle subscriberHandle;
      size_t activityBufferSize;
      FILE *pOutputFile;
      void *pUserData;
      uint64_t buffersRequested;
      uint64_t buffersCompleted;
  } GlobalState;
  ```
- **User Configuration Options**:
  ```cpp
  typedef struct UserData_st {
      size_t activityBufferSize;
      size_t deviceBufferSize;
      uint8_t flushAtStreamSync;
      uint8_t flushAtCtxSync;
      uint8_t printActivityRecords;
      void (*pPostProcessActivityRecords)(CUpti_Activity *pRecord);
  } UserData;
  ```

#### Core Functions:
- `InitCuptiTrace()`: Initialize CUPTI subsystem with callbacks
- `BufferRequested()`: Allocate aligned buffers for activity records
- `BufferCompleted()`: Process completed buffers and print records
- `PrintActivityBuffer()`: Parse and output activity records
- `HandleDomainStateCallback()`: Process domain state changes
- `GetActivityKindString()`: Convert activity enums to readable strings

### 3. **helper_cupti.h**
Common error handling and utility macros for CUPTI operations.

#### Key Macros:
- **Error Handling Macros**:
  - `CUPTI_API_CALL()`: Check CUPTI calls and exit on error
  - `CUPTI_API_CALL_VERBOSE()`: Same with verbose logging
  - `DRIVER_API_CALL()`: Check CUDA Driver API calls
  - `RUNTIME_API_CALL()`: Check CUDA Runtime API calls
  - `MEMORY_ALLOCATION_CALL()`: Verify memory allocations
  - `CHECK_CONDITION()`: General condition checking
  - `CHECK_INTEGER_CONDITION()`: Integer comparison checking

#### Constants:
- `CUDA_MAX_DEVICES`: 256 (theoretical maximum)
- `DEV_NAME_LEN`: 256 bytes for device names
- `EXIT_WAIVED`: 2 (special exit code)

### Python Tools and Scripts

#### 4. **cupti_trace_parser.py**
Python module for parsing CUPTI trace data with support for various event types.

##### Key Features:
- Parses CUPTI trace output into structured data
- Regular expression patterns for different trace formats:
  - Runtime API events
  - Driver API events  
  - Kernel executions
  - Memory operations (MEMCPY, MEMSET, MEMORY2)
  - Overhead measurements
  - Grid and device information
- Converts traces to Chrome Trace Format (JSON)
- Extensible parser class for custom processing

#### 5. **gpuperf.py**
High-level GPU profiling orchestration tool that automates the profiling workflow.

##### Key Features:
- Automatic CUPTI injection setup
- Integration with CPU profiler (if available)
- Temporary file management
- Process management for profiled applications
- Automatic trace parsing and conversion
- Support for both GPU-only and combined CPU/GPU profiling

#### 6. **merge_gpu_cpu_trace.py**
Tool for merging GPU and CPU traces into unified visualization formats.

##### Key Classes:
- `GPUEvent`: Represents GPU operations (kernels, memory transfers)
- `CPUSample`: Represents CPU stack samples
- `TraceMerger`: Combines traces based on timestamp alignment

##### Output Formats:
- Folded flamegraph format for combined visualization
- Chrome Trace Format with merged timeline
- Support for correlating CPU stacks with GPU operations

#### 7. **combined_flamegraph.pl**
Perl script for generating flamegraph visualizations from merged trace data.

##### Features:
- Processes folded stack format
- Generates SVG flamegraphs
- Combines CPU and GPU execution visualization
- Color coding for different operation types

### Build System

#### 8. **Makefile**
Build system for the injection library with platform-specific configurations.

##### Key Features:
- Auto-detection of CUDA installation path
- Cross-platform support (Linux, Windows, macOS)
- Architecture-specific builds (x86_64, ARM/aarch64)
- Automatic library path configuration
- Support for CUDA versions (default: 13.0)

## Workflow

### Initialization Phase

```
1. Application starts
   ↓
2. CUDA runtime detects CUDA_INJECTION64_PATH
   ↓
3. Loads libcupti_trace_injection.so/dll
   ↓
4. Calls InitializeInjection()
   ↓
5. InitializeInjection() performs:
   • Initialize global state
   • Register atexit handler
   • Setup CUPTI callbacks
   • Enable activity collection
   ↓
6. Application continues normal execution
```

### Runtime Phase

```
Application makes CUDA call
   ↓
CUPTI Callback triggered (ENTER)
   ↓
InjectionCallbackHandler() processes:
   • Domain check (STATE/DRIVER/RUNTIME)
   • Callback ID check
   • Context-specific handling
   ↓
CUDA operation executes
   ↓
CUPTI Callback triggered (EXIT)
   ↓
Activity records generated
   ↓
Buffer management (if needed)
```

### Termination Phase

```
Application exits or cudaDeviceReset
   ↓
AtExitHandler() called
   ↓
cuptiActivityFlushAll(1) - Force flush
   ↓
Process remaining buffers
   ↓
Output final records
```

## Call Graph

### Main Execution Flow

```
InitializeInjection()
├── InitializeInjectionGlobals()
├── RegisterAtExitHandler()
│   └── atexit(AtExitHandler)
└── SetupCupti()
    ├── InitCuptiTrace()
    │   ├── cuptiSubscribe()
    │   ├── cuptiEnableCallback()
    │   └── cuptiActivityRegisterCallbacks()
    ├── cuptiEnableCallback(cuProfilerStart)
    ├── cuptiEnableCallback(cuProfilerStop)
    └── EnableCuptiActivities()
        └── cuptiActivityEnable() [for each activity type]
```

### Callback Processing

```
InjectionCallbackHandler(domain, callbackId, callbackData)
├── CUPTI_CB_DOMAIN_STATE
│   └── HandleDomainStateCallback()
├── CUPTI_CB_DOMAIN_DRIVER_API
│   ├── CUPTI_DRIVER_TRACE_CBID_cuProfilerStart
│   │   └── OnProfilerStart()
│   │       └── EnableCuptiActivities(context)
│   └── CUPTI_DRIVER_TRACE_CBID_cuProfilerStop
│       └── OnProfilerStop()
│           ├── cuptiActivityFlushAll()
│           └── DisableCuptiActivities(context)
└── CUPTI_CB_DOMAIN_RUNTIME_API
    └── CUPTI_RUNTIME_TRACE_CBID_cudaDeviceReset
        └── OnCudaDeviceReset()
            └── cuptiActivityFlushAll()
```

### Activity Management

```
EnableCuptiActivities(context)
├── cuptiEnableCallback(cudaDeviceReset)
├── SelectActivities()
│   ├── CUPTI_ACTIVITY_KIND_DRIVER
│   ├── CUPTI_ACTIVITY_KIND_RUNTIME
│   ├── CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL
│   ├── CUPTI_ACTIVITY_KIND_MEMCPY
│   ├── CUPTI_ACTIVITY_KIND_MEMSET
│   └── [... other activities]
└── For each selected activity:
    ├── If context == NULL:
    │   └── cuptiActivityEnable(activity)
    └── Else:
        ├── cuptiActivityEnableContext(context, activity)
        └── If CUPTI_ERROR_INVALID_KIND:
            └── cuptiActivityEnable(activity)
```

## Building

### Linux
```bash
make
# Output: libcupti_trace_injection.so
```

### Windows
1. Build Detours library (see usage.txt)
2. Run nmake
3. Output: libcupti_trace_injection.dll

## Generated Files and Artifacts

### Build Artifacts
- **libcupti_trace_injection.so** (Linux) / **libcupti_trace_injection.dll** (Windows)
  - The compiled injection library
  - Must be loaded via CUDA_INJECTION64_PATH environment variable

### Trace Output Files
- **gpu_results.txt**: Raw CUPTI trace output with activity records
- **gpu_results.json**: Chrome Trace Format JSON for visualization
- **cpu_results.txt**: CPU profiling data (when using combined profiling)
- **merged_trace.folded**: Folded stack format for flamegraph generation
- **merged_flamegraph.svg**: Interactive SVG flamegraph visualization

## Usage

### Basic Usage

```bash
# Set injection path
export CUDA_INJECTION64_PATH=/path/to/libcupti_trace_injection.so

# Optional: Enable NVTX support
export NVTX_INJECTION64_PATH=/usr/local/cuda/extras/CUPTI/lib64/libcupti.so

# Set library path
export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

# Run your CUDA application
./your_cuda_app > trace_output.txt
```

### Using the Python Tools

#### Parse and Convert Traces
```bash
# Parse CUPTI trace output
python3 cupti_trace_parser.py -i trace_output.txt -o parsed_trace.json
```

#### Automated GPU Profiling
```bash
# Profile a CUDA application
python3 gpuperf.py ./your_cuda_app [args]

# With custom output file
python3 gpuperf.py -o my_profile.json ./your_cuda_app

# Combined CPU/GPU profiling (requires CPU profiler)
python3 gpuperf.py --cpu ./your_cuda_app
```

#### Merge GPU and CPU Traces
```bash
# Merge traces into folded format
python3 merge_gpu_cpu_trace.py --gpu gpu_results.json --cpu cpu_results.txt -o merged.folded

# Generate flamegraph from merged trace
./combined_flamegraph.pl merged.folded > merged_flamegraph.svg
```

### Converting to Chrome Trace Format

```bash
# View in Chrome
# 1. Open Chrome browser
# 2. Navigate to chrome://tracing
# 3. Load gpu_results.json or any generated JSON trace
```

## Output Formats

### Raw CUPTI Trace
```
RUNTIME [ 1234567890, 1234567900 ] duration 10, "cudaMalloc", cbid 123, processId 456, threadId 789, correlationId 111
CONCURRENT_KERNEL [ 1234567900, 1234568000 ] duration 100, "myKernel", correlationId 111
    grid [ 256, 1, 1 ], block [ 128, 1, 1 ]
    deviceId 0, contextId 1, streamId 2
```

### Chrome Trace Format (JSON)
```json
{
  "name": "Runtime: cudaMalloc",
  "ph": "X",
  "ts": 1234567.890,
  "dur": 0.010,
  "tid": 789,
  "pid": 456,
  "cat": "CUDA_Runtime",
  "args": {
    "cbid": "123",
    "correlationId": 111
  }
}
```

## Technical Details

### Activity Types Tracked

| Activity Kind | Description |
|--------------|-------------|
| DRIVER | CUDA Driver API calls |
| RUNTIME | CUDA Runtime API calls |
| CONCURRENT_KERNEL | Kernel executions |
| MEMCPY | Memory copy operations |
| MEMSET | Memory set operations |
| MEMORY2 | Memory allocations/deallocations |
| OVERHEAD | CUPTI overhead measurements |
| NAME | NVTX naming |
| MARKER | NVTX markers |
| MARKER_DATA | NVTX marker data |

### Buffer Management

- **Default Buffer Size**: 8MB (configurable)
- **Alignment**: 8-byte aligned
- **Flushing Strategy**:
  - Automatic flush on buffer full
  - Manual flush on cudaDeviceReset
  - Force flush on exit (incomplete records included)

### Thread Safety

- Mutex protection for initialization
- CUPTI handles thread safety for callbacks
- Activity buffers are thread-local

### Platform-Specific Features

#### Windows
- Uses Detours library for exit handling
- Intercepts RtlExitUserProcess for reliable cleanup

#### Linux
- Uses standard atexit() handler
- Simpler cleanup mechanism

## Performance Considerations

1. **Overhead**: Tracing adds 5-15% overhead typically
2. **Buffer Size**: Larger buffers reduce flush frequency but increase memory usage
3. **Activity Selection**: Enable only needed activities to reduce overhead
4. **Context vs Global**: Context-specific activities have lower overhead

## Limitations

1. Some activities cannot be enabled per-context (fallback to global)
2. Force flush may include incomplete records
3. Windows requires Detours library for proper exit handling
4. NVTX support requires separate injection path

## Troubleshooting

### Common Issues

1. **Library not loading**: Check CUDA_INJECTION64_PATH is absolute path
2. **Missing activities**: Ensure CUPTI library path in LD_LIBRARY_PATH
3. **No output**: Check if application uses CUDA (injection only works with CUDA apps)
4. **Incomplete traces**: Allow proper cleanup, avoid force-killing application

### Debug Tips

- Enable verbose output by modifying CUPTI_API_CALL to CUPTI_API_CALL_VERBOSE
- Check cuptiGetLastError() returns for silent failures
- Use smaller test applications to verify setup

## License

Copyright 2021-2024 NVIDIA Corporation. All rights reserved.

## References

- [CUPTI Documentation](https://docs.nvidia.com/cuda/cupti/)
- [Chrome Trace Event Format](https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/)
- [NVTX Documentation](https://docs.nvidia.com/nvtx/)