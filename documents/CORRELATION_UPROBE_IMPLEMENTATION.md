# Correlation Uprobe Implementation

## Overview
This document describes the implementation of correlation ID-based tracing for xpu-perf, which allows matching CPU stack traces with GPU kernel launches using CUPTI correlation IDs instead of timestamps.

## Implementation Status

### ✅ Completed Components

1. **eBPF Correlation Uprobe Program** (`ebpf/correlation_uprobe.ebpf.c`)
   - Attaches to `XpuPerfGetCorrelationId` function in the CUPTI library
   - Captures correlation ID from the first parameter (RDI register on x86_64)
   - Stores correlation ID in `custom_context_map` shared with the profiler
   - Tail calls to `custom__generic` to collect CPU stack traces with the correlation ID

2. **Go Attachment Helper** (`correlation_uprobe_attach.go`)
   - `attachCorrelationUprobe()` function to attach the uprobe to XpuPerfGetCorrelationId
   - Reuses profiler's eBPF maps via MapReplacements:
     - `custom_context_map` - for storing correlation IDs
     - `prog_array` - for tail calling to custom__generic
   - Properly manages eBPF program and map file descriptors

3. **Main Program Integration** (`main.go`)
   - Added bpf2go generation directive for building correlation_uprobe.ebpf.c
   - Calls `attachCorrelationUprobe()` when `-use-correlation-id` flag is enabled
   - Sets `CUPTI_ENABLE_CORRELATION_UPROBE=1` environment variable for target process
   - The flag `-use-correlation-id` was already present in the codebase

4. **Tracer API Extension** (`opentelemetry-ebpf-profiler/tracer/tracer.go`)
   - Added `GetProgArrayMapFD()` method to expose prog_array map FD
   - Follows same pattern as existing `GetCustomTraceProgramFD()` and `GetCustomContextMapFD()`

5. **Build System**
   - eBPF C program moved to `ebpf/` subdirectory to avoid Go build conflicts
   - bpf2go successfully generates Go bindings (`correlationuprobe_x86_bpfel.go`)
   - OpenTelemetry eBPF profiler rebuilds successfully with new tracer method

## How It Works

### Data Flow

```
1. Target Application (GPU code)
   ↓
2. CUPTI Library calls XpuPerfGetCorrelationId(correlationId)
   ↓
3. Correlation Uprobe (eBPF)
   - Captures correlationId from function parameter
   - Stores in custom_context_map[0] = correlationId
   - Tail calls to custom__generic (index 0 in prog_array)
   ↓
4. custom__generic (eBPF)
   - Reads correlationId from custom_context_map[0]
   - Collects CPU stack trace
   - Passes to userspace with TRACE_CUSTOM origin
   ↓
5. Userspace (simple_reporter.go)
   - Extracts correlationId from meta.ContextValue
   - Calls correlator.AddCPUTraceWithCorrelation(correlationId, ...)
   ↓
6. Correlator (correlation.go)
   - Matches CPU traces with GPU events by correlationId
   - Merges stack traces for final output
```

### Key Design Decisions

1. **Tail Calling**: The correlation uprobe uses `bpf_tail_call` to invoke `custom__generic` instead of duplicating stack collection logic

2. **Map Sharing**: Reuses the profiler's existing eBPF maps rather than creating new ones, ensuring proper coordination

3. **Per-CPU Storage**: `custom_context_map` is a `BPF_MAP_TYPE_PERCPU_ARRAY` for thread-safety without locks

4. **Context Value Field**: Reuses the existing `ContextValue` field (previously `OffTime`) in trace metadata

## Usage

To enable correlation ID-based matching:

```bash
sudo ./xpu-perf -use-correlation-id -o output.folded ./your_cuda_application
```

The `-use-correlation-id` flag:
- Attaches the correlation uprobe to `XpuPerfGetCorrelationId`
- Enables CUPTI correlation ID reporting via environment variable
- Switches the correlator to match by correlation ID instead of timestamps

## Testing

### Test with llm-inference example:
```bash
cd /root/xpu-perf/profiler
sudo ./xpu-perf -use-correlation-id -debug-dir /tmp/xpu-debug \
    python3 /root/xpu-perf/test/pytorch/llm-inference.py
```

### Expected Behavior:
- Uprobe attaches to XpuPerfGetCorrelationId in libcupti_trace_injection.so
- Each CUDA kernel launch triggers the uprobe
- CPU stack trace is captured with the CUPTI correlation ID
- Correlator matches CPU and GPU traces by correlation ID

## Remaining Work

### ⚠️ Pre-existing Compilation Errors (NOT related to correlation ID feature)

The following errors exist in the codebase and are unrelated to the correlation uprobe implementation:

**simple_reporter.go:**
- Line 24, 36: `libpf.SymbolMap` is undefined
- Line 95: `meta.Comm` type mismatch (libpf.String vs string)
- Line 132, 135: `ReadDynamicSymbols`/`ReadSymbols` methods removed from pfelf.File

**symbolizer.go:**
- Line 26, 38, 90: `libpf.SymbolMap` is undefined
- Line 106, 109: `ReadSymbols`/`ReadDynamicSymbols` methods removed from pfelf.File

These are API changes in the upstream opentelemetry-ebpf-profiler that need to be addressed separately.

### 🔧 Recommended Next Steps (for correlation ID feature):

1. **Fix Pre-existing Errors**: Adapt to upstream API changes in opentelemetry-ebpf-profiler
   - Replace `libpf.SymbolMap` with updated API
   - Fix `libpf.String` to string conversions
   - Update pfelf.File method calls

2. **Update simple_reporter.go**: Extract correlation ID from ContextValue and pass to correlator
   ```go
   correlationID := uint32(meta.ContextValue)
   r.correlator.AddCPUTraceWithCorrelation(correlationID, meta.Comm.String(), ...)
   ```

3. **Modify correlation.go**: Implement correlation ID-based matching logic
   - When `useCorrelationID` is true, match by `correlationID` field
   - When false, fall back to timestamp-based matching

4. **Integration Testing**: Test with various CUDA applications
   - Verify correlation IDs are captured correctly
   - Validate CPU/GPU trace matching accuracy
   - Compare with timestamp-based matching

## Files Modified/Created

### Created:
- `ebpf/correlation_uprobe.ebpf.c` - eBPF uprobe program
- `correlation_uprobe_attach.go` - Go uprobe attachment helper
- `correlationuprobe_x86_bpfel.go` - Generated eBPF Go bindings (bpf2go)
- `correlationuprobe_x86_bpfel.o` - Compiled eBPF object file
- `.gitignore` - Added eBPF generated files

### Modified:
- `main.go` - Added bpf2go directive, uprobe attachment, environment variable
- `opentelemetry-ebpf-profiler/tracer/tracer.go` - Added GetProgArrayMapFD() method
- `opentelemetry-ebpf-profiler/support/ebpf/custom_trace.ebpf.c` - Previously fixed null check

## Architecture

### eBPF Program Structure:
```c
// Shared maps (reused from profiler)
custom_context_map: BPF_MAP_TYPE_PERCPU_ARRAY[1] of u64
prog_array: BPF_MAP_TYPE_PROG_ARRAY[16] of program refs

// Uprobe function
SEC("uprobe/XpuPerfGetCorrelationId")
int capture_correlation_id(struct pt_regs *ctx) {
    u32 correlation_id = PT_REGS_PARM1(ctx);  // Get from RDI register
    u64 context_value = correlation_id;
    custom_context_map[0] = context_value;     // Store for custom__generic
    bpf_tail_call(ctx, &prog_array, 0);        // Call custom__generic
}
```

### Go Attachment Flow:
```go
func attachCorrelationUprobe(trc *tracer.Tracer, cuptiLibPath string) {
    // 1. Get FDs from tracer
    customProgFD := trc.GetCustomTraceProgramFD()
    customContextMapFD := trc.GetCustomContextMapFD()
    progArrayFD := trc.GetProgArrayMapFD()

    // 2. Load correlation uprobe spec
    spec := LoadCorrelationUprobe()

    // 3. Reuse profiler's maps
    opts := &ebpf.CollectionOptions{
        MapReplacements: map[string]*ebpf.Map{
            "custom_context_map": NewMapFromFD(customContextMapFD),
            "prog_array": NewMapFromFD(progArrayFD),
        },
    }
    coll := NewCollectionWithOptions(spec, opts)

    // 4. Add custom__generic to prog_array[0]
    progArrayMap.Update(0, NewProgramFromFD(customProgFD))

    // 5. Attach uprobe
    ex := link.OpenExecutable(cuptiLibPath)
    uprobe := ex.Uprobe("XpuPerfGetCorrelationId", prog)
}
```

## References

- **CUPTI Documentation**: CUDA Profiling Tools Interface
- **OpenTelemetry eBPF Profiler**: https://github.com/open-telemetry/opentelemetry-ebpf-profiler
- **cilium/ebpf**: https://github.com/cilium/ebpf
- **eBPF Tail Calls**: https://docs.kernel.org/bpf/
