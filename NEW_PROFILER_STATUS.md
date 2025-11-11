# New Profiler Status Report

## Summary
The new profiler has been simplified to use only correlation ID-based matching between CPU traces and GPU kernels.

## Architecture Changes

###  Completed Simplifications:
1. **Removed cudaLaunchKernel uprobes** - No longer attaching to CUDA runtime library functions
2. **Single correlation method** - Only using correlation ID uprobe on `XpuPerfGetCorrelationId`
3. **Removed timestamp fallback** - Only exact correlation ID matching
4. **Simplified command-line interface** - Removed `-cpu-only`, `-gpu-only`, `-use-correlation-id` flags
5. **Always enabled** - Correlation uprobe is always attached and enabled

### Current Architecture:
- **CPU Sampling**: eBPF perf events collect regular CPU sampling traces
- **GPU Events**: CUPTI captures GPU kernel launches with correlation IDs via JSON pipe
- **Correlation Uprobe**: eBPF uprobe attached to `XpuPerfGetCorrelationId` in CUPTI library
- **Matching**: Events matched purely by correlation ID

## Current Status

### What Works:
✅ Build completes successfully  
✅ CUPTI library exports `XpuPerfGetCorrelationId` function
✅ CUPTI captures GPU kernel events with correlation IDs (29, 48, 164, etc.)
✅ CPU sampling traces are collected (2000+ samples)
✅ Correlation uprobe attaches without errors

### What Doesn't Work:
❌ Correlation uprobe doesn't capture correlation IDs (0 matched events)
❌ eBPF tail call from uprobe to `custom__generic` may be failing

## Possible Root Causes

1. **Program Type Incompatibility**: The uprobe (type: kprobe) may not be compatible with tail-calling to `custom__generic` (type: perf_event)
   
2. **Function Not Being Called**: `XpuPerfGetCorrelationId` might not be invoked frequently enough or at the right time

3. **eBPF Tail Call Failure**: Tail calls have restrictions and may silently fail

## Files Modified

### new-profiler/main.go
- Removed `cpuOnly`, `gpuOnly`, `useCorrelationUprobe`, `cudaLibPath` from Config
- Removed cudaLaunchKernel uprobe building logic
- Always attach correlation uprobe
- Always enable `CUPTI_ENABLE_CORRELATION_UPROBE=1`
- Simplified command-line flags

### new-profiler/correlation/strategy.go
- Removed timestamp-based matching logic
- Removed `unmatchedCPU` and `unmatchedGPU` buffers
- Removed `toleranceNs` configuration
- Simplified `Flush()` to only count unmatched events

### new-profiler/source/primitive/cupti/cupti_source.go
- Added debug logging for GPU kernel events

## Test Results

### CPU-Only Mode (Original Profiler):
```
Total samples: 22,086
Unique stacks: 34  
Output: Working ✅
```

### Simplified New Profiler:
```
CPU sampling traces: 2,536
GPU kernels captured: 14 (with correlation IDs)
Matched CPU+GPU: 0 ❌
Correlation IDs captured by uprobe: 0 ❌
```

## Next Steps

To fix the correlation uprobe:

1. **Investigate eBPF tail call compatibility**
   - Check if uprobe can tail-call to perf_event programs
   - May need to use a different approach (e.g., perf_event_output)

2. **Alternative approaches**:
   - Use `bpf_perf_event_output()` instead of tail call
   - Create a ring buffer for correlation IDs
   - Use a shared map that CPU sampling reads from

3. **Verify function is called**:
   - Use bpftrace to confirm `XpuPerfGetCorrelationId` is being invoked
   - Add logging to CUPTI library

## Comparison with Original Profiler

Both profilers share the same issue - neither successfully correlates CPU and GPU events yet. However:

- **Original**: Uses complex multi-mode architecture (cpu-only, gpu-only, merge)
- **New**: Simplified single-mode architecture, easier to debug

The new architecture is cleaner but needs the correlation uprobe mechanism fixed.
