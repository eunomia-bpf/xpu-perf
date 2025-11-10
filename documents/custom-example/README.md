# Custom Trace Example - eBPF Tail Call Integration

This example demonstrates how to use **custom traces** with the OpenTelemetry eBPF profiler. Custom traces allow external eBPF programs to trigger trace collection with context-specific metadata by tail-calling into the profiler's `custom__generic` program.

## Overview

This proof-of-concept shows how to:
- Load the profiler's `custom__generic` eBPF program for tail-calling
- Share the `custom_context_map` between your eBPF program and the profiler
- Attach your own uprobe that reads context data and tail-calls to `custom__generic`
- Propagate context values (like correlation IDs) through the trace collection pipeline
- Receive and process custom traces with the context metadata

## Architecture

### Flow Diagram
```
┌─────────────────────────────────────────────────────────────────┐
│ Your eBPF Program (example_tailcall.c)                          │
│                                                                  │
│ 1. Uprobe triggers on target function                           │
│ 2. Read context_id from function parameter (RDI register)       │
│ 3. Store context_id in custom_context_map[0]                    │
│ 4. Tail call to custom__generic (index 0 in prog_array)         │
└─────────────────────────────────────────────────────────────────┘
                            │
                            │ bpf_tail_call()
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ Profiler's custom__generic (custom_trace.ebpf.c)                │
│                                                                  │
│ 5. Retrieve context_id from custom_context_map[0]               │
│ 6. Call collect_trace() with TRACE_CUSTOM origin                │
│ 7. Pass context_id as last parameter                            │
└─────────────────────────────────────────────────────────────────┘
                            │
                            │ Trace collection
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ Profiler Trace Pipeline                                         │
│                                                                  │
│ 8. Unwind stack, build trace                                    │
│ 9. Store context_id in Trace.Meta.OffTime field                 │
│ 10. Report trace with TRACE_CUSTOM origin                       │
└─────────────────────────────────────────────────────────────────┘
                            │
                            │ User-space processing
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ Your Reporter (simple_reporter.go)                              │
│                                                                  │
│ 11. Receive trace with origin == TraceOriginCustom              │
│ 12. Extract context_id from Meta.OffTime                        │
│ 13. Process trace with context correlation                      │
└─────────────────────────────────────────────────────────────────┘
```

## Building

Use the provided Makefile for easy building:

```bash
# Build everything (profiler + test program)
make

# Build only the profiler
make custom-example

# Build only the test program
make test_program

# Clean all generated files
make clean

# Show help
make help
```

### Build Targets

- `make all` (default) - Builds both `custom-example` and `test/test_program`
- `make generate` - Generates `vmlinux.h` and eBPF Go bindings
- `make custom-example` - Builds the profiler binary
- `make test_program` - Builds the test program
- `make run` - Builds and runs the profiler (requires root)
- `make clean` - Removes all generated files

## Running

### Quick Start

```bash
# Build everything
make

# Run the profiler (in one terminal)
sudo ./custom-example ./test/test_program:report_trace

# Run the test program (in another terminal)
./test/test_program
```

### Using make run

```bash
# Start the profiler and wait for test program
sudo make run

# Then in another terminal:
./test/test_program
```

## Usage

```
sudo ./custom-example <executable:function>
```

**Parameters:**
- `executable` - Path to the executable to profile
- `function` - Function name to attach uprobe to

**Example:**
```bash
sudo ./custom-example ./test/test_program:report_trace
```

## How It Works

### 1. Profiler Setup

The profiler (`main.go`):
- Loads the OpenTelemetry eBPF profiler with `custom__generic` program
- Exposes two critical FDs via API:
  - `GetCustomTraceProgramFD()` - FD of the `custom__generic` program
  - `GetCustomContextMapFD()` - FD of the shared `custom_context_map`

### 2. External eBPF Program

Your eBPF program (`example_tailcall.c`):
- Defines a `prog_array` map for tail calling
- Reuses the profiler's `custom_context_map` via map replacement
- Uprobe reads context data from function parameters
- Stores context_id in `custom_context_map[0]`
- Tail calls to `custom__generic` at `prog_array[0]`

### 3. Map Sharing

The key to this architecture is **map reuse**:

```go
// Get the profiler's custom_context_map FD
customContextMapFD := trc.GetCustomContextMapFD()

// Create a map object from the FD
customContextMap, err := ebpf.NewMapFromFD(customContextMapFD)

// Load your eBPF program with map replacement
opts := &ebpf.CollectionOptions{
    MapReplacements: map[string]*ebpf.Map{
        "custom_context_map": customContextMap,  // Reuse profiler's map
    },
}
spec.LoadAndAssign(objs, opts)
```

### 4. Tail Call Setup

```go
// Get custom__generic program FD
customProgFD := trc.GetCustomTraceProgramFD()

// Add to your prog_array at index 0
objs.ProgArray.Put(uint32(0), uint32(customProgFD))
```

### 5. Context Propagation

The `context_value` flows through the system:
- **eBPF side:** Stored in `custom_context_map` (per-CPU array)
- **Kernel side:** Passed to `collect_trace()` as last parameter
- **User-space side:** Retrieved from `Trace.Meta.OffTime` field

Note: The `OffTime` field is reused for `context_value` when `Origin == TRACE_CUSTOM`, similar to how it stores off-CPU time for `TRACE_OFF_CPU`.

## Test Program

The included test program (`test/test_program.c`) demonstrates:

```c
// Function that reports traces with context ID
void report_trace(uint64_t context_id) {
    // Uprobe attached here will:
    // 1. Read context_id from RDI register
    // 2. Store in custom_context_map
    // 3. Tail call to custom__generic
}

int main() {
    for (int i = 0; i < 5; i++) {
        report_trace(0x1000 + i);  // Context IDs: 0x1000-0x1004
    }
}
```

## Example Output

When running, you'll see traces with context values:

```
=== CUSTOM Trace #1 (Hash: ee0b2293145c9872dc24735c18f7ed8e) ===
Context Value: 4096 (0x1000)
PID: 307926, TID: 307926, CPU: 15
Stack trace (5 frames):
  #0: report_trace [test_program+0x1169]
  #1: main [test_program+0x11e6]
  #2: __libc_start_call_main [libc.so.6+0x2a1ca]
  ...

=== CUSTOM Trace #2 (Hash: ee0b2293145c9872dc24735c18f7ed8e) ===
Context Value: 4097 (0x1001)
PID: 307926, TID: 307926, CPU: 15
Stack trace (5 frames):
  ...
```

## Debugging Process

### Step 1: Verify Profiler Loading

Look for these log messages:
```
INFO Loaded custom__generic program (FD: 73)
INFO Got custom__generic program FD: 73
INFO Got custom_context_map FD: 23
```

If `custom__generic` fails to load, check:
- Kernel version (needs BPF support)
- BPF verifier logs (increase `BPFVerifierLogLevel`)

### Step 2: Verify Map Reuse

```
INFO Successfully reused custom_context_map from profiler
```

If this fails:
- Check that `MapReplacements` is set correctly
- Verify FD is valid and not closed
- Ensure map definitions match (key/value types)

### Step 3: Verify Tail Call Setup

```
INFO Successfully added custom__generic to prog_array at index 0
```

If this fails:
- Ensure both programs are same type (uprobe/uprobe)
- Check that prog_array is defined correctly
- Verify FD is still valid

### Step 4: Verify Uprobe Attachment

```
INFO Successfully attached uprobe to ./test/test_program:report_trace
```

If this fails:
- Check that executable path is correct
- Verify function symbol exists: `nm test/test_program | grep report_trace`
- Ensure executable is not stripped of symbols

### Step 5: Run Test and Check Traces

```bash
./test/test_program
```

Look for:
- `Traces - Total: X, Custom: Y, Sampling: Z` - Custom count should increase
- Custom trace output with context values

### Common Issues

#### No Custom Traces Received

**Debug steps:**
1. Check if uprobe is actually triggering:
   ```bash
   sudo cat /sys/kernel/debug/tracing/uprobe_events
   ```

2. Verify tail call is working:
   ```bash
   sudo bpftool prog show  # Look for custom__generic
   sudo bpftool map show   # Look for prog_array
   ```

3. Check BPF verifier logs:
   - Increase `BPFVerifierLogLevel` to 1 or 2 in `main.go`
   - Look for tail call verification errors

#### Context Value is 0 or Wrong

**Debug steps:**
1. Verify context_id is being read from correct register
   - x86_64: First parameter is in RDI (`ctx->di`)
   - Check your architecture's calling convention

2. Check map update is succeeding:
   ```c
   int ret = bpf_map_update_elem(&custom_context_map, &key, &context_id, BPF_ANY);
   // Add error checking
   ```

3. Verify map key is correct (should be 0)

#### Program Type Mismatch

Error: `invalid argument` when adding to prog_array

**Fix:** Ensure both programs use same section type:
```c
SEC("uprobe/custom__generic")  // Profiler program
SEC("uprobe/example_function") // Your program
```

### Useful Debugging Commands

```bash
# List loaded BPF programs
sudo bpftool prog list

# Show BPF maps
sudo bpftool map list

# Dump prog_array contents
sudo bpftool map dump id <map_id>

# Show uprobe events
sudo cat /sys/kernel/debug/tracing/uprobe_events

# Check if profiler is running
ps aux | grep custom-example

# View recent kernel logs
sudo dmesg | tail -50
```

## Technical Details

### Map Definitions

**custom_context_map** (defined in profiler's `custom_trace.ebpf.c`):
```c
struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __type(key, u32);
    __type(value, u64);
    __uint(max_entries, 1);
} custom_context_map SEC(".maps");
```

**prog_array** (defined in your `example_tailcall.c`):
```c
struct {
    __uint(type, BPF_MAP_TYPE_PROG_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, __u32);
} prog_array SEC(".maps");
```

### Trace Origins

The profiler supports multiple trace origins:
- `TRACE_SAMPLING` (0) - Regular sampling-based traces
- `TRACE_UPROBE` (1) - Explicit uprobe traces (deprecated)
- `TRACE_OFF_CPU` (2) - Off-CPU scheduling traces
- `TRACE_CUSTOM` (3) - Custom traces with context propagation

### Context Value Field

The `Trace.Meta.OffTime` field is multipurpose:
- For `TRACE_OFF_CPU`: Stores off-CPU duration in nanoseconds
- For `TRACE_CUSTOM`: Stores context_value (e.g., correlation ID)

## Use Cases

This custom trace mechanism enables:

1. **Correlation ID Tracking**
   - Propagate request IDs through distributed traces
   - Link eBPF traces to application-level spans

2. **Custom Context Profiling**
   - Track GPU operation IDs with CPU stack traces
   - Correlate CUDA API calls with kernel launches

3. **Event Correlation**
   - Link system calls to application events
   - Connect database queries to stack traces

4. **Performance Attribution**
   - Tag traces with user IDs, tenant IDs
   - Group traces by transaction type

## Files

- `main.go` - Profiler entry point, demonstrates API usage
- `simple_reporter.go` - Custom trace reporter implementation
- `example_tailcall.c` - Example eBPF program with tail call
- `test/test_program.c` - Test program that triggers custom traces
- `Makefile` - Build system
- `.gitignore` - Excludes generated binaries and eBPF objects

## Requirements

- Linux kernel with eBPF support (4.15+)
- Root privileges (for attaching eBPF programs)
- Go 1.21+ (for building)
- `bpftool` (for generating vmlinux.h)
- GCC (for building test program)

## Performance Notes

- Per-CPU maps avoid contention
- Tail calls are very efficient (~10ns overhead)
- No additional syscalls needed
- Context propagation is zero-copy

## License

Same as OpenTelemetry eBPF Profiler (Apache 2.0)
