# eBPF Profiler - Uprobe Example

This example demonstrates how to use uprobes with the opentelemetry-ebpf-profiler library to profile specific functions in executables.

## Overview

Uprobes (user-space probes) allow you to attach profiling to specific functions in user-space programs. This example shows how to:
- Attach uprobes to arbitrary executables and symbols
- Capture stack traces when those functions are called
- Use the public `tracer` API with uprobe support

## What are Uprobes?

Uprobes dynamically instrument user-space functions. When the instrumented function is called, the profiler captures a full stack trace showing:
- What called the function
- The complete call chain
- Mixed kernel/user space stack unwinding

## Building

```bash
cd /tmp/standalone-profiler/uprobe-example
go mod tidy
go build -o uprobe-profiler
```

## Running

```bash
# Attach to malloc in libc
sudo ./uprobe-profiler /lib/x86_64-linux-gnu/libc.so.6:malloc

# Attach to Python eval loop
sudo ./uprobe-profiler /usr/bin/python3.12:PyEval_EvalFrameDefault

# Attach to bash readline
sudo ./uprobe-profiler /bin/bash:readline

# Attach to multiple functions
sudo ./uprobe-profiler /lib/x86_64-linux-gnu/libc.so.6:malloc /lib/x86_64-linux-gnu/libc.so.6:free
```

## Usage

```
./uprobe-profiler <executable:symbol> [<executable:symbol>...]
```

Where:
- `executable` is the full path to the binary or shared library
- `symbol` is the function name to probe

## Example Output

```
=== UPROBE Trace #1 (Hash: 6269d7d2e7479684ec0205918c6467a8) ===
PID: 343693, TID: 343693, CPU: 23
Comm: node, Process: node
Executable: /home/yunwei37/.vscode-server/cli/servers/Stable-7d842fb85a0275a4a8e4d7e040d2625abbf7f084/server/node
Stack trace (17 frames):
  #0: malloc [libc.so.6+0xad650] (type: native)
  #1: _Znwm [libstdc++.so.6.0.33+0xbb903] (type: native)  // C++ operator new
  #2: <unknown> [node+0x1583de8] (type: native)
  ...
```

**Symbol Resolution:** This example includes native symbol resolution using the **official pfelf package** from opentelemetry-ebpf-profiler:

- Uses `libpf/pfelf.ReadDynamicSymbols()` and `libpf/pfelf.ReadSymbols()`
- Same infrastructure the profiler uses for Python, Ruby, Node.js symbolization
- Works with stripped binaries (reads `.dynsym` section)
- Efficient symbol lookup with `libpf.SymbolMap.LookupByAddress()`
- Symbols are resolved post-unwinding directly in the custom reporter

The profiler **already has** all the symbol resolution infrastructure built-in! We apply it to native frames directly using `libpf/pfelf`.

## Finding Symbols

To find available symbols in an executable:

```bash
# List all symbols
nm -D /lib/x86_64-linux-gnu/libc.so.6 | grep ' T '

# Search for specific functions
nm -D /usr/bin/python3.12 | grep PyEval

# For Go binaries
go tool nm /usr/bin/mygoapp | grep main
```

## Use Cases

- Profile memory allocation patterns (malloc/free)
- Trace database query execution
- Monitor file I/O operations
- Debug performance issues in specific functions
- Understand call patterns in third-party libraries

## Features

- Full stack trace capture on uprobe trigger
- Multi-language unwinding (C/C++, Go, Python, etc.)
- Symbol resolution without debug symbols
- Kernel + user space mixed stack traces
- Support for multiple concurrent uprobes

## Notes

- Requires root privileges to attach uprobes
- The profiled function must be in the symbol table
- Works with stripped binaries (using .eh_frame)
- Supports both executables and shared libraries
