# Go eBPF Uprobe Example

A minimal example demonstrating how to write eBPF programs in Go using the Cilium eBPF library. This example includes:
- A simple C++ program with USDT probes
- A Go eBPF program that attaches uprobes to monitor function calls

## What it does

- **example.cpp**: A simple C++ program that runs in a loop, calling `add_numbers()` function with USDT probes
- **counter.c**: eBPF program with uprobe handlers
- **main.go**: Go program that attaches uprobes to the example program and monitors function calls

## Prerequisites

- Linux kernel with eBPF support (4.x or later)
- Go 1.19 or later
- Root/sudo privileges (required for loading eBPF programs)
- Clang/LLVM (for compiling eBPF C code)
- systemtap-sdt-dev (for USDT probe support)
- Kernel headers installed

### Install dependencies on Ubuntu/Debian:

```bash
sudo apt-get install -y clang llvm libbpf-dev linux-headers-$(uname -r) systemtap-sdt-dev g++
```

## Building

Use the provided Makefile:

```bash
# Build everything (C++ example + Go eBPF program)
make all

# Or build separately
make example  # Build C++ example program
make ebpf     # Build Go eBPF program
```

Manual build:
```bash
# Build the C++ example
g++ -o example example.cpp

# Build the Go eBPF program
go generate
go build -o ebpf-demo
```

## Running

**Terminal 1** - Run the example program:
```bash
./example
```

Output:
```
Starting example program with USDT probes...
PID: 12345
Running in a loop, press Ctrl+C to exit

Iteration 1: add_numbers(1, 2) = 3
Iteration 2: add_numbers(2, 4) = 6
...
```

**Terminal 2** - Run the eBPF monitor (requires sudo):
```bash
sudo ./ebpf-demo
```

Output:
```
Successfully attached uprobe to add_numbers function in ./example
Monitoring function calls...
Press Ctrl+C to exit

=== Uprobe Stats ===
PID 12345: 10 calls

=== USDT Stats ===
No activity yet...
```

The uprobe successfully captures calls to the `add_numbers()` function!

## How it works

1. **counter.c** - eBPF program written in C that:
   - Defines a hash map to store function call counts per PID
   - Implements the uprobe handler that increments the counter

2. **main.go** - Go program that:
   - Loads the compiled eBPF program into the kernel
   - Attaches the uprobe to bash's readline function
   - Reads and displays the statistics from the eBPF map

3. **bpf2go** - Code generation tool that:
   - Compiles the C code to eBPF bytecode
   - Generates Go bindings for loading and interacting with the eBPF program

## Files

- `counter.c` - eBPF program in C
- `main.go` - Go loader and monitor
- `counter_x86_bpfel.go` - Auto-generated Go bindings (created by `go generate`)
- `counter_x86_bpfel.o` - Compiled eBPF bytecode (created by `go generate`)

## Cleanup

Stop the program with Ctrl+C. The eBPF program is automatically detached when the process exits.

To clean generated files:
```bash
rm -f ebpf-demo counter_x86_bpfel.go counter_x86_bpfel.o
```

## References

- [Cilium eBPF Library](https://github.com/cilium/ebpf)
- [eBPF Documentation](https://ebpf.io/)
- [Getting Started Guide](https://ebpf-go.dev/guides/getting-started/)
