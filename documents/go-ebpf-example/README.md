# Go eBPF Uprobe Example

A minimal example demonstrating how to write eBPF programs in Go using the Cilium eBPF library. This example uses a uprobe to monitor the `readline` function in bash.

## What it does

- Attaches a **uprobe** to the `readline` function in `/usr/bin/bash`
- Tracks the number of readline calls per process ID
- Displays real-time statistics every 2 seconds

## Prerequisites

- Linux kernel with eBPF support (4.x or later)
- Go 1.19 or later
- Root/sudo privileges (required for loading eBPF programs)
- Clang/LLVM (for compiling eBPF C code)
- Kernel headers installed

### Install dependencies on Ubuntu/Debian:

```bash
sudo apt-get install -y clang llvm libbpf-dev linux-headers-$(uname -r)
```

## Building

1. Install Go dependencies:
```bash
go mod tidy
```

2. Generate eBPF code from C:
```bash
go generate
```

3. Build the program:
```bash
go build
```

Or do all steps at once:
```bash
go generate && go build
```

## Running

Run with sudo (eBPF programs require root privileges):

```bash
sudo ./ebpf-demo
```

You should see output like:
```
Successfully attached uprobe to bash readline function
Monitoring function calls per process...
Open a bash shell and run commands to see activity
Press Ctrl+C to exit

=== Function Call Stats ===
No activity yet...
```

## Testing

To see the uprobe in action:

1. Keep the program running in one terminal
2. Open a new bash shell in another terminal
3. Run some commands (each command triggers readline)
4. Watch the first terminal display the PID and call counts

Example output:
```
=== Function Call Stats ===
PID 12345: 5 calls
PID 67890: 3 calls
```

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
