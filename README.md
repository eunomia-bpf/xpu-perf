# SystemScope - Wall-Clock eBPF Profiler

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
[![Build and publish](https://github.com/yunwei37/systemscope/actions/workflows/publish.yml/badge.svg)](https://github.com/yunwei37/systemscope/actions/workflows/publish.yml)

SystemScope is a high-performance wall-clock eBPF profiler for Linux systems. It provides zero-instrumentation profiling with minimal overhead, capturing both on-CPU and off-CPU time to give a complete picture of application performance.

## Features

- **Wall-clock profiling**: Track both on-CPU and off-CPU time
- **Multiple analyzers**: On-CPU, off-CPU, and combined wall-clock analysis
- **eBPF-based**: Zero instrumentation, minimal overhead (~1% CPU)
- **Flexible output**: Generate flamegraphs and detailed performance reports
- **Process/thread filtering**: Target specific PIDs or TIDs
- **Simple deployment**: Single binary, no target application changes needed

## Quick Start

### Prerequisites

- Linux kernel 4.9+ with eBPF support
- CMake 3.16+
- C++20 compatible compiler
- Root privileges (for eBPF programs)

### Installation

```bash
# Install dependencies (Ubuntu/Debian)
make install

# Build
make build

# Run basic profiling
sudo ./build/profiler profile --duration 30

# View generated flamegraph
ls profile_profile_*
```

## Usage Guide

### Analyzer Types

SystemScope provides three analyzer types:

1. **profile** - On-CPU profiling using perf events
2. **offcputime** - Off-CPU time tracking (blocking, I/O wait, sleep)
3. **wallclock** - Combined on-CPU and off-CPU analysis

### Basic Commands

```bash
# On-CPU profiling for 10 seconds
sudo ./build/profiler profile --duration 10

# Off-CPU profiling with minimum 10ms blocking time
sudo ./build/profiler offcputime --duration 30 --min-block 10000

# Wall-clock analysis (combined on-CPU + off-CPU)
sudo ./build/profiler wallclock --duration 30

# Profile specific process
sudo ./build/profiler profile --pid 1234 --duration 10

# Profile multiple processes
sudo ./build/profiler profile --pid 1234,5678 --duration 10

# Profile specific threads
sudo ./build/profiler profile --tid 9012,9013 --duration 10
```

### Command-Line Options

```
Usage: profiler ANALYZER [options]

ANALYZER:
  profile       On-CPU profiling
  offcputime    Off-CPU time analysis
  wallclock     Combined wall-clock profiling
  server        Start HTTP server mode

Options:
  -d, --duration SECONDS         Duration to run (default: until interrupted)
  -p, --pid PID[,PID...]        Process IDs to profile
  -t, --tid TID[,TID...]        Thread IDs to profile
  -f, --frequency HZ            Sampling frequency (default: 49 Hz)
  -m, --min-block MICROSECONDS  Min blocking time for off-CPU (default: 1000)
  -c, --cpu CPU                 CPU to profile (default: all)
  -U, --user-stacks-only        Show only user-space stacks
  -K, --kernel-stacks-only      Show only kernel-space stacks
  --user-threads-only           Profile only user threads
  --kernel-threads-only         Profile only kernel threads
  -v, --verbose                 Enable verbose output
  --version                     Show version information
```

### Output Files

SystemScope generates flamegraph files in the current directory with the naming pattern:
```
{analyzer}_profile_[pid{PID}]_{timestamp}/
├── flame_cpu_{PID}.svg       # Interactive SVG flamegraph
├── flame_offcpu_{PID}.svg    # Off-CPU flamegraph (wallclock mode)
├── flame_wallclock_{PID}.svg # Combined flamegraph (wallclock mode)
└── stacks_{PID}.txt          # Raw stack traces
```

### Examples

#### Profile a Python Application
```bash
# Start your Python app
python myapp.py &
PID=$!

# Profile it
sudo ./build/profiler profile --pid $PID --duration 30

# View the flamegraph
firefox profile_profile_pid${PID}_*/flame_cpu_${PID}.svg
```

#### Analyze Database Performance
```bash
# Find MySQL process ID
PID=$(pgrep mysqld)

# Wall-clock analysis to see both CPU and I/O wait
sudo ./build/profiler wallclock --pid $PID --duration 60

# Check where time is spent
ls wallclock_profile_pid${PID}_*/
```

#### Debug High CPU Usage
```bash
# Quick system-wide CPU profile
sudo ./build/profiler profile --duration 5 --frequency 99

# Focus on user-space only
sudo ./build/profiler profile --duration 10 -U
```

#### Investigate Lock Contention
```bash
# Off-CPU analysis with 1ms minimum
sudo ./build/profiler offcputime --duration 30 --min-block 1000
```

## Server Mode

SystemScope can run as a server for continuous profiling:

```bash
# Start server (runs on port 8080)
sudo ./build/profiler server

# The server provides HTTP endpoints for real-time profiling
# Optional: Use with systemscope-vis frontend for visualization
```

## Architecture

SystemScope uses eBPF programs to collect profiling data with minimal overhead:

1. **eBPF Collectors**: Kernel-space programs for data collection
   - On-CPU: Samples via perf events at specified frequency
   - Off-CPU: Tracks blocking events and duration

2. **Data Processing**: User-space processing pipeline
   - Stack trace symbolization using blazesym
   - Flamegraph generation with aggregation

3. **Output Generation**: Multiple output formats
   - SVG flamegraphs for visualization
   - Text format for further processing

## Performance Impact

- **CPU overhead**: Less than 1% at default sampling rate (49 Hz)
- **Memory**: ~10-50 MB depending on stack depth and duration
- **Sampling rate**: Adjustable from 1-999 Hz

## Troubleshooting

### Permission Denied
```bash
# eBPF requires root or CAP_SYS_ADMIN capability
sudo setcap cap_sys_admin+ep ./build/profiler
```

### No Stack Traces
```bash
# Check if frame pointers are enabled
echo 1 | sudo tee /proc/sys/kernel/perf_event_paranoid

# For compiled languages, ensure -fno-omit-frame-pointer
```

### Missing Symbols
```bash
# Install debug symbols for better stack traces
sudo apt-get install linux-tools-common linux-tools-$(uname -r)
```

## Building from Source

```bash
# Clone the repository
git clone https://github.com/yunwei37/systemscope
cd systemscope

# Install dependencies
make install

# Build
make build

# Run tests
make test
```

## Optional Visualization

A separate visualization package `systemscope-vis` is available in the `frontend/` directory for those who want a built-in web UI for viewing profile data. This is optional and SystemScope works perfectly with external tools like flamegraph.pl, pprof, or speedscope.

```bash
# Optional: Install visualization frontend
cd frontend
npm install
npm run dev
```

## Contributing

Contributions are welcome! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

MIT License. See [LICENSE](LICENSE) file for details.

## Acknowledgments

SystemScope builds upon:
- [libbpf](https://github.com/libbpf/libbpf) for eBPF functionality
- [blazesym](https://github.com/libblazevm/blazesym) for symbolization
- Brendan Gregg's flamegraph visualization concepts