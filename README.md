# SystemScope - Wall-Clock eBPF Profiler

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
[![Build and publish](https://github.com/yunwei37/systemscope/actions/workflows/publish.yml/badge.svg)](https://github.com/yunwei37/systemscope/actions/workflows/publish.yml)

SystemScope is a high-performance wall-clock eBPF profiler for Linux systems. It provides zero-instrumentation profiling with minimal overhead, outputting standard formats compatible with popular visualization tools.

## Features

- **Wall-clock profiling**: Track both on-CPU and off-CPU time
- **eBPF-based**: Zero instrumentation, minimal overhead
- **Standard formats**: Output to pprof, folded stacks, or Chrome trace format
- **Production ready**: Designed for continuous operation in production
- **Simple**: Single binary, no dependencies on target applications

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

# Run profiler
sudo ./build/src/bpf_profiler --duration 30 --output folded > profile.txt

# Visualize with external tools
flamegraph.pl profile.txt > flamegraph.svg
```

## Usage

### Basic Profiling

```bash
# Profile for 30 seconds, output folded stacks
sudo systemscope profile --duration 30 --output folded > stacks.txt

# Profile specific PIDs
sudo systemscope profile --pid 1234,5678 --duration 10

# Off-CPU profiling
sudo systemscope offcpu --duration 30 --output pprof -o offcpu.pb.gz
```

### Output Formats

SystemScope supports multiple output formats for compatibility with existing tools:

- **folded**: Brendan Gregg's folded stack format (for flamegraph.pl)
- **pprof**: Google's pprof format (for go tool pprof)
- **chrome**: Chrome trace format (for chrome://tracing)

### Integration with Visualization Tools

```bash
# With Brendan Gregg's FlameGraph
systemscope profile --output folded | flamegraph.pl > flame.svg

# With pprof web UI
systemscope profile --output pprof -o profile.pb.gz
go tool pprof -http=:8080 profile.pb.gz

# With speedscope
systemscope profile --output speedscope -o profile.json
# Open https://speedscope.app and load profile.json
```

## Architecture

SystemScope focuses on efficient data collection:

1. **eBPF Collectors**: Kernel-space programs for profiling
   - On-CPU sampling via perf events
   - Off-CPU time tracking
   - Minimal overhead design

2. **Data Processing**: User-space processing pipeline
   - Stack trace symbolization
   - Format conversion
   - Streaming output support

## Optional Visualization

A separate visualization package `systemscope-vis` is available in the `frontend/` directory for those who want a built-in web UI for viewing profile data. This is optional and SystemScope works perfectly with external tools like flamegraph.pl, pprof, or speedscope.

```bash
# Optional: Install visualization frontend
cd frontend
npm install
npm run dev
```

## Development

### Running Tests

```bash
make test
```

### Project Structure

```
systemscope/
├── src/
│   ├── collectors/     # eBPF profiling programs
│   ├── exporters/      # Output format converters
│   └── main.cpp        # CLI entry point
├── frontend/           # Optional visualization (systemscope-vis)
└── tests/             # Test suite
```

## Docker Support

```bash
docker run --rm -it --privileged ghcr.io/yunwei37/systemscope:latest \
    profile --duration 30 --output folded
```

## Performance

- Less than 1% CPU overhead during profiling
- Minimal memory footprint
- No impact when not profiling

## Contributing

Contributions are welcome! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

MIT License. See [LICENSE](LICENSE) file for details.

## Acknowledgments

SystemScope builds upon:
- [libbpf](https://github.com/libbpf/libbpf) for eBPF functionality
- [blazesym](https://github.com/libblazevm/blazesym) for symbolization