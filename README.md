# **SystemScope**

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
[![Build and publish](https://github.com/yunwei37/systemscope/actions/workflows/publish.yml/badge.svg)](https://github.com/yunwei37/systemscope/actions/workflows/publish.yml)

**SystemScope** is a unified real-time profiling, analysis, and optimization system for modern heterogeneous computing environments. It provides zero-instrumentation system profiling using eBPF technology with minimal overhead, offering real-time visualization and analysis capabilities through an interactive web frontend.

## **Vision**

SystemScope aims to revolutionize system observability by:
- **Real-time profiling** across CPU, GPU, and other accelerators without code modification
- **Zero-instrumentation** deployment with negligible overhead when not actively profiling
- **Multi-layer correlation** of events from hardware level to application level
- **Interactive visualization** through 3D flamegraphs and real-time dashboards
- **Automated optimization** discovery through cross-layer analysis

## **Features**

- **eBPF-based Profiling**: On-CPU and off-CPU profiling with minimal overhead
- **Real-time Streaming**: WebSocket-based live data streaming to frontend
- **3D Flamegraph Visualization**: Interactive Three.js-based visualization
- **Zero Configuration**: Single binary deployment with no code changes required
- **Production Ready**: Designed for continuous operation in production environments
- **Extensible Architecture**: Easy to add new collectors and visualization methods

## **Quick Start**

### **Prerequisites**

- Linux kernel 4.9+ with eBPF support
- CMake 3.16+
- C++20 compatible compiler
- Node.js 16+ (for frontend development)
- Root privileges (for eBPF programs)

### **Installation**

Install dependencies on Ubuntu:

```bash
make install
# or manually:
sudo apt-get install -y --no-install-recommends \
        libelf1 libelf-dev zlib1g-dev \
        make clang llvm
```

### **Build**

```bash
# Build the entire project
make build

# Or using CMake directly
cmake -B build
cmake --build build
```

### **Run**

Start the profiler (requires root):

```bash
sudo ./build/src/bpf_profiler
```

The web interface will be available at `http://localhost:8080`

### **Frontend Development**

```bash
cd frontend
npm install
npm run dev  # Start development server
```

## **Architecture**

SystemScope consists of three main components:

1. **eBPF Collectors**: Kernel-space programs that collect profiling data
   - On-CPU profiling using performance events
   - Off-CPU time tracking
   - Network and I/O event collection (planned)

2. **Profile Server**: User-space daemon that processes and serves data
   - Stack trace symbolization via blazesym
   - Real-time data aggregation
   - WebSocket server for frontend communication

3. **Web Frontend**: Interactive visualization interface
   - React + TypeScript + Three.js
   - Real-time 3D flamegraph rendering
   - Performance metrics dashboard

## **Development**

### **Running Tests**

```bash
# Run all tests
make test

# Run frontend tests
cd frontend
npm run test
```

### **Code Structure**

```
systemscope/
├── src/
│   ├── collectors/     # eBPF collectors
│   ├── server/         # WebSocket server
│   └── main.cpp        # Entry point
├── frontend/           # React frontend
├── tests/             # C++ tests
└── tools/             # Utility scripts
```

## **Docker Support**

Run SystemScope in a container:

```bash
docker run --rm -it --privileged ghcr.io/yunwei37/systemscope:latest
```

## **Roadmap**

- [ ] GPU profiling support (CUDA/ROCm)
- [ ] Distributed tracing integration
- [ ] Automated optimization recommendations
- [ ] Kubernetes operator for cluster-wide deployment
- [ ] Machine learning-based anomaly detection

## **Contributing**

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## **Documentation**

For detailed documentation, see:
- [Architecture Overview](documents/intro.md)
- [Development Guide](CLAUDE.md)
- [API Reference](docs/api.md)

## **License**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## **Acknowledgments**

SystemScope builds upon several excellent open-source projects:
- [libbpf](https://github.com/libbpf/libbpf) for eBPF functionality
- [blazesym](https://github.com/libblazevm/blazesym) for symbolization
- [React Three Fiber](https://github.com/pmndrs/react-three-fiber) for 3D visualization