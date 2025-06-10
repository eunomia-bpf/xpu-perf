# BPF Profiler Server

This server component provides an HTTP API interface for the BPF profiler, allowing remote control and access to profiling results.

## Features

- Start/stop profiling via HTTP API
- Support for all analyzer types (profile, offcputime, wallclock)
- Real-time status monitoring
- Result file serving and listing
- JSON API for easy integration

## Endpoints

### Server Info
- `GET /` - Server information and status
- `GET /health` - Health check

### Profiling Control
- `POST /profile/start` - Start profiling session
- `POST /profile/stop` - Stop current profiling session
- `GET /profile/status` - Get current profiling status

### Results
- `GET /results` - List all available result directories
- `GET /results/{dir}/{file}` - Serve specific result file (HTML, SVG, JSON, etc.)

## Usage

### Starting the Server
```bash
# Build the project
make -j$(nproc)

# Start the server (listens on 0.0.0.0:8080)
./profiler server
```

### API Examples

#### Start Profiling
```bash
curl -X POST http://localhost:8080/profile/start \
  -H "Content-Type: application/json" \
  -d '{
    "analyzer": "profile",
    "duration": 30,
    "frequency": 49,
    "pids": [1234, 5678]
  }'
```

#### Check Status
```bash
curl http://localhost:8080/profile/status
```

#### Stop Profiling
```bash
curl -X POST http://localhost:8080/profile/stop
```

#### List Results
```bash
curl http://localhost:8080/results
```

#### Get Flamegraph
```bash
curl http://localhost:8080/results/profile_profile_1234567890/flamegraph.html
```

## Configuration

The server runs on `0.0.0.0:8080` by default. All profiling output is saved to the current working directory.

## Dependencies

- [cpp-httplib](https://github.com/yhirose/cpp-httplib) - HTTP server library
- [nlohmann/json](https://github.com/nlohmann/json) - JSON processing
- Same BPF profiler dependencies as the main application 