# BPF Profiler Server

A simple HTTP server component for the BPF profiler that serves files and provides basic status information.

## Features

- Simple file browser at root endpoint (/)
- Basic status API endpoint (/api/status)
- Static file serving with proper MIME types
- CORS support for web integration

## Endpoints

### File Browser
- `GET /` - HTML file browser showing current directory files

### API
- `GET /api/status` - JSON status information

### File Serving
- `GET /files/{filename}` - Serve specific files from current directory

## Usage

### Starting the Server
```bash
# Build the project
make -j$(nproc)

# Start the server (listens on 0.0.0.0:8080)
./profiler server
```

### Examples

#### Access File Browser
```bash
curl http://localhost:8080/
```

#### Check Status
```bash
curl http://localhost:8080/api/status
```

#### Download a File
```bash
curl http://localhost:8080/files/example.html
```

## Configuration

The server runs on `0.0.0.0:8080` by default and serves files from the current working directory.

## Dependencies

- [cpp-httplib](https://github.com/yhirose/cpp-httplib) - HTTP server library
- [nlohmann/json](https://github.com/nlohmann/json) - JSON processing 