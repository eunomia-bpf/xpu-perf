# BPF Profiler Server

A focused HTTP server for the BPF profiler that serves frontend files and provides API endpoints.

## Architecture

The server is organized into separate handlers:

- **API Handler** (`api_handler.hpp/cpp`) - Handles API endpoints like status
- **Frontend Handler** (`frontend_handler.hpp/cpp`) - Serves static frontend files from a specific directory
- **Config** (`config.hpp`) - Server configuration
- **Profile Server** (`profile_server.hpp/cpp`) - Main server orchestration

## Features

- Serves frontend files from a dedicated `frontend/` directory
- API endpoints for server status and future profiling endpoints
- CORS support for web integration
- Structured logging with spdlog
- SPA (Single Page Application) support with fallback routing

## Endpoints

### Frontend
- `GET /` - Serves `frontend/index.html`
- `GET /static/*` - Serves static files from `frontend/` directory
- `GET /*` - SPA fallback - serves `index.html` for unmatched routes (except `/api/*`)

### API
- `GET /api/status` - JSON status information

## Usage

### Starting the Server
```bash
# Build the project
make -j$(nproc)

# Start the server (listens on 0.0.0.0:8080)
./profiler server

# Start with debug logging
./profiler server -v
```

### Frontend Setup
Create a `frontend/` directory in your project root and place your frontend files there:
```
frontend/
├── index.html
├── static/
│   ├── css/
│   ├── js/
│   └── images/
└── ...
```

### Examples

#### Check Status
```bash
curl http://localhost:8080/api/status
```

#### Access Frontend
```bash
# Main page
curl http://localhost:8080/

# Static file
curl http://localhost:8080/static/css/main.css
```

## Configuration

Default configuration:
- Host: `0.0.0.0`
- Port: `8080`
- Frontend directory: `frontend`
- Log level: `info` (or `debug` with `-v`)
- CORS: enabled

## Dependencies

- [cpp-httplib](https://github.com/yhirose/cpp-httplib) - HTTP server library
- [nlohmann/json](https://github.com/nlohmann/json) - JSON processing
- [spdlog](https://github.com/gabime/spdlog) - Logging library 