# systemscope-vis

Profile data visualization for SystemScope profiler.

## Overview

`systemscope-vis` is an optional web-based visualization frontend for SystemScope profiler. It provides a modern React-based interface for viewing and analyzing profiling data.

## Features

- Profile data viewer with metadata display
- Real-time data streaming via WebSocket
- Multiple view modes (profile view, data table)
- Data export capabilities

## Installation

```bash
npm install
```

## Development

```bash
# Start development server
npm run dev

# Build for production
npm run build

# Run tests
npm run test
```

## Usage

This frontend is designed to work with SystemScope's server mode:

```bash
# Start SystemScope server
sudo systemscope server --port 8080

# In another terminal, start the frontend
npm run dev
```

The visualization will be available at `http://localhost:5173`

## Standalone Usage

You can also use this as a standalone flamegraph viewer by loading profile data files directly through the UI.

## Note

SystemScope works perfectly with external visualization tools like flamegraph.pl, pprof, or speedscope. This frontend is optional and provided for users who prefer an integrated solution.

## License

MIT