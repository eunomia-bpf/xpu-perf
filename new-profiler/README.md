# New Multi-Source Profiler

A minimal implementation of the Multi-Source Profiling architecture described in `DESIGN_PROPOSAL.md`.

## Architecture

This implementation follows the **unified EventSource abstraction** principle:

```
EventSource Interface
├── Primitive Sources (data collectors)
│   ├── CUPTISource - GPU events from CUPTI
│   └── OTelSource - CPU traces from eBPF profiler
└── Composite Sources (operations on sources)
    └── CorrelateSource - combines two sources with correlation
```

### Key Design Features

1. **Everything is an EventSource** - both primitives and composites implement the same interface
2. **Composable** - sources can be nested and combined arbitrarily
3. **No layers** - just operations on sources
4. **Event-driven** - sources emit events through channels

## Directory Structure

```
new-profiler/
├── event/                      # Core event model
│   └── event.go               # Event, EventSource interface
├── source/
│   ├── primitive/             # Primitive sources (data collectors)
│   │   ├── cupti/            # CUPTI GPU events
│   │   │   └── cupti_source.go
│   │   └── otel/             # OTel CPU traces
│   │       └── otel_source.go
│   └── composite/             # Composite sources (operations)
│       └── correlate.go      # Combine sources with correlation
├── correlation/               # Correlation strategies
│   └── strategy.go           # CorrelationIDStrategy
├── output/                    # Output formatters
│   └── folded.go             # Flamegraph folded format
└── main.go                    # Orchestration
```

## Building

```bash
cd /root/xpu-perf/new-profiler
go build -o new-xpu-perf
```

## Usage

The profiler correlates CPU traces (from eBPF) with GPU kernel launches (from CUPTI) using correlation IDs.

### Prerequisites

1. **CUPTI trace injection library** - built from the old profiler
2. **Root access** - required for eBPF
3. **CUDA application** - target to profile

### Running

```bash
sudo ./new-xpu-perf \
  --cupti-pipe /tmp/cupti_trace.pipe \
  --output gpu_stacks.folded \
  --cuda-lib /usr/local/cuda/lib64/libcudart.so \
  python3 your_cuda_app.py
```

### Environment Variables

The CUPTI library path can be set via:
```bash
export CUPTI_LIB_PATH=/path/to/libcupti_trace_injection.so
```

## Output

The profiler generates a **folded stack format** suitable for flamegraph generation:

```
cpu_func1;cpu_func2;[GPU_Kernel]kernel_name 42
```

Generate flamegraph:
```bash
flamegraph.pl gpu_stacks.folded > flamegraph.svg
```

## How It Works

1. **CUPTISource** reads GPU events from a named pipe:
   - CUPTI library (injected via LD_PRELOAD) writes events to pipe
   - Each event has: kernel name, correlation ID, timestamp, duration

2. **OTelSource** collects CPU traces via eBPF:
   - Attaches uprobes to `cudaLaunchKernel` functions
   - Captures stack trace when kernel is launched
   - Extracts correlation ID from CUPTI callback

3. **CorrelateSource** matches events:
   - Uses `CorrelationIDStrategy` to match by correlation ID
   - Creates merged CPU+GPU stacks
   - Outputs to flamegraph format

## Example Pipeline

```go
// Create primitive sources
cupti := cupti.NewCUPTISource("/tmp/cupti_trace.pipe")
otel := otel.NewOTelSource(otelConfig)

// Create correlation strategy
strategy := correlation.NewCorrelationIDStrategy()

// Combine with correlation
correlated := composite.NewCorrelateSource(cupti, otel, strategy)

// correlated is an EventSource - can be composed further!
// For example: filtered := composite.NewFilterSource(correlated, predicate)
```

## Extending

### Adding a New Primitive Source

Implement the `EventSource` interface:

```go
type MySource struct{}

func (s *MySource) Name() string { return "mysource" }
func (s *MySource) Initialize() error { /* setup */ }
func (s *MySource) Start(ctx context.Context, events chan<- *event.Event) error {
    go func() {
        // Emit events
        events <- &event.Event{...}
    }()
    return nil
}
func (s *MySource) Stop() error { /* cleanup */ }
```

### Adding a New Composite Source

Same interface! Example - FilterSource:

```go
type FilterSource struct {
    source EventSource
    predicate func(*Event) bool
}

func (s *FilterSource) Start(ctx context.Context, events chan<- *Event) error {
    internal := make(chan *Event)
    s.source.Start(ctx, internal)
    go func() {
        for evt := range internal {
            if s.predicate(evt) {
                events <- evt
            }
        }
    }()
    return nil
}
```

Then use it:
```go
filtered := NewFilterSource(cuptiSource, func(e *Event) bool {
    return e.Type == EventGPUKernel
})
```

## Current Limitations

This is a **minimal implementation** focusing on:
- ✅ CUPTI + OTel sources
- ✅ Correlation by ID
- ✅ Flamegraph output
- ✅ Composable architecture

Not yet implemented:
- ❌ Filter, Enrich, Merge, Sample composites (architecture supports them!)
- ❌ Timestamp-based correlation
- ❌ PMU source
- ❌ Symbol resolution (shows file IDs instead of names)
- ❌ Comprehensive error handling

## Next Steps

To extend this implementation:

1. **Add composite sources** - implement Filter, Enrich, Merge, Sample
2. **Add PMU source** - collect hardware performance counters
3. **Improve symbol resolution** - show actual function names
4. **Add more correlation strategies** - timestamp-based, stream-based
5. **Better output formats** - JSON, protobuf, etc.

## Comparison with Old Profiler

| Feature | Old Profiler | New Profiler |
|---------|-------------|-------------|
| Architecture | Mixed, hard-coded | Composable sources |
| CUPTI+OTel | ✅ | ✅ |
| Correlation by ID | ✅ | ✅ |
| Extensibility | Hard to add sources | Easy - implement EventSource |
| Composability | ❌ | ✅ |
| Code size | ~3000 lines | ~800 lines (core) |
| Symbol resolution | ✅ (full) | ⚠️ (basic) |

The new profiler trades some features (like full symbol resolution) for a much cleaner, more extensible architecture.
