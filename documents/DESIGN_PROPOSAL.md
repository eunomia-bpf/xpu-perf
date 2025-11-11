# Multi-Source Profiling Architecture Design

## Design Goals

### Primary Goals
1. **Unified Source Abstraction**: Everything is an EventSource
   - Primitive sources: CUPTI, OTel, PMU (collect raw data)
   - Composite sources: Filter, Merge, Correlate (transform/combine sources)
   - **Same interface for both** - no distinction at the abstraction level

2. **Source Composability**: Build complex pipelines by composing simple sources
   - Single source transformation (filter, enrich, sample)
   - Multi-source combination (merge, correlate)
   - Sources are first-class composable values

3. **Extensibility Without Modification**:
   - Add new data sources without touching existing code
   - Add new correlation strategies as plugins
   - Add new event transformations as composable functions

4. **Declarative Configuration**:
   - Build complex pipelines through composition, not procedural logic
   - Modes are configurations, not global state flags
   - Clear separation: what to collect vs how to correlate

5. **Clean Abstractions**:
   - EventSource interface for all data providers
   - Common Event model across all sources
   - Correlation strategies as pluggable algorithms

### Non-Goals
- **Not a generic stream processing framework**: Purpose-built for profiling
- **Not zero-copy**: Favor clarity over micro-optimization
- **Not backwards compatible**: Clean break to establish better patterns

---

## Problem Statement

Currently xpu-perf has **two independent data sources** that need correlation:

1. **CUPTI Tracer**: GPU events (kernels, memcpy, etc.) from NVIDIA CUPTI
2. **OTel Profiler**: CPU traces from OpenTelemetry eBPF profiler

Future expansion will add more sources:
- **PMU counters**: Direct hardware performance counter access
- **Custom instrumentation**: User-defined trace points
- **System metrics**: I/O, network, power, etc.

### Current Architecture Challenge

The challenge is **not about layers**, but about **operations on sources**:

1. **Primitive sources** (data collectors):
   ```
   CUPTI → GPU events
   OTel  → CPU traces (with different collection modes)
   PMU   → Hardware counters
   ```

2. **Operations needed**:
   ```
   - Filter(source, predicate)           → filtered source
   - Enrich(source, enricher)            → enriched source
   - Sample(source, rate)                → sampled source
   - Merge(source1, source2, ...)        → merged source
   - Correlate(source1, source2, strategy) → correlated source
   ```

3. **Result**: Complex pipelines built from simple operations
   ```
   Correlate(
       Enrich(OTel, addContext),
       Filter(CUPTI, onlyKernels),
       CorrelationIDStrategy
   ) → Correlated CPU+GPU source
   ```

The key insight: **OTel collection modes** (sampling, uprobes, correlation capturers) are **internal implementation details** of the OTel source, not a separate abstraction layer. They're configured via `OTelCollectionMode`, but from the outside, OTel is just another EventSource.

### Current Design Problems

1. **Tight coupling**: CUPTI and OTel logic mixed in main.go
2. **Hard-coded correlation**: TraceCorrelator assumes CUPTI + OTel only
3. **No source abstraction**: Each source has different APIs (pipes, channels, etc.)
4. **Mode confusion**: CPU-only, GPU-only, merge mode flags are confusing
5. **Not extensible**: Adding PMU requires rewriting correlation logic

---

## Proposed Architecture: Multi-Source Event Framework

### Design Principles

1. **Source independence**: Each source is a standalone component
2. **Common event model**: All sources emit standardized events
3. **Pluggable correlation**: Correlation strategies are composable
4. **Mode as configuration**: Modes are source configurations, not global flags
5. **Lazy initialization**: Sources only initialize when needed

---

## Architecture Diagram

### Primitive Sources (Data Collectors)
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ CUPTI       │    │ OTel        │    │ PMU         │
│ Source      │    │ Source      │    │ Source      │
├─────────────┤    ├─────────────┤    ├─────────────┤
│ GPU Events  │    │ CPU Traces  │    │ HW Counters │
│ - Kernels   │    │ - Sampling  │    │ - Cycles    │
│ - Memcpy    │    │ - Uprobes   │    │ - Cache     │
└─────────────┘    └─────────────┘    └─────────────┘
       │                  │                   │
       └──────────────────┴───────────────────┘
                          │
                     EventSource
                     interface
```

### Composite Sources (Operations on Sources)
```
┌──────────────────────────────────────────────────────────┐
│                     Operations                           │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Filter(source) ──────────► Filtered EventSource        │
│  Enrich(source) ──────────► Enriched EventSource        │
│  Sample(source) ──────────► Sampled EventSource         │
│  Merge(s1, s2, ...) ──────► Merged EventSource          │
│  Correlate(s1, s2) ───────► Correlated EventSource      │
│                                                          │
└──────────────────────────────────────────────────────────┘
       │
       │  All return EventSource
       └───────────────────────────────────────►
                                                │
                                          EventSource
                                          interface
                                          (same as primitives!)
```

### Example Pipeline
```
┌─────────┐   Filter    ┌──────────────┐
│ CUPTI   │───────────► │ Only Kernels │
└─────────┘             └──────┬───────┘
                               │
┌─────────┐   Enrich    ┌──────▼───────┐   Correlate   ┌─────────────┐
│ OTel    │───────────► │ Add Context  │──────────────►│ CPU+GPU     │
└─────────┘             └──────────────┘               │ Correlated  │
                                                        └─────────────┘
                                                               │
                                                          EventSource
                                                          (ready to use)
```

---

## Event Source Abstraction

All sources (primitive and composite) implement the same interface:

### Common Event Model

All sources emit events with common metadata:

```go
// Event is the universal container for profiling data
type Event struct {
    // Common metadata
    Timestamp   int64           // Nanoseconds since epoch
    Source      string          // "cupti", "otel", "pmu"
    Type        EventType       // CPU_TRACE, GPU_KERNEL, PMU_COUNTER, etc.

    // Correlation metadata
    CorrelationID uint32        // For cross-source correlation (0 if N/A)
    ThreadID      uint32        // OS thread ID
    ProcessID     uint32        // OS process ID

    // Event-specific data
    Data        EventData       // Type-specific payload
}

type EventType int
const (
    EventCPUTrace EventType = iota
    EventGPUKernel
    EventGPUMemcpy
    EventPMUCounter
    EventCustom
)

// EventData is a union-like interface
type EventData interface {
    Type() EventType
}

// CPU trace event data
type CPUTraceData struct {
    Stack       []string
    CPU         int
    Comm        string
    IsUprobe    bool
}

// GPU kernel event data
type GPUKernelData struct {
    KernelName  string
    Duration    int64
    StreamID    uint32
    DeviceID    int
}

// PMU counter event data
type PMUCounterData struct {
    CounterName string
    Value       uint64
}
```

### EventSource Interface

```go
// EventSource represents any source of profiling events
// Sources can be:
// - Primitive sources (CUPTI, OTel, PMU) that collect raw data
// - Composite sources that transform or combine other sources
type EventSource interface {
    // Name returns the unique identifier for this source
    Name() string

    // Initialize prepares the source for data collection
    Initialize() error

    // Start begins event collection
    // Events are sent to the provided channel
    Start(ctx context.Context, events chan<- *Event) error

    // Stop halts event collection and cleanup
    Stop() error

    // GetCapabilities returns what this source can provide
    GetCapabilities() SourceCapabilities
}

type SourceCapabilities struct {
    ProvidesCorrelationID bool
    ProvidesStackTraces   bool
    ProvidesTimestamp     bool
    SupportedEventTypes   []EventType
}
```

---

## Primitive Source Implementations

### CUPTI Source

```go
// CUPTISource wraps the CUPTI tracer
type CUPTISource struct {
    parser          *CUPTIParser
    pipePath        string
    enableGraph     bool
    enableCorrelation bool
}

func NewCUPTISource(cuptiLibPath string, enableCorrelation bool) *CUPTISource {
    return &CUPTISource{
        enableCorrelation: enableCorrelation,
    }
}

func (s *CUPTISource) Name() string {
    return "cupti"
}

func (s *CUPTISource) GetCapabilities() SourceCapabilities {
    return SourceCapabilities{
        ProvidesCorrelationID: s.enableCorrelation,
        ProvidesStackTraces:   false,
        ProvidesTimestamp:     true,
        SupportedEventTypes:   []EventType{
            EventGPUKernel,
            EventGPUMemcpy,
        },
    }
}

func (s *CUPTISource) Start(ctx context.Context, events chan<- *Event) error {
    // Start CUPTI parser, read from pipe
    go func() {
        for {
            select {
            case <-ctx.Done():
                return
            default:
                // Read CUPTI event from pipe
                cuptiEvent := s.parser.ReadEvent()

                // Convert to common Event format
                event := &Event{
                    Timestamp:     cuptiEvent.Timestamp,
                    Source:        "cupti",
                    Type:          EventGPUKernel,
                    CorrelationID: cuptiEvent.CorrelationID,
                    Data: &GPUKernelData{
                        KernelName: cuptiEvent.KernelName,
                        Duration:   cuptiEvent.Duration,
                    },
                }

                events <- event
            }
        }
    }()
    return nil
}
```

### OTel Source

```go
// OTelSource wraps the OpenTelemetry eBPF profiler
type OTelSource struct {
    tracer      *tracer.Tracer
    mode        OTelCollectionMode
}

type OTelCollectionMode struct {
    EnableSampling           bool
    EnableUprobeKernelLaunch bool
    EnableUprobeCorrelation  bool
    SamplingFrequencyHz      int
}

func NewOTelSource(mode OTelCollectionMode) *OTelSource {
    return &OTelSource{mode: mode}
}

func (s *OTelSource) Name() string {
    return "otel"
}

func (s *OTelSource) GetCapabilities() SourceCapabilities {
    return SourceCapabilities{
        ProvidesCorrelationID: s.mode.EnableUprobeCorrelation,
        ProvidesStackTraces:   true,
        ProvidesTimestamp:     true,
        SupportedEventTypes:   []EventType{EventCPUTrace},
    }
}

func (s *OTelSource) Initialize() error {
    // Setup tracer based on mode
    cfg := tracer.Config{
        Sampling:    s.mode.EnableSampling,
        UProbes:     s.buildUprobeList(),
        // ... other config
    }

    var err error
    s.tracer, err = tracer.NewTracer(context.Background(), cfg)
    return err
}

func (s *OTelSource) Start(ctx context.Context, events chan<- *Event) error {
    // Start tracer, convert traces to events
    traceCh := make(chan *host.Trace)
    s.tracer.StartMapMonitors(ctx, traceCh)

    go func() {
        for {
            select {
            case <-ctx.Done():
                return
            case trace := <-traceCh:
                // Convert OTel trace to common Event format
                event := &Event{
                    Timestamp:     int64(trace.Meta.Timestamp),
                    Source:        "otel",
                    Type:          EventCPUTrace,
                    CorrelationID: uint32(trace.Meta.ContextValue),
                    ThreadID:      trace.Meta.TID,
                    ProcessID:     trace.Meta.PID,
                    Data: &CPUTraceData{
                        Stack:    ExtractStack(trace),
                        CPU:      int(trace.Meta.CPU),
                        Comm:     trace.Meta.Comm.String(),
                        IsUprobe: trace.Meta.Origin == TraceOriginUProbe,
                    },
                }
                events <- event
            }
        }
    }()
    return nil
}

func (s *OTelSource) buildUprobeList() []string {
    uprobes := []string{}

    if s.mode.EnableUprobeKernelLaunch {
        uprobes = append(uprobes, findCudaLaunchSymbols()...)
    }

    // EnableUprobeCorrelation is handled separately via ContextCapturer
    // (Layer 2 abstraction - see below)

    return uprobes
}
```

### PMU Source (Future)

```go
// PMUSource reads hardware performance counters
type PMUSource struct {
    counters []string // "cycles", "instructions", "cache-misses"
}

func NewPMUSource(counters []string) *PMUSource {
    return &PMUSource{counters: counters}
}

func (s *PMUSource) Start(ctx context.Context, events chan<- *Event) error {
    // Use perf_event_open syscall to read PMU
    // Periodically emit PMU counter events
    return nil
}
```

---

## Source Composition (Combinators)

The EventSource interface allows **composing sources** to create more complex sources. This enables:
1. **Single source transformation** (filter, enrich, sample)
2. **Multi-source combination** (merge, correlate, join)

### Composition Patterns

#### 1. Filter Source (Transform Single Source)

```go
// FilterSource wraps a source and filters events
type FilterSource struct {
    source    EventSource
    predicate func(*Event) bool
    name      string
}

func NewFilterSource(source EventSource, predicate func(*Event) bool) *FilterSource {
    return &FilterSource{
        source:    source,
        predicate: predicate,
        name:      fmt.Sprintf("filter(%s)", source.Name()),
    }
}

func (s *FilterSource) Name() string {
    return s.name
}

func (s *FilterSource) Initialize() error {
    return s.source.Initialize()
}

func (s *FilterSource) Start(ctx context.Context, events chan<- *Event) error {
    // Create internal channel to receive from wrapped source
    internal := make(chan *Event, 1000)

    // Start wrapped source
    if err := s.source.Start(ctx, internal); err != nil {
        return err
    }

    // Filter and forward events
    go func() {
        for {
            select {
            case <-ctx.Done():
                return
            case event := <-internal:
                if s.predicate(event) {
                    events <- event
                }
            }
        }
    }()

    return nil
}

func (s *FilterSource) GetCapabilities() SourceCapabilities {
    return s.source.GetCapabilities()
}

// Usage: Filter only GPU kernel events from CUPTI
cuptiSource := NewCUPTISource(...)
kernelOnlySource := NewFilterSource(cuptiSource, func(e *Event) bool {
    return e.Type == EventGPUKernel
})
```

#### 2. Enrich Source (Transform Single Source)

```go
// EnrichSource wraps a source and adds additional data to events
type EnrichSource struct {
    source   EventSource
    enricher func(*Event) *Event
}

func NewEnrichSource(source EventSource, enricher func(*Event) *Event) *EnrichSource {
    return &EnrichSource{
        source:   source,
        enricher: enricher,
    }
}

func (s *EnrichSource) Start(ctx context.Context, events chan<- *Event) error {
    internal := make(chan *Event, 1000)

    if err := s.source.Start(ctx, internal); err != nil {
        return err
    }

    go func() {
        for {
            select {
            case <-ctx.Done():
                return
            case event := <-internal:
                enriched := s.enricher(event)
                events <- enriched
            }
        }
    }()

    return nil
}

// Usage: Add process name to CPU traces
otelSource := NewOTelSource(...)
enrichedSource := NewEnrichSource(otelSource, func(e *Event) *Event {
    if e.Type == EventCPUTrace {
        // Look up process name from PID
        e.Data.(*CPUTraceData).ProcessName = getProcessName(e.ProcessID)
    }
    return e
})
```

#### 3. Merge Source (Combine Multiple Sources)

```go
// MergeSource combines multiple sources into one stream
type MergeSource struct {
    sources []EventSource
}

func NewMergeSource(sources ...EventSource) *MergeSource {
    return &MergeSource{sources: sources}
}

func (s *MergeSource) Name() string {
    names := make([]string, len(s.sources))
    for i, src := range s.sources {
        names[i] = src.Name()
    }
    return fmt.Sprintf("merge(%s)", strings.Join(names, ","))
}

func (s *MergeSource) Initialize() error {
    for _, source := range s.sources {
        if err := source.Initialize(); err != nil {
            return fmt.Errorf("failed to initialize %s: %w", source.Name(), err)
        }
    }
    return nil
}

func (s *MergeSource) Start(ctx context.Context, events chan<- *Event) error {
    // Start all sources, they all send to the same output channel
    for _, source := range s.sources {
        if err := source.Start(ctx, events); err != nil {
            return fmt.Errorf("failed to start %s: %w", source.Name(), err)
        }
    }
    return nil
}

func (s *MergeSource) GetCapabilities() SourceCapabilities {
    // Union of all source capabilities
    caps := SourceCapabilities{}
    for _, source := range s.sources {
        srcCaps := source.GetCapabilities()
        caps.ProvidesCorrelationID = caps.ProvidesCorrelationID || srcCaps.ProvidesCorrelationID
        caps.ProvidesStackTraces = caps.ProvidesStackTraces || srcCaps.ProvidesStackTraces
        caps.ProvidesTimestamp = caps.ProvidesTimestamp || srcCaps.ProvidesTimestamp
        caps.SupportedEventTypes = append(caps.SupportedEventTypes, srcCaps.SupportedEventTypes...)
    }
    return caps
}

// Usage: Merge CUPTI and OTel into single stream
cuptiSource := NewCUPTISource(...)
otelSource := NewOTelSource(...)
mergedSource := NewMergeSource(cuptiSource, otelSource)
```

#### 4. Correlate Source (Combine with Correlation)

```go
// CorrelateSource combines two sources and correlates events
type CorrelateSource struct {
    source1   EventSource
    source2   EventSource
    strategy  CorrelationStrategy
    bufferDuration time.Duration
}

func NewCorrelateSource(source1, source2 EventSource, strategy CorrelationStrategy, bufferDuration time.Duration) *CorrelateSource {
    return &CorrelateSource{
        source1:        source1,
        source2:        source2,
        strategy:       strategy,
        bufferDuration: bufferDuration,
    }
}

func (s *CorrelateSource) Start(ctx context.Context, events chan<- *Event) error {
    internal := make(chan *Event, 10000)
    buffer := NewEventBuffer(s.bufferDuration)

    // Start both sources
    if err := s.source1.Start(ctx, internal); err != nil {
        return err
    }
    if err := s.source2.Start(ctx, internal); err != nil {
        return err
    }

    // Correlate events
    go func() {
        for {
            select {
            case <-ctx.Done():
                return
            case event := <-internal:
                buffer.Add(event)

                // Try to correlate
                if merged := s.strategy.TryCorrelate(event, buffer); merged != nil {
                    // Convert MergedEvent back to Event
                    correlatedEvent := mergedEventToEvent(merged)
                    events <- correlatedEvent
                } else {
                    // No correlation found, forward as-is (optional)
                    events <- event
                }
            }
        }
    }()

    return nil
}

// Usage: Create a pre-correlated source
cuptiSource := NewCUPTISource(...)
otelSource := NewOTelSource(...)
correlatedSource := NewCorrelateSource(
    cuptiSource,
    otelSource,
    &CorrelationIDStrategy{},
    10 * time.Second,
)
```

#### 5. Sample Source (Transform Single Source)

```go
// SampleSource reduces event rate by sampling
type SampleSource struct {
    source EventSource
    rate   int // Keep 1 out of every N events
}

func NewSampleSource(source EventSource, rate int) *SampleSource {
    return &SampleSource{source: source, rate: rate}
}

func (s *SampleSource) Start(ctx context.Context, events chan<- *Event) error {
    internal := make(chan *Event, 1000)

    if err := s.source.Start(ctx, internal); err != nil {
        return err
    }

    go func() {
        counter := 0
        for {
            select {
            case <-ctx.Done():
                return
            case event := <-internal:
                counter++
                if counter%s.rate == 0 {
                    events <- event
                }
            }
        }
    }()

    return nil
}

// Usage: Sample PMU to reduce overhead
pmuSource := NewPMUSource([]string{"cycles", "instructions"})
sampledPMU := NewSampleSource(pmuSource, 10) // Keep 1 in 10
```

---

## Composition Examples

### Example 1: Filtered and Enriched CUPTI

```go
// Only kernel events, with duration histogram
cuptiSource := NewCUPTISource(cuptiLibPath, true)

// Filter: only kernels
kernelSource := NewFilterSource(cuptiSource, func(e *Event) bool {
    return e.Type == EventGPUKernel
})

// Enrich: add duration bucket
enrichedSource := NewEnrichSource(kernelSource, func(e *Event) *Event {
    if data, ok := e.Data.(*GPUKernelData); ok {
        data.DurationBucket = getDurationBucket(data.Duration)
    }
    return e
})

coordinator.RegisterSource(enrichedSource)
```

### Example 2: Pre-Correlated CUPTI+OTel Source

```go
// Create individual sources
cuptiSource := NewCUPTISource(cuptiLibPath, true)
otelSource := NewOTelSource(OTelCollectionMode{
    EnableUprobeCorrelation: true,
})

// Combine with correlation
correlatedSource := NewCorrelateSource(
    cuptiSource,
    otelSource,
    &CorrelationIDStrategy{},
    10 * time.Second,
)

// Register single correlated source
coordinator.RegisterSource(correlatedSource)

// No need for separate correlation strategies in coordinator!
// The source itself is already correlated
```

### Example 3: Multi-Source Pipeline

```go
// Build a complex pipeline
cuptiSource := NewCUPTISource(cuptiLibPath, true)
otelSource := NewOTelSource(otelMode)
pmuSource := NewPMUSource([]string{"cycles", "cache-misses"})

// Sample PMU to reduce overhead
sampledPMU := NewSampleSource(pmuSource, 100)

// Merge CPU sources (OTel + PMU)
cpuSources := NewMergeSource(otelSource, sampledPMU)

// Filter PMU to only attach to CPU traces
enrichedCPU := NewEnrichSource(cpuSources, func(e *Event) *Event {
    // Attach recent PMU counters to CPU traces
    if e.Type == EventCPUTrace {
        attachPMUCounters(e)
    }
    return e
})

// Correlate with GPU
final := NewCorrelateSource(
    enrichedCPU,
    cuptiSource,
    &CorrelationIDStrategy{},
    10 * time.Second,
)

coordinator.RegisterSource(final)
```

### Example 4: Conditional Sources

```go
// Build source based on config
var finalSource EventSource

if cfg.gpuOnly {
    // GPU-only: CUPTI + OTel uprobes, correlated
    cuptiSource := NewCUPTISource(cfg.cuptiLibPath, true)
    otelSource := NewOTelSource(OTelCollectionMode{
        EnableSampling:          false,
        EnableUprobeCorrelation: true,
    })
    finalSource = NewCorrelateSource(cuptiSource, otelSource, &CorrelationIDStrategy{}, 10*time.Second)

} else if cfg.cpuOnly {
    // CPU-only: just OTel sampling
    finalSource = NewOTelSource(OTelCollectionMode{
        EnableSampling: true,
    })

} else {
    // Merge mode: CUPTI + OTel sampling + OTel uprobes
    cuptiSource := NewCUPTISource(cfg.cuptiLibPath, true)
    otelSampling := NewOTelSource(OTelCollectionMode{
        EnableSampling:          true,
        EnableUprobeCorrelation: false,
    })
    otelUprobes := NewOTelSource(OTelCollectionMode{
        EnableSampling:          false,
        EnableUprobeCorrelation: true,
    })

    // Correlate uprobes with GPU
    correlated := NewCorrelateSource(otelUprobes, cuptiSource, &CorrelationIDStrategy{}, 10*time.Second)

    // Merge correlated with pure CPU sampling
    finalSource = NewMergeSource(correlated, otelSampling)
}

coordinator.RegisterSource(finalSource)
```

---

## OTel Source Internals: Collection Modes

Within the OTel source implementation, different collection modes are handled by **context capturers**. This is an internal detail, not exposed to the EventSource abstraction:

```go
// ContextCapturer captures additional context for OTel traces
// This is an internal implementation detail of OTel source
type ContextCapturer interface {
    Name() string
    Attach(trc *tracer.Tracer) error
    Close() error
}

// CorrelationIDCapturer attaches uprobe to XpuPerfGetCorrelationId
type CorrelationIDCapturer struct {
    state       *ebpf.Collection
    link        link.Link
    cuptiPath   string
}

func (c *CorrelationIDCapturer) Attach(trc *tracer.Tracer) error {
    // Load eBPF program, attach to XpuPerfGetCorrelationId
    // Store correlation ID in custom_context_map
    // Tail call to custom__generic
    return nil
}

// Usage in OTelSource
func (s *OTelSource) Initialize() error {
    // Create tracer
    s.tracer = tracer.NewTracer(...)

    // Attach context capturers based on mode
    if s.mode.EnableUprobeCorrelation {
        capturer := &CorrelationIDCapturer{cuptiPath: ...}
        capturer.Attach(s.tracer)
    }

    return nil
}
```

This keeps the **correlation uprobe logic isolated within OTel source**, not exposed to the EventSource abstraction. From the outside, you just configure OTel with `EnableUprobeCorrelation: true`.

---

## Event Coordinator & Correlation

The EventCoordinator is **optional** - you can use composite sources directly. It's useful when you want to apply correlation strategies globally rather than per-source.

### Event Coordinator

```go
// EventCoordinator collects events from all sources and correlates them
type EventCoordinator struct {
    sources     []EventSource
    strategies  []CorrelationStrategy
    events      chan *Event
    output      chan *MergedEvent
}

func NewEventCoordinator() *EventCoordinator {
    return &EventCoordinator{
        sources:    make([]EventSource, 0),
        strategies: make([]CorrelationStrategy, 0),
        events:     make(chan *Event, 10000),
        output:     make(chan *MergedEvent, 1000),
    }
}

func (c *EventCoordinator) RegisterSource(source EventSource) {
    c.sources = append(c.sources, source)
}

func (c *EventCoordinator) RegisterCorrelationStrategy(strategy CorrelationStrategy) {
    c.strategies = append(c.strategies, strategy)
}

func (c *EventCoordinator) Start(ctx context.Context) error {
    // Initialize all sources
    for _, source := range c.sources {
        if err := source.Initialize(); err != nil {
            return fmt.Errorf("failed to initialize %s: %w", source.Name(), err)
        }
    }

    // Start all sources (they emit to c.events channel)
    for _, source := range c.sources {
        if err := source.Start(ctx, c.events); err != nil {
            return fmt.Errorf("failed to start %s: %w", source.Name(), err)
        }
    }

    // Start correlation engine
    go c.correlationLoop(ctx)

    return nil
}

func (c *EventCoordinator) correlationLoop(ctx context.Context) {
    buffer := NewEventBuffer(10 * time.Second) // Buffer for time-based correlation

    for {
        select {
        case <-ctx.Done():
            return
        case event := <-c.events:
            buffer.Add(event)

            // Try to correlate with buffered events
            for _, strategy := range c.strategies {
                if merged := strategy.TryCorrelate(event, buffer); merged != nil {
                    c.output <- merged
                }
            }
        }
    }
}
```

### Correlation Strategies

```go
// CorrelationStrategy defines how to match events from different sources
type CorrelationStrategy interface {
    Name() string
    TryCorrelate(event *Event, buffer *EventBuffer) *MergedEvent
}

// CorrelationIDStrategy matches events by correlation ID
type CorrelationIDStrategy struct{}

func (s *CorrelationIDStrategy) TryCorrelate(event *Event, buffer *EventBuffer) *MergedEvent {
    if event.CorrelationID == 0 {
        return nil // No correlation ID
    }

    // Find matching event with same correlation ID from different source
    match := buffer.FindByCorrelationID(event.CorrelationID, event.Source)
    if match == nil {
        return nil
    }

    // Merge CPU trace + GPU kernel
    return &MergedEvent{
        CPUEvent: extractCPUEvent(event, match),
        GPUEvent: extractGPUEvent(event, match),
    }
}

// TimestampStrategy matches events by timestamp proximity
type TimestampStrategy struct {
    ToleranceNs int64
}

func (s *TimestampStrategy) TryCorrelate(event *Event, buffer *EventBuffer) *MergedEvent {
    // Find events within tolerance window
    candidates := buffer.FindByTimestamp(event.Timestamp, s.ToleranceNs)

    // Match CPU trace with nearest GPU kernel
    for _, candidate := range candidates {
        if canMatch(event, candidate) {
            return merge(event, candidate)
        }
    }
    return nil
}
```

---

## Main.go Integration

```go
func main() {
    // Parse config
    cfg := parseFlags()

    // Create event coordinator
    coordinator := NewEventCoordinator()

    // Register sources based on config
    if !cfg.cpuOnly {
        // Add CUPTI source
        cuptiSource := NewCUPTISource(cfg.cuptiLibPath, cfg.useCorrelationUprobe)
        coordinator.RegisterSource(cuptiSource)
    }

    // Add OTel source with appropriate mode
    otelMode := OTelCollectionMode{
        EnableSampling:           !cfg.gpuOnly,
        EnableUprobeKernelLaunch: cfg.gpuOnly || cfg.mergeMode,
        EnableUprobeCorrelation:  cfg.useCorrelationUprobe,
        SamplingFrequencyHz:      50,
    }
    otelSource := NewOTelSource(otelMode)
    coordinator.RegisterSource(otelSource)

    // Future: Add PMU source
    // pmuSource := NewPMUSource([]string{"cycles", "instructions"})
    // coordinator.RegisterSource(pmuSource)

    // Register correlation strategies
    if cfg.useCorrelationUprobe {
        coordinator.RegisterCorrelationStrategy(&CorrelationIDStrategy{})
    } else {
        coordinator.RegisterCorrelationStrategy(&TimestampStrategy{
            ToleranceNs: 10 * time.Millisecond.Nanoseconds(),
        })
    }

    // Start coordination
    ctx, cancel := context.WithCancel(context.Background())
    defer cancel()

    if err := coordinator.Start(ctx); err != nil {
        log.Fatalf("Failed to start coordinator: %v", err)
    }

    // Collect merged events and write output
    writer := NewFoldedStackWriter(cfg.outputFile)
    for mergedEvent := range coordinator.Output() {
        writer.Write(mergedEvent)
    }
}
```

---

## File Structure

```
profiler/
├── main.go                          # Clean orchestration (~50 lines)
│
├── event/
│   ├── event.go                     # Event, EventType, EventData types
│   ├── coordinator.go               # EventCoordinator (optional)
│   ├── buffer.go                    # EventBuffer for correlation
│   └── correlation.go               # CorrelationStrategy interface
│
├── source/
│   ├── source.go                    # EventSource interface (unified)
│   │
│   ├── primitive/                   # Primitive sources (data collectors)
│   │   ├── cupti/
│   │   │   ├── cupti_source.go     # CUPTISource implementation
│   │   │   └── parser.go           # CUPTI event parsing
│   │   ├── otel/
│   │   │   ├── otel_source.go      # OTelSource implementation
│   │   │   ├── mode.go             # OTelCollectionMode config
│   │   │   └── context/            # Internal: Context capturers
│   │   │       ├── capturer.go     # ContextCapturer interface
│   │   │       ├── correlation_id.go   # CorrelationIDCapturer
│   │   │       └── ebpf/
│   │   │           └── correlation_id.ebpf.c
│   │   └── pmu/
│   │       └── pmu_source.go       # Future: PMU source
│   │
│   └── composite/                   # Composite sources (operations on sources)
│       ├── filter.go                # FilterSource - transform
│       ├── enrich.go                # EnrichSource - transform
│       ├── sample.go                # SampleSource - transform
│       ├── merge.go                 # MergeSource - combine
│       └── correlate.go             # CorrelateSource - combine with correlation
│
├── correlation/
│   ├── strategy.go                  # CorrelationStrategy interface
│   ├── correlation_id_strategy.go  # Match by correlation ID
│   └── timestamp_strategy.go       # Match by timestamp proximity
│
└── output/
    └── folded_writer.go             # Output formatting
```

---

## Benefits

### Unified Abstraction
✅ **Single interface**: Primitives and composites both implement EventSource
✅ **Source independence**: CUPTI, OTel, PMU are fully decoupled
✅ **Common event model**: Easy to add new sources
✅ **Composability**: Build complex pipelines from simple operations

### Extensibility
✅ **New primitives**: Add PMU by implementing EventSource
✅ **New operations**: Add new composite sources (e.g., Aggregate, Window)
✅ **New strategies**: Pluggable correlation strategies
✅ **Internal flexibility**: OTel context capturers are isolated plugins

### Overall
✅ **Testable**: Each source and strategy can be unit tested
✅ **Maintainable**: Clear ownership and responsibility
✅ **Extensible**: Adding PMU/new sources is straightforward
✅ **Not too complex**: ~300 lines of framework code

---

## Migration Path

### Phase 1: Extract Event Model (1-2 days)
1. Define Event, EventType, EventData types
2. Create EventSource interface
3. Keep existing code, add adapters

### Phase 2: Refactor CUPTI Source (1 day)
1. Wrap CUPTIParser in CUPTISource
2. Convert CUPTI events to common Event format
3. Test with existing correlation logic

### Phase 3: Refactor OTel Source (2-3 days)
1. Create OTelSource with collection modes
2. Extract context capturers (correlation uprobe)
3. Convert OTel traces to common Event format

### Phase 4: Implement Coordinator (2 days)
1. Create EventCoordinator
2. Implement correlation strategies
3. Migrate correlation logic from TraceCorrelator

### Phase 5: Update Main (1 day)
1. Replace direct source creation with coordinator
2. Remove old correlation code
3. Clean up flags/config

**Total effort**: ~7-9 days

---

## Complexity Assessment

### Added Components
- **Event model**: ~100 lines (event.go)
- **EventSource interface**: ~50 lines
- **EventCoordinator**: ~200 lines
- **CorrelationStrategy**: ~100 lines
- **Source implementations**: ~300 lines (CUPTI + OTel wrappers)

**Total new code**: ~750 lines

### Removed/Simplified
- **TraceCorrelator**: Replaced by strategies (~500 lines removed)
- **Main.go**: Simplified (~100 lines removed)
- **Hard-coded logic**: Removed (~200 lines)

**Net change**: ~+50 lines, but much better structure

---

## Example: Adding PMU Source

```go
// 1. Create PMU source (source/pmu/pmu_source.go)
type PMUSource struct {
    counters []string
}

func (s *PMUSource) Start(ctx context.Context, events chan<- *Event) error {
    // perf_event_open, read counters, emit events
    return nil
}

// 2. Register in main.go
if cfg.enablePMU {
    pmuSource := NewPMUSource([]string{"cycles", "instructions"})
    coordinator.RegisterSource(pmuSource)
}

// 3. Add correlation strategy (optional)
type PMUCorrelationStrategy struct{}

func (s *PMUCorrelationStrategy) TryCorrelate(event *Event, buffer *EventBuffer) *MergedEvent {
    // Attach PMU counters to CPU traces by timestamp
    return nil
}

// Done! PMU data now flows through the same pipeline
```

---

## Summary: Key Design Decisions

### 1. Sources as Composable Building Blocks
**Decision**: EventSource interface applies to both primitive sources (CUPTI, OTel) and composite sources (Filter, Merge, Correlate)

**Benefit**: Build complex pipelines declaratively
```go
// Instead of:
coordinator.AddSource(cupti)
coordinator.AddSource(otel)
coordinator.AddCorrelationStrategy(...)

// We write:
final := NewCorrelateSource(
    NewFilterSource(cupti, onlyKernels),
    NewEnrichSource(otel, addContext),
    &CorrelationIDStrategy{},
    10*time.Second,
)
coordinator.RegisterSource(final)
```

### 2. Operations on Sources (Not Layers)
**All sources implement EventSource** - no distinction between primitives and composites

**Operations available**:
- Transform single source: Filter, Enrich, Sample
- Combine sources: Merge, Correlate
- All operations return EventSource (composable)

**OTel collection modes** are internal implementation details:
- Configured via OTelCollectionMode struct
- Context capturers (correlation ID, etc.) hidden inside
- From outside: just another EventSource

**Benefit**: No layering complexity. Everything is an operation on EventSource. Adding PMU is same as adding any primitive. Adding Filter is same as adding any composite.

### 3. Correlation as Source Composition
**Decision**: Correlation can happen in two places:
1. **Within a CorrelateSource**: Pre-correlate two sources into one
2. **In EventCoordinator**: Apply strategies to all events

**Benefit**: Flexibility. Choose where correlation happens based on use case.

```go
// Approach 1: Pre-correlate as source
correlatedSource := NewCorrelateSource(cupti, otel, strategy, duration)
coordinator.RegisterSource(correlatedSource)

// Approach 2: Correlate in coordinator
coordinator.RegisterSource(cupti)
coordinator.RegisterSource(otel)
coordinator.RegisterCorrelationStrategy(strategy)
```

### 4. No Global Flags
**Decision**: Replace `gpuOnly`, `cpuOnly`, `mergeMode` flags with declarative source construction

**Before**:
```go
if cfg.gpuOnly {
    // Do stuff...
} else if cfg.mergeMode {
    // Do other stuff...
}
```

**After**:
```go
var source EventSource
switch cfg.mode {
case "gpu-only":
    source = buildGPUOnlySource(cfg)
case "cpu-only":
    source = buildCPUOnlySource(cfg)
case "merge":
    source = buildMergeSource(cfg)
}
coordinator.RegisterSource(source)
```

---

## Questions for Discussion

1. **Composition performance**: Are nested channels too expensive? Need benchmarking?
2. **Event model**: Is the proposed Event structure sufficient? Missing fields?
3. **Buffering**: How long should EventBuffer keep events for correlation?
4. **Error handling**: How to handle source failures (restart, skip, fail)?
5. **Configuration**: Use config file instead of flags for complex setups?
6. **EventCoordinator necessity**: Do we even need it, or just use composite sources directly?
7. **Backwards compatibility**: Keep old flags working during migration?
