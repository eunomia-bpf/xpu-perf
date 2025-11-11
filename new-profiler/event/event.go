package event

import (
	"context"
)

// EventType defines the type of profiling event
type EventType int

const (
	EventCPUTrace EventType = iota
	EventGPUKernel
	EventGPURuntime
)

// Event is the universal container for profiling data
type Event struct {
	Timestamp     int64           // Nanoseconds since epoch
	Source        string          // "cupti", "otel", etc.
	Type          EventType       // CPU_TRACE, GPU_KERNEL, etc.
	CorrelationID uint32          // For cross-source correlation (0 if N/A)
	ThreadID      uint32          // OS thread ID
	ProcessID     uint32          // OS process ID
	Data          EventData       // Type-specific payload
}

// EventData is the interface for event-specific data
type EventData interface {
	Type() EventType
}

// CPUTraceData contains CPU stack trace information
type CPUTraceData struct {
	Stack      []string
	CPU        int
	Comm       string
	IsUprobe   bool
	IsSampling bool
}

func (d *CPUTraceData) Type() EventType {
	return EventCPUTrace
}

// GPUKernelData contains GPU kernel execution information
type GPUKernelData struct {
	KernelName string
	Duration   int64
	StreamID   uint32
	DeviceID   int
}

func (d *GPUKernelData) Type() EventType {
	return EventGPUKernel
}

// GPURuntimeData contains GPU runtime API call information
type GPURuntimeData struct {
	APIName  string
	Duration int64
}

func (d *GPURuntimeData) Type() EventType {
	return EventGPURuntime
}

// EventSource represents any source of profiling events
// Both primitive sources (CUPTI, OTel) and composite sources (Filter, Merge)
// implement this interface
type EventSource interface {
	Name() string
	Initialize() error
	Start(ctx context.Context, events chan<- *Event) error
	Stop() error
}
