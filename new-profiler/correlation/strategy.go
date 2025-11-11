package correlation

import (
	"new-profiler/event"
	"sync"
)

// MergedEvent represents a correlated CPU+GPU event
type MergedEvent struct {
	CPUStack   []string
	GPUKernel  string
	DurationUs int64 // Duration in microseconds (for GPU kernels)
}

// CorrelationStrategy defines how to match events from different sources
type CorrelationStrategy interface {
	AddEvent(e *event.Event)
	GetMerged() []*MergedEvent
	GetCPUSamplingStacks() map[string]int // For merge mode: pure CPU sampling stacks
	GetStats() (matched, unmatched, noCorrelation, cpuSampling int)
	Flush()
	SetTargetPID(pid uint32)      // For merge mode: filter sampling traces by PID
	SetToleranceMs(toleranceMs float64) // For timestamp-based matching
}

// CorrelationIDStrategy matches events by correlation ID only
type CorrelationIDStrategy struct {
	mu         sync.Mutex
	cpuTraces  map[uint32]*event.Event // correlationID -> CPU event (from uprobe)
	gpuKernels map[uint32]*event.Event // correlationID -> GPU event (from CUPTI)
	merged     []*MergedEvent

	// For CPU-only sampling stacks (no GPU correlation)
	cpuSamplingStacks map[string]int

	// Statistics
	matched       int
	unmatched     int
	noCorrelation int
	cpuSampling   int

	// Configuration
	targetPID uint32 // Filter sampling traces by this PID (0 = all PIDs)
}

func NewCorrelationIDStrategy() *CorrelationIDStrategy {
	return &CorrelationIDStrategy{
		cpuTraces:         make(map[uint32]*event.Event),
		gpuKernels:        make(map[uint32]*event.Event),
		merged:            make([]*MergedEvent, 0),
		cpuSamplingStacks: make(map[string]int),
	}
}

func (s *CorrelationIDStrategy) SetTargetPID(pid uint32) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.targetPID = pid
}

func (s *CorrelationIDStrategy) SetToleranceMs(toleranceMs float64) {
	// No-op: timestamp-based matching removed
}

func (s *CorrelationIDStrategy) AddEvent(e *event.Event) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if e.Type == event.EventCPUTrace {
		cpuData := e.Data.(*event.CPUTraceData)

		// Handle CPU sampling traces separately (for merge mode)
		if cpuData.IsSampling {
			// Filter by target PID if set
			if s.targetPID == 0 || e.ProcessID == s.targetPID {
				// This is a pure CPU sampling trace (no GPU)
				s.addCPUSamplingStack(cpuData.Stack)
				s.cpuSampling++
			}
			return
		}

		// Handle uprobe traces with correlation ID
		if e.CorrelationID == 0 {
			// No correlation ID - cannot correlate
			s.noCorrelation++
			return
		}

		s.cpuTraces[e.CorrelationID] = e

		// Try to match with existing GPU event
		if gpu, exists := s.gpuKernels[e.CorrelationID]; exists {
			s.createMergedEvent(e, gpu)
			delete(s.cpuTraces, e.CorrelationID)
			delete(s.gpuKernels, e.CorrelationID)
			s.matched++
		}
	} else if e.Type == event.EventGPUKernel {
		if e.CorrelationID == 0 {
			// No correlation ID - cannot correlate
			s.noCorrelation++
			return
		}

		s.gpuKernels[e.CorrelationID] = e

		// Try to match with existing CPU event
		if cpu, exists := s.cpuTraces[e.CorrelationID]; exists {
			s.createMergedEvent(cpu, e)
			delete(s.cpuTraces, e.CorrelationID)
			delete(s.gpuKernels, e.CorrelationID)
			s.matched++
		}
	}
}

func (s *CorrelationIDStrategy) addCPUSamplingStack(stack []string) {
	// Convert stack to folded format
	stackStr := ""
	for i, frame := range stack {
		if i > 0 {
			stackStr += ";"
		}
		stackStr += frame
	}
	s.cpuSamplingStacks[stackStr]++
}

func (s *CorrelationIDStrategy) createMergedEvent(cpu, gpu *event.Event) {
	cpuData := cpu.Data.(*event.CPUTraceData)
	gpuData := gpu.Data.(*event.GPUKernelData)

	// Duration is already in nanoseconds in GPUKernelData
	// Convert to microseconds for consistency
	durationUs := gpuData.Duration / 1000

	s.merged = append(s.merged, &MergedEvent{
		CPUStack:   cpuData.Stack,
		GPUKernel:  gpuData.KernelName,
		DurationUs: durationUs,
	})
}

func (s *CorrelationIDStrategy) GetMerged() []*MergedEvent {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.merged
}

func (s *CorrelationIDStrategy) GetCPUSamplingStacks() map[string]int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.cpuSamplingStacks
}

func (s *CorrelationIDStrategy) GetStats() (matched, unmatched, noCorrelation, cpuSampling int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.matched, s.unmatched, s.noCorrelation, s.cpuSampling
}

func (s *CorrelationIDStrategy) Flush() {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Count remaining unmatched events
	s.unmatched = len(s.cpuTraces) + len(s.gpuKernels)

	// Clear buffers
	s.cpuTraces = make(map[uint32]*event.Event)
	s.gpuKernels = make(map[uint32]*event.Event)
}
