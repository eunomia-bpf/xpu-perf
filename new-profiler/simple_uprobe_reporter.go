package main

import (
	"fmt"
	"sync"

	"go.opentelemetry.io/ebpf-profiler/libpf"
	"go.opentelemetry.io/ebpf-profiler/reporter"
	"go.opentelemetry.io/ebpf-profiler/reporter/samples"
	"go.opentelemetry.io/ebpf-profiler/support"
)

type simpleUprobeReporter struct {
	mu                       sync.Mutex
	totalEvents              int
	customTraceEvents        int
	eventsWithCorrelation    int
	eventsWithoutCorrelation int
	correlationCount         map[uint32]int
}

func (r *simpleUprobeReporter) ReportTraceEvent(trace *libpf.Trace, meta *samples.TraceEventMeta) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.totalEvents++

	// Debug: Print first few events with their origins
	if r.totalEvents <= 10 {
		fmt.Printf("[DEBUG] Event #%d: Origin=%d (Custom=%d, Sampling=%d) ContextValue=%d PID=%d TID=%d\n",
			r.totalEvents, meta.Origin, support.TraceOriginCustom, support.TraceOriginSampling,
			meta.ContextValue, meta.PID, meta.TID)
	}

	// Check if this is a custom trace (from uprobe)
	isCustom := meta.Origin == support.TraceOriginCustom

	if isCustom {
		r.customTraceEvents++
		correlationID := uint32(meta.ContextValue)

		// Print ALL custom trace events
		fmt.Printf("\n🎯 CUSTOM TRACE #%d: CorrelationID=%d PID=%d TID=%d ContextValue=%d\n",
			r.customTraceEvents, correlationID, meta.PID, meta.TID, meta.ContextValue)
		fmt.Printf("   Stack depth: %d frames\n", len(trace.Frames))

		if correlationID != 0 {
			r.eventsWithCorrelation++
			r.correlationCount[correlationID]++
		} else {
			r.eventsWithoutCorrelation++
		}
	}

	return nil
}

func (r *simpleUprobeReporter) ReportFramesForTrace(_ *libpf.Trace) {
	// Not needed for testing
}

func (r *simpleUprobeReporter) ReportCountForTrace(_ *libpf.Trace, _ uint16) {
	// Not needed for testing
}

func (r *simpleUprobeReporter) ExecutableKnown(_ libpf.FileID, _ string) bool {
	return true
}

func (r *simpleUprobeReporter) ExecutableReports() {
	// Not needed for testing
}

func (r *simpleUprobeReporter) ReportExecutable(_ *reporter.ExecutableMetadata) {
	// Not needed for testing
}
