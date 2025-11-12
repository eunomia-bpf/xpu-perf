// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"sync"

	"go.opentelemetry.io/ebpf-profiler/libpf"
	_ "go.opentelemetry.io/ebpf-profiler/libpf/pfelf" // Disabled for now
	"go.opentelemetry.io/ebpf-profiler/reporter/samples"
	"go.opentelemetry.io/ebpf-profiler/support"
)

// SimpleReporter implements the TraceReporter interface with built-in symbol resolution.
// Uses the official pfelf package from opentelemetry-ebpf-profiler for symbol resolution,
// the same infrastructure the profiler uses internally for interpreter symbolization.
type SimpleReporter struct {
	mu            sync.Mutex
	traceCount    int
	uprobeCount   int
	samplingCount int
	// Symbol cache: fileName -> SymbolMap (disabled for now - API changed)
	// symbolCache map[string]*libpf.SymbolMap
	// Track files that failed to load
	// symbolMisses map[string]bool
	// Correlator for merging CPU and GPU traces
	correlator *TraceCorrelator
	// Target PID to filter sampling traces (0 = all PIDs)
	targetPID int
}

// NewSimpleReporter creates a new simple reporter
func NewSimpleReporter(correlator *TraceCorrelator) *SimpleReporter {
	return &SimpleReporter{
		// symbolCache:  make(map[string]*libpf.SymbolMap),
		// symbolMisses: make(map[string]bool),
		correlator:   correlator,
		targetPID:    0, // Will be set later
	}
}

// SetTargetPID sets the target PID for filtering sampling traces
func (r *SimpleReporter) SetTargetPID(pid int) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.targetPID = pid
}

// ReportTraceEvent processes and prints trace information
func (r *SimpleReporter) ReportTraceEvent(trace *libpf.Trace, meta *samples.TraceEventMeta) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.traceCount++

	// Count by origin
	switch meta.Origin {
	case support.TraceOriginUProbe:
		r.uprobeCount++
	case support.TraceOriginSampling:
		r.samplingCount++
	case support.TraceOriginCustom:
		// Custom traces (e.g., from correlation ID uprobe) count as uprobe traces
		r.uprobeCount++
	}

	// Extract stack for correlation
	// - CPU-only mode: process all sampling traces
	// - Merge mode: process sampling traces from target PID only (correlate with GPU)
	// - GPU-only mode: process uprobe traces (for CPU/GPU correlation)
	if r.correlator != nil {
		shouldProcess := false

		if (meta.Origin == support.TraceOriginUProbe || meta.Origin == support.TraceOriginCustom) && (r.correlator.gpuOnly || r.correlator.mergeMode) {
			// Process uprobes and custom traces in GPU-only mode and merge mode (for correlation)
			shouldProcess = true
		} else if meta.Origin == support.TraceOriginSampling && (r.correlator.cpuOnly || r.correlator.mergeMode) {
			// In merge mode, only process traces from the target PID
			if r.correlator.mergeMode {
				if r.targetPID > 0 && int(meta.PID) == r.targetPID {
					shouldProcess = true
				}
			} else {
				// CPU-only mode: process all sampling traces
				shouldProcess = true
			}
		}

		if shouldProcess {
			stack := ExtractStackFromTrace(trace, meta)
			isUprobe := (meta.Origin == support.TraceOriginUProbe || meta.Origin == support.TraceOriginCustom)

			// Filter: Drop uprobes without stack frames
			// Empty stacks don't provide useful correlation information
			if isUprobe && len(stack) == 0 {
				// Skip this uprobe - no stack to correlate
				return nil
			}

			// Extract correlation ID from ContextValue if using correlation mode
			correlationID := uint32(meta.ContextValue)

			r.correlator.AddCPUTraceWithCorrelation(
				int64(meta.Timestamp),
				int(meta.PID),
				int(meta.TID),
				int(meta.CPU),
				meta.Comm.String(),
				stack,
				isUprobe,
				correlationID,
			)
		}
	}

	return nil
}

func (r *SimpleReporter) GetStats() (int, int, int) {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.traceCount, r.uprobeCount, r.samplingCount
}

// resolveSymbol tries to resolve an address to a symbol name using libpf/pfelf.
// This uses the same symbol resolution infrastructure that the profiler uses
// internally for Python, Ruby, Node.js symbolization, but applies it to native frames.
func (r *SimpleReporter) resolveSymbol(fileName string, addr uint64) string {
	// Check if this file previously failed to load
	// Symbol resolution temporarily disabled
	return ""
	return ""
}
