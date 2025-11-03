// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"sync"

	"go.opentelemetry.io/ebpf-profiler/libpf"
	"go.opentelemetry.io/ebpf-profiler/libpf/pfelf"
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
	// Symbol cache: fileName -> SymbolMap
	symbolCache map[string]*libpf.SymbolMap
	// Track files that failed to load
	symbolMisses map[string]bool
	// Correlator for merging CPU and GPU traces
	correlator *TraceCorrelator
}

// NewSimpleReporter creates a new simple reporter
func NewSimpleReporter(correlator *TraceCorrelator) *SimpleReporter {
	return &SimpleReporter{
		symbolCache:  make(map[string]*libpf.SymbolMap),
		symbolMisses: make(map[string]bool),
		correlator:   correlator,
	}
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
	}

	// Only process uprobe traces
	if meta.Origin != support.TraceOriginUProbe {
		return nil
	}

	// Extract stack for correlation
	if r.correlator != nil {
		stack := ExtractStackFromTrace(trace, meta)
		r.correlator.AddCPUTrace(
			int64(meta.Timestamp),
			int(meta.PID),
			int(meta.TID),
			int(meta.CPU),
			meta.Comm,
			stack,
		)
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
	if r.symbolMisses[fileName] {
		return ""
	}

	// Check cache
	syms, cached := r.symbolCache[fileName]
	if !cached {
		// Try to load symbols using official pfelf API
		ef, err := pfelf.Open(fileName)
		if err != nil {
			r.symbolMisses[fileName] = true
			return ""
		}
		defer ef.Close()

		// Try to read dynamic symbols first (works with stripped binaries)
		syms, err = ef.ReadDynamicSymbols()
		if err != nil {
			// Fall back to regular symbol table
			syms, err = ef.ReadSymbols()
			if err != nil {
				r.symbolMisses[fileName] = true
				return ""
			}
		}

		r.symbolCache[fileName] = syms
	}

	// Look up symbol by address using libpf.SymbolMap
	name, _, found := syms.LookupByAddress(libpf.SymbolValue(addr))
	if found {
		return string(name)
	}
	return ""
}
