// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"fmt"
	"sync"

	"go.opentelemetry.io/ebpf-profiler/libpf"
	"go.opentelemetry.io/ebpf-profiler/reporter/samples"
	"go.opentelemetry.io/ebpf-profiler/support"
)

// SimpleReporter implements the TraceReporter interface with built-in symbol resolution.
// Uses the official pfelf package from opentelemetry-ebpf-profiler for symbol resolution,
// the same infrastructure the profiler uses internally for interpreter symbolization.
type SimpleReporter struct {
	mu            sync.Mutex
	traceCount    int
	customCount   int
	samplingCount int
}

// NewSimpleReporter creates a new simple reporter
func NewSimpleReporter() *SimpleReporter {
	return &SimpleReporter{}
}

// ReportTraceEvent processes and prints trace information
func (r *SimpleReporter) ReportTraceEvent(trace *libpf.Trace, meta *samples.TraceEventMeta) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.traceCount++

	// Count by origin
	switch meta.Origin {
	case support.TraceOriginCustom:
		r.customCount++
	case support.TraceOriginUProbe:
		// Skip regular uprobe traces
		return nil
	case support.TraceOriginSampling:
		r.samplingCount++
		return nil
	default:
		return nil
	}

	// Only print custom traces
	fmt.Printf("\n=== CUSTOM Trace #%d (Hash: %x) ===\n", r.customCount, trace.Hash)
	fmt.Printf("PID: %d, TID: %d, CPU: %d\n", meta.PID, meta.TID, meta.CPU)
	fmt.Printf("Comm: %s, Process: %s\n", meta.Comm, meta.ProcessName)
	fmt.Printf("Executable: %s\n", meta.ExecutablePath)
	fmt.Printf("Timestamp: %d ns\n", meta.Timestamp)

	// Print context value for custom traces
	fmt.Printf("🎯 Context Value: %d (0x%x)\n", meta.OffTime, meta.OffTime)

	// Print custom labels if any
	if len(trace.CustomLabels) > 0 {
		fmt.Println("Labels:")
		for k, v := range trace.CustomLabels {
			fmt.Printf("  %s: %s\n", k, v)
		}
	}

	// Print stack trace
	fmt.Printf("Stack trace (%d frames):\n", len(trace.Frames))
	maxFrames := 10
	if len(trace.Frames) < maxFrames {
		maxFrames = len(trace.Frames)
	}
	for i := 0; i < maxFrames; i++ {
		frame := trace.Frames[i].Value()

		sourceFile := frame.SourceFile.String()
		if sourceFile == "" {
			sourceFile = "<unknown>"
		}

		// Get file information
		fileName := "<unknown>"
		if frame.MappingFile.Valid() {
			fileData := frame.MappingFile.Value()
			fileName = fileData.FileName.String()
		}

		functionName := frame.FunctionName.String()
		if functionName == "" {
			functionName = "<unknown>"
		}

		fmt.Printf("  #%d: %s", i, functionName)
		if frame.SourceLine > 0 {
			fmt.Printf(" at %s:%d", sourceFile, frame.SourceLine)
		}
		fmt.Printf(" [%s+0x%x]", fileName, frame.AddressOrLineno)
		fmt.Printf(" (type: %s)\n", frame.Type.String())
	}
	if len(trace.Frames) > maxFrames {
		fmt.Printf("  ... (%d more frames)\n", len(trace.Frames)-maxFrames)
	}

	return nil
}

func (r *SimpleReporter) GetStats() (int, int, int) {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.traceCount, r.customCount, r.samplingCount
}
