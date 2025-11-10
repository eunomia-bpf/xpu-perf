// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"fmt"
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
}

// NewSimpleReporter creates a new simple reporter
func NewSimpleReporter() *SimpleReporter {
	return &SimpleReporter{
		symbolCache:  make(map[string]*libpf.SymbolMap),
		symbolMisses: make(map[string]bool),
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

	// Only print uprobe traces
	if meta.Origin != support.TraceOriginUProbe {
		return nil
	}

	// Print trace metadata
	fmt.Printf("\n=== UPROBE Trace #%d (Hash: %x) ===\n", r.uprobeCount, trace.Hash)
	fmt.Printf("PID: %d, TID: %d, CPU: %d\n", meta.PID, meta.TID, meta.CPU)
	fmt.Printf("Comm: %s, Process: %s\n", meta.Comm, meta.ProcessName)
	fmt.Printf("Executable: %s\n", meta.ExecutablePath)
	fmt.Printf("Timestamp: %d ns\n", meta.Timestamp)

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

		// Get file information and resolve symbols if available
		fileName := "<unknown>"
		var filePath string
		if frame.MappingFile.Valid() {
			fileData := frame.MappingFile.Value()
			fileName = fileData.FileName.String()
			// Try to build full path for system libraries
			if fileName != "" && fileName[0] != '/' {
				// Common library paths
				for _, prefix := range []string{"/lib/x86_64-linux-gnu/", "/usr/lib/x86_64-linux-gnu/", "/lib64/", "/usr/lib/"} {
					testPath := prefix + fileName
					if name := r.resolveSymbol(testPath, uint64(frame.AddressOrLineno)); name != "" {
						frame.FunctionName = libpf.Intern(name)
						filePath = testPath
						break
					}
				}
			} else {
				filePath = fileName
				if name := r.resolveSymbol(filePath, uint64(frame.AddressOrLineno)); name != "" {
					frame.FunctionName = libpf.Intern(name)
				}
			}
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
