// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"

	"go.opentelemetry.io/ebpf-profiler/libpf"
	"go.opentelemetry.io/ebpf-profiler/reporter/samples"
)

// CPUStackTrace represents a CPU stack trace from uprobe or sampling
type CPUStackTrace struct {
	Timestamp int64
	PID       int
	TID       int
	CPU       int
	Comm      string
	Stack     []string
	IsUprobe  bool // true if from uprobe, false if from sampling
}

// MergedTrace represents a correlated CPU+GPU stack trace
type MergedTrace struct {
	CPUStack      []string
	GPUKernelName string
	DurationUs    int64 // GPU kernel duration in microseconds
}

// TraceCorrelator correlates CPU uprobe traces with GPU CUPTI events
type TraceCorrelator struct {
	cpuTraces       []CPUStackTrace
	gpuParser       *CUPTIParser
	mergedStacks    map[string]int64 // folded stack -> sample count
	toleranceMs     float64
	samplesPerSec   int              // CPU sampling frequency (Hz)
	mu              sync.Mutex
	cpuOnly         bool
	gpuOnly         bool // Now means: use uprobes for correlation
	mergeMode       bool // Merge CPU sampling + GPU samples
}

// NewTraceCorrelator creates a new trace correlator
func NewTraceCorrelator(gpuParser *CUPTIParser, toleranceMs float64, samplesPerSec int, cpuOnly, gpuOnly, mergeMode bool) *TraceCorrelator {
	return &TraceCorrelator{
		cpuTraces:     make([]CPUStackTrace, 0),
		gpuParser:     gpuParser,
		mergedStacks:  make(map[string]int64),
		toleranceMs:   toleranceMs,
		samplesPerSec: samplesPerSec,
		cpuOnly:       cpuOnly,
		gpuOnly:       gpuOnly,
		mergeMode:     mergeMode,
	}
}

// AddCPUTrace adds a CPU stack trace for correlation
func (c *TraceCorrelator) AddCPUTrace(timestamp int64, pid, tid, cpu int, comm string, stack []string, isUprobe bool) {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.cpuTraces = append(c.cpuTraces, CPUStackTrace{
		Timestamp: timestamp,
		PID:       pid,
		TID:       tid,
		CPU:       cpu,
		Comm:      comm,
		Stack:     stack,
		IsUprobe:  isUprobe,
	})
}

// CorrelateTraces correlates CPU and GPU traces using timestamp matching
func (c *TraceCorrelator) CorrelateTraces() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	// CPU-only mode: just output CPU stacks with sample count
	if c.cpuOnly {
		for _, cpuTrace := range c.cpuTraces {
			stackStr := strings.Join(cpuTrace.Stack, ";")
			c.mergedStacks[stackStr] += 1 // Count samples
		}
		return nil
	}

	// Merge mode: Separate uprobe and sampling traces, then merge
	// - Uprobe traces: correlate with GPU kernels (shows CPU→GPU causality)
	// - Sampling traces: keep as pure CPU activity
	if c.mergeMode {
		// Separate uprobes from sampling
		var uprobeTraces []CPUStackTrace
		var samplingTraces []CPUStackTrace

		for _, trace := range c.cpuTraces {
			if trace.IsUprobe {
				uprobeTraces = append(uprobeTraces, trace)
			} else {
				samplingTraces = append(samplingTraces, trace)
			}
		}

		// Add pure CPU sampling traces (no GPU kernel)
		for _, cpuTrace := range samplingTraces {
			stackStr := strings.Join(cpuTrace.Stack, ";")
			c.mergedStacks[stackStr] += 1
		}

		// Correlate uprobes with GPU kernels (reuse gpu-only logic)
		if len(uprobeTraces) > 0 {
			c.correlateUprobesToGPU(uprobeTraces)
		}

		return nil
	}

	// GPU-only mode: correlate CPU uprobe stacks with GPU kernels
	c.correlateUprobesToGPU(c.cpuTraces)
	return nil
}

// correlateUprobesToGPU correlates CPU uprobe traces with GPU kernels
// This is used by both gpu-only and merge modes
func (c *TraceCorrelator) correlateUprobesToGPU(cpuTraces []CPUStackTrace) error {
	runtimes := c.gpuParser.GetRuntimes()
	kernels := c.gpuParser.GetKernels()

	if len(cpuTraces) == 0 {
		return fmt.Errorf("no CPU traces collected")
	}

	if len(kernels) == 0 {
		return fmt.Errorf("no GPU kernels found in CUPTI trace")
	}

	// Build correlation ID -> kernel mapping
	kernelByCorr := make(map[int]*GPUKernelEvent)
	for i := range kernels {
		kernelByCorr[kernels[i].CorrelationID] = &kernels[i]
	}

	// Build correlation ID -> runtime mapping
	runtimeByCorr := make(map[int]*GPURuntimeEvent)
	for i := range runtimes {
		runtimeByCorr[runtimes[i].CorrelationID] = &runtimes[i]
	}

	// Sort CPU traces by timestamp
	sort.Slice(cpuTraces, func(i, j int) bool {
		return cpuTraces[i].Timestamp < cpuTraces[j].Timestamp
	})

	// Sort runtimes by timestamp
	sort.Slice(runtimes, func(i, j int) bool {
		return runtimes[i].StartNs < runtimes[j].StartNs
	})

	toleranceNs := int64(c.toleranceMs * 1e6)
	matched := 0
	unmatched := 0

	// Try sequential matching with time window validation
	if len(cpuTraces) == len(runtimes) {
		// Perfect 1:1 correspondence
		for i, cpuTrace := range cpuTraces {
			runtime := &runtimes[i]
			kernel := kernelByCorr[runtime.CorrelationID]

			if kernel != nil {
				merged := cpuTrace.Stack
				merged = append(merged, fmt.Sprintf("[GPU_Kernel]%s", kernel.Name))
				stackStr := strings.Join(merged, ";")
				durationUs := (kernel.EndNs - kernel.StartNs) / 1000
				// Convert GPU duration to sample count using CPU sampling frequency
				samples := (durationUs * int64(c.samplesPerSec)) / 1000000
				if samples < 1 {
					samples = 1
				}
				c.mergedStacks[stackStr] += samples
				matched++
			} else {
				unmatched++
			}
		}
	} else {
		// More events on one side - use time-based matching
		gpuIdx := 0

		for _, cpuTrace := range cpuTraces {
			// Skip GPU events too far behind
			for gpuIdx < len(runtimes) {
				timeDiff := cpuTrace.Timestamp - runtimes[gpuIdx].StartNs
				if timeDiff > toleranceNs {
					gpuIdx++
				} else {
					break
				}
			}

			if gpuIdx >= len(runtimes) {
				unmatched++
				continue
			}

			// Check if within tolerance window
			timeDiff := int64(math.Abs(float64(cpuTrace.Timestamp - runtimes[gpuIdx].StartNs)))
			if timeDiff <= toleranceNs {
				runtime := &runtimes[gpuIdx]
				kernel := kernelByCorr[runtime.CorrelationID]

				if kernel != nil {
					merged := cpuTrace.Stack
					merged = append(merged, fmt.Sprintf("[GPU_Kernel]%s", kernel.Name))
					stackStr := strings.Join(merged, ";")
					durationUs := (kernel.EndNs - kernel.StartNs) / 1000
					// Convert GPU duration to sample count using CPU sampling frequency
					samples := (durationUs * int64(c.samplesPerSec)) / 1000000
					if samples < 1 {
						samples = 1
					}
					c.mergedStacks[stackStr] += samples
					matched++
					gpuIdx++
				} else {
					unmatched++
				}
			} else {
				unmatched++
			}
		}
	}

	fmt.Printf("Correlation complete: matched=%d, unmatched=%d\n", matched, unmatched)
	return nil
}

// WriteFoldedOutput writes the merged stacks in folded format for flamegraph
func (c *TraceCorrelator) WriteFoldedOutput(filename string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create output file: %v", err)
	}
	defer file.Close()

	// Sort stacks for consistent output
	type stackEntry struct {
		stack string
		count int64
	}
	entries := make([]stackEntry, 0, len(c.mergedStacks))
	for stack, count := range c.mergedStacks {
		entries = append(entries, stackEntry{stack, count})
	}
	sort.Slice(entries, func(i, j int) bool {
		return entries[i].stack < entries[j].stack
	})

	// Write folded format: stack1;stack2;stack3 count
	totalSamples := int64(0)
	for _, entry := range entries {
		fmt.Fprintf(file, "%s %d\n", entry.stack, entry.count)
		totalSamples += entry.count
	}

	fmt.Printf("Wrote %d unique stacks (%d total samples) to %s\n", len(entries), totalSamples, filename)
	return nil
}

// Use the global symbolizer for all symbol resolution
// (symbolCache, pidExeCache, lookupSymbol removed - now using symbolizer.go)

// ExtractStackFromTrace extracts function names from a libpf.Trace
func ExtractStackFromTrace(trace *libpf.Trace, meta *samples.TraceEventMeta) []string {
	stack := make([]string, 0, len(trace.Frames))
	symbolizer := GetSymbolizer()

	// Iterate in reverse order: frames are stored innermost-first,
	// but flamegraph format needs outermost-first (root to leaf)
	for i := len(trace.Frames) - 1; i >= 0; i-- {
		frame := trace.Frames[i].Value()

		functionName := frame.FunctionName.String()
		frameName := ""

		// Try to get a meaningful frame identifier
		if functionName != "" && functionName != "<unknown>" {
			// Demangle if needed
			frameName = symbolizer.demangle(functionName)
		} else if frame.MappingFile.Valid() {
			fileData := frame.MappingFile.Value()
			fileName := fileData.FileName.String()
			if fileName != "" {
				baseName := filepath.Base(fileName)

				// For PIE binaries, we need to calculate the file offset
				// The frame contains:
				// - AddressOrLineno: Could be runtime VA or file offset depending on eBPF unwinder
				// - MappingStart: Base address where the file is mapped in memory
				// - MappingFileOffset: Offset within the file where this mapping starts

				// The correct file offset calculation depends on whether this is a PIE binary
				// For now, try AddressOrLineno directly (it might already be a file offset)
				fileOffset := uint64(frame.AddressOrLineno)

				// Resolve basename to full path using PID
				fullPath := ResolveExecutablePath(int(meta.PID), baseName)

				// Try to resolve symbol from ELF using the file offset
				symbol := symbolizer.Symbolize(fullPath, fileOffset)
				if symbol != "" {
					frameName = symbol
				} else {
					// Fall back to file+offset
					frameName = fmt.Sprintf("%s+0x%x", baseName, fileOffset)
				}
			}
		}

		if frameName == "" {
			frameName = fmt.Sprintf("0x%x", frame.AddressOrLineno)
		}

		// Filter out common uninteresting frames
		if frameName == "_start" || frameName == "__libc_start_main" {
			continue
		}

		// Collapse cudaLaunchKernel wrappers
		if strings.Contains(frameName, "cudaLaunchKernel") {
			if len(stack) == 0 || !strings.Contains(stack[len(stack)-1], "cudaLaunchKernel") {
				stack = append(stack, "cudaLaunchKernel")
			}
		} else {
			stack = append(stack, frameName)
		}
	}

	return stack
}
