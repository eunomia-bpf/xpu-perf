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
	"go.opentelemetry.io/ebpf-profiler/libpf/pfelf"
	"go.opentelemetry.io/ebpf-profiler/reporter/samples"
)

// CPUStackTrace represents a CPU stack trace from uprobe
type CPUStackTrace struct {
	Timestamp int64
	PID       int
	TID       int
	CPU       int
	Comm      string
	Stack     []string
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
	mergedStacks    map[string]int64 // folded stack -> duration sum (microseconds)
	toleranceMs     float64
	mu              sync.Mutex
	cpuOnly         bool
	gpuOnly         bool
}

// NewTraceCorrelator creates a new trace correlator
func NewTraceCorrelator(gpuParser *CUPTIParser, toleranceMs float64, cpuOnly, gpuOnly bool) *TraceCorrelator {
	return &TraceCorrelator{
		cpuTraces:    make([]CPUStackTrace, 0),
		gpuParser:    gpuParser,
		mergedStacks: make(map[string]int64),
		toleranceMs:  toleranceMs,
		cpuOnly:      cpuOnly,
		gpuOnly:      gpuOnly,
	}
}

// AddCPUTrace adds a CPU stack trace for correlation
func (c *TraceCorrelator) AddCPUTrace(timestamp int64, pid, tid, cpu int, comm string, stack []string) {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.cpuTraces = append(c.cpuTraces, CPUStackTrace{
		Timestamp: timestamp,
		PID:       pid,
		TID:       tid,
		CPU:       cpu,
		Comm:      comm,
		Stack:     stack,
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

	// GPU-only mode: output GPU kernels only
	if c.gpuOnly {
		kernels := c.gpuParser.GetKernels()
		for _, kernel := range kernels {
			stackStr := fmt.Sprintf("[GPU_Kernel]%s", kernel.Name)
			durationUs := (kernel.EndNs - kernel.StartNs) / 1000
			c.mergedStacks[stackStr] += durationUs
		}
		return nil
	}

	// Merged mode: correlate CPU stacks with GPU kernels
	runtimes := c.gpuParser.GetRuntimes()
	kernels := c.gpuParser.GetKernels()

	if len(c.cpuTraces) == 0 {
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
	sort.Slice(c.cpuTraces, func(i, j int) bool {
		return c.cpuTraces[i].Timestamp < c.cpuTraces[j].Timestamp
	})

	// Sort runtimes by timestamp
	sort.Slice(runtimes, func(i, j int) bool {
		return runtimes[i].StartNs < runtimes[j].StartNs
	})

	toleranceNs := int64(c.toleranceMs * 1e6)
	matched := 0
	unmatched := 0

	// Try sequential matching with time window validation
	if len(c.cpuTraces) == len(runtimes) {
		// Perfect 1:1 correspondence
		for i, cpuTrace := range c.cpuTraces {
			runtime := &runtimes[i]
			kernel := kernelByCorr[runtime.CorrelationID]

			if kernel != nil {
				merged := cpuTrace.Stack
				merged = append(merged, fmt.Sprintf("[GPU_Kernel]%s", kernel.Name))
				stackStr := strings.Join(merged, ";")
				durationUs := (kernel.EndNs - kernel.StartNs) / 1000
				c.mergedStacks[stackStr] += durationUs
				matched++
			} else {
				unmatched++
			}
		}
	} else {
		// More events on one side - use time-based matching
		gpuIdx := 0

		for _, cpuTrace := range c.cpuTraces {
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
					c.mergedStacks[stackStr] += durationUs
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

// symbolCache caches ELF symbol maps to avoid repeated parsing
var symbolCache = struct {
	sync.RWMutex
	cache map[string]*libpf.SymbolMap
}{
	cache: make(map[string]*libpf.SymbolMap),
}

// pidExeCache caches PID -> executable path mappings
var pidExeCache = struct {
	sync.RWMutex
	cache map[int]string
}{
	cache: make(map[int]string),
}

// resolveExecutablePath resolves a basename to full path using /proc/<pid>/exe
func resolveExecutablePath(pid int, basename string) string {
	pidExeCache.RLock()
	cached, found := pidExeCache.cache[pid]
	pidExeCache.RUnlock()

	if found {
		if filepath.Base(cached) == basename {
			return cached
		}
	}

	// Read /proc/<pid>/exe symlink
	exePath, err := os.Readlink(fmt.Sprintf("/proc/%d/exe", pid))
	if err == nil {
		pidExeCache.Lock()
		pidExeCache.cache[pid] = exePath
		pidExeCache.Unlock()

		if filepath.Base(exePath) == basename {
			return exePath
		}
	}

	// Fall back to basename (will fail to open)
	return basename
}

// lookupSymbol tries to resolve an address to a function name using ELF symbols
func lookupSymbol(fileName string, fileOffset uint64, debugInfo string) string {
	// Try to get from cache first
	symbolCache.RLock()
	symmap, cached := symbolCache.cache[fileName]
	symbolCache.RUnlock()

	if !cached {
		// Load ELF file using pfelf
		elfFile, err := pfelf.Open(fileName)
		if err != nil {
			return ""
		}
		defer elfFile.Close()

		// Try regular symbol table first (for non-stripped binaries)
		symmap, err = elfFile.ReadSymbols()
		if err != nil || symmap == nil || symmap.Len() == 0 {
			// Fall back to dynamic symbols
			symmap, err = elfFile.ReadDynamicSymbols()
			if err != nil || symmap == nil {
				return ""
			}
		}

		// Cache the symbol map
		symbolCache.Lock()
		symbolCache.cache[fileName] = symmap
		symbolCache.Unlock()
	}

	// Use SymbolMap's LookupByAddress for efficient symbol resolution
	symbolName, offset, found := symmap.LookupByAddress(libpf.SymbolValue(fileOffset))
	if found && symbolName != "" {
		if offset == 0 {
			return string(symbolName)
		}
		return fmt.Sprintf("%s+0x%x", symbolName, offset)
	}

	return ""
}

// ExtractStackFromTrace extracts function names from a libpf.Trace
func ExtractStackFromTrace(trace *libpf.Trace, meta *samples.TraceEventMeta) []string {
	stack := make([]string, 0, len(trace.Frames))

	// Iterate in reverse order: frames are stored innermost-first,
	// but flamegraph format needs outermost-first (root to leaf)
	for i := len(trace.Frames) - 1; i >= 0; i-- {
		frame := trace.Frames[i].Value()

		functionName := frame.FunctionName.String()
		frameName := ""

		// Try to get a meaningful frame identifier
		if functionName != "" && functionName != "<unknown>" {
			frameName = functionName
		} else if frame.MappingFile.Valid() {
			fileData := frame.MappingFile.Value()
			fileName := fileData.FileName.String()
			if fileName != "" {
				baseName := filepath.Base(fileName)

				// AddressOrLineno already contains the file offset (relative to segment base)
				// For PIE binaries, this matches the ELF symbol values directly
				symbolAddress := uint64(frame.AddressOrLineno)

				// Resolve basename to full path using PID
				fullPath := resolveExecutablePath(int(meta.PID), baseName)

				// Try to resolve symbol from ELF using the address
				debugInfo := fmt.Sprintf("addr=0x%x,start=0x%x",
					frame.AddressOrLineno, frame.MappingStart)
				symbol := lookupSymbol(fullPath, symbolAddress, debugInfo)
				if symbol != "" {
					frameName = symbol
				} else {
					// Fall back to file+offset
					// For display, show offset from mapping start
					displayOffset := symbolAddress
					if frame.MappingStart > 0 {
						displayOffset = uint64(libpf.Address(frame.AddressOrLineno) - frame.MappingStart)
					}
					frameName = fmt.Sprintf("%s+0x%x", baseName, displayOffset)
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
