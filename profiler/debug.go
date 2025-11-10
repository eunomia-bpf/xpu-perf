// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"fmt"
	"os"
	"path/filepath"
)

// writeDebugOutput writes debug files for CUPTI events, CPU traces, and correlation details
func (c *TraceCorrelator) writeDebugOutput() error {
	// Create debug directory if it doesn't exist
	if err := os.MkdirAll(c.debugDir, 0755); err != nil {
		return fmt.Errorf("failed to create debug directory: %v", err)
	}

	// 1. Write CUPTI GPU events
	if c.gpuParser != nil {
		if err := c.writeCUPTIDebug(); err != nil {
			return fmt.Errorf("failed to write CUPTI debug: %v", err)
		}
	}

	// 2. Write CPU traces (both uprobes and sampling)
	if err := c.writeCPUTracesDebug(); err != nil {
		return fmt.Errorf("failed to write CPU traces debug: %v", err)
	}

	// 3. Write correlation details
	if err := c.writeCorrelationDebug(); err != nil {
		return fmt.Errorf("failed to write correlation debug: %v", err)
	}

	fmt.Printf("Debug output written to: %s\n", c.debugDir)
	return nil
}

// writeCUPTIDebug writes CUPTI GPU events to debug files
func (c *TraceCorrelator) writeCUPTIDebug() error {
	runtimes := c.gpuParser.GetRuntimes()
	kernels := c.gpuParser.GetKernels()

	// Write runtime events
	runtimeFile := filepath.Join(c.debugDir, "cupti_runtimes.txt")
	f, err := os.Create(runtimeFile)
	if err != nil {
		return err
	}
	defer f.Close()

	fmt.Fprintf(f, "# CUPTI Runtime Events (Total: %d)\n", len(runtimes))
	fmt.Fprintf(f, "# Format: CorrelationID | StartNs | EndNs | Name\n")
	fmt.Fprintf(f, "#========================================\n")
	for i, rt := range runtimes {
		fmt.Fprintf(f, "[%d] CorrID=%d | Start=%d | End=%d | %s\n",
			i, rt.CorrelationID, rt.StartNs, rt.EndNs, rt.Name)
	}

	// Write kernel events
	kernelFile := filepath.Join(c.debugDir, "cupti_kernels.txt")
	f2, err := os.Create(kernelFile)
	if err != nil {
		return err
	}
	defer f2.Close()

	fmt.Fprintf(f2, "# CUPTI Kernel Events (Total: %d)\n", len(kernels))
	fmt.Fprintf(f2, "# Format: CorrelationID | StartNs | EndNs | DurationUs | Name\n")
	fmt.Fprintf(f2, "#========================================\n")
	for i, k := range kernels {
		durationUs := (k.EndNs - k.StartNs) / 1000
		fmt.Fprintf(f2, "[%d] CorrID=%d | Start=%d | End=%d | Duration=%dus | %s\n",
			i, k.CorrelationID, k.StartNs, k.EndNs, durationUs, k.Name)
	}

	return nil
}

// writeCPUTracesDebug writes CPU traces (uprobes and sampling) to debug files
func (c *TraceCorrelator) writeCPUTracesDebug() error {
	// Separate uprobes and sampling traces
	var uprobes []CPUStackTrace
	var sampling []CPUStackTrace

	for _, trace := range c.cpuTraces {
		if trace.IsUprobe {
			uprobes = append(uprobes, trace)
		} else {
			sampling = append(sampling, trace)
		}
	}

	// Write uprobe traces
	uprobeFile := filepath.Join(c.debugDir, "cpu_uprobes.txt")
	f, err := os.Create(uprobeFile)
	if err != nil {
		return err
	}
	defer f.Close()

	fmt.Fprintf(f, "# CPU Uprobe Traces (Total: %d)\n", len(uprobes))
	fmt.Fprintf(f, "# Format: Index | Timestamp | PID | TID | CPU | Comm | Stack\n")
	fmt.Fprintf(f, "#========================================\n")
	for i, trace := range uprobes {
		fmt.Fprintf(f, "[%d] Timestamp=%d | PID=%d | TID=%d | CPU=%d | Comm=%s\n",
			i, trace.Timestamp, trace.PID, trace.TID, trace.CPU, trace.Comm)
		fmt.Fprintf(f, "    Stack (%d frames):\n", len(trace.Stack))
		for j, frame := range trace.Stack {
			fmt.Fprintf(f, "      [%d] %s\n", j, frame)
		}
		fmt.Fprintf(f, "\n")
	}

	// Write sampling traces
	samplingFile := filepath.Join(c.debugDir, "cpu_sampling.txt")
	f2, err := os.Create(samplingFile)
	if err != nil {
		return err
	}
	defer f2.Close()

	fmt.Fprintf(f2, "# CPU Sampling Traces (Total: %d)\n", len(sampling))
	fmt.Fprintf(f2, "# Format: Index | Timestamp | PID | TID | CPU | Comm | Stack\n")
	fmt.Fprintf(f2, "#========================================\n")
	for i, trace := range sampling {
		fmt.Fprintf(f2, "[%d] Timestamp=%d | PID=%d | TID=%d | CPU=%d | Comm=%s\n",
			i, trace.Timestamp, trace.PID, trace.TID, trace.CPU, trace.Comm)
		fmt.Fprintf(f2, "    Stack (%d frames):\n", len(trace.Stack))
		for j, frame := range trace.Stack {
			fmt.Fprintf(f2, "      [%d] %s\n", j, frame)
		}
		fmt.Fprintf(f2, "\n")
	}

	return nil
}

// writeCorrelationDebug writes detailed correlation analysis
func (c *TraceCorrelator) writeCorrelationDebug() error {
	corrFile := filepath.Join(c.debugDir, "correlation_details.txt")
	f, err := os.Create(corrFile)
	if err != nil {
		return err
	}
	defer f.Close()

	fmt.Fprintf(f, "# Correlation Analysis\n")
	fmt.Fprintf(f, "#========================================\n")
	fmt.Fprintf(f, "Mode: ")
	if c.cpuOnly {
		fmt.Fprintf(f, "CPU-only\n")
	} else if c.gpuOnly {
		fmt.Fprintf(f, "GPU-only (uprobe correlation)\n")
	} else if c.mergeMode {
		fmt.Fprintf(f, "Merge mode (CPU sampling + GPU)\n")
	}
	fmt.Fprintf(f, "Tolerance: %.2f ms\n", c.toleranceMs)
	fmt.Fprintf(f, "Sampling rate: %d Hz\n", c.samplesPerSec)
	fmt.Fprintf(f, "\n")

	if c.gpuParser != nil {
		runtimes := c.gpuParser.GetRuntimes()
		kernels := c.gpuParser.GetKernels()

		fmt.Fprintf(f, "CUPTI Events:\n")
		fmt.Fprintf(f, "  Runtime events: %d\n", len(runtimes))
		fmt.Fprintf(f, "  Kernel events: %d\n", len(kernels))
	}

	// Count traces by type
	uprobeCount := 0
	samplingCount := 0
	for _, trace := range c.cpuTraces {
		if trace.IsUprobe {
			uprobeCount++
		} else {
			samplingCount++
		}
	}

	fmt.Fprintf(f, "\nCPU Traces:\n")
	fmt.Fprintf(f, "  Uprobe traces: %d\n", uprobeCount)
	fmt.Fprintf(f, "  Sampling traces: %d\n", samplingCount)
	fmt.Fprintf(f, "  Total: %d\n", len(c.cpuTraces))
	fmt.Fprintf(f, "\n")

	// Analyze stack depths
	if uprobeCount > 0 {
		minDepth, maxDepth, avgDepth := c.analyzeStackDepths(true)
		fmt.Fprintf(f, "Uprobe Stack Depths:\n")
		fmt.Fprintf(f, "  Min: %d frames\n", minDepth)
		fmt.Fprintf(f, "  Max: %d frames\n", maxDepth)
		fmt.Fprintf(f, "  Avg: %.2f frames\n", avgDepth)
		fmt.Fprintf(f, "\n")
	}

	if samplingCount > 0 {
		minDepth, maxDepth, avgDepth := c.analyzeStackDepths(false)
		fmt.Fprintf(f, "Sampling Stack Depths:\n")
		fmt.Fprintf(f, "  Min: %d frames\n", minDepth)
		fmt.Fprintf(f, "  Max: %d frames\n", maxDepth)
		fmt.Fprintf(f, "  Avg: %.2f frames\n", avgDepth)
		fmt.Fprintf(f, "\n")
	}

	return nil
}

// analyzeStackDepths returns min, max, and average stack depths for uprobe or sampling traces
func (c *TraceCorrelator) analyzeStackDepths(isUprobe bool) (int, int, float64) {
	min := int(^uint(0) >> 1) // max int
	max := 0
	total := 0
	count := 0

	for _, trace := range c.cpuTraces {
		if trace.IsUprobe == isUprobe {
			depth := len(trace.Stack)
			if depth < min {
				min = depth
			}
			if depth > max {
				max = depth
			}
			total += depth
			count++
		}
	}

	if count == 0 {
		return 0, 0, 0
	}

	avg := float64(total) / float64(count)
	return min, max, avg
}
