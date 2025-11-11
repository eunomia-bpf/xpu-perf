package output

import (
	"fmt"
	"new-profiler/correlation"
	"os"
	"sort"
	"strings"
)

// WriteFoldedStacks writes correlated events in folded stack format for flamegraph
// GPU kernel durations are converted to sample counts using the CPU sampling frequency
// Also writes CPU sampling stacks (pure CPU activity without GPU correlation)
func WriteFoldedStacks(filename string, merged []*correlation.MergedEvent, cpuSamplingStacks map[string]int, samplesPerSec int) error {
	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create output file: %w", err)
	}
	defer file.Close()

	// Count occurrences of each unique stack
	// Use float64 for sample counts to avoid rounding errors during accumulation
	stackCounts := make(map[string]float64)

	// Add CPU sampling stacks (pure CPU activity)
	for stack, count := range cpuSamplingStacks {
		stackCounts[stack] += float64(count)
	}

	// Add correlated CPU+GPU stacks
	for _, m := range merged {
		// Build folded stack: CPU stack ; GPU kernel
		cpuStack := strings.Join(m.CPUStack, ";")
		fullStack := cpuStack
		if m.GPUKernel != "" {
			fullStack = cpuStack + ";[GPU_Kernel]" + m.GPUKernel
		}

		// Convert GPU duration (microseconds) to sample count using CPU sampling frequency
		// Formula: samples = (durationUs / 1e6) * samplesPerSec
		// Or: samples = (durationUs * samplesPerSec) / 1e6
		// Or in nanoseconds: samples = (durationNs * samplesPerSec) / 1e9

		// DurationUs is in microseconds, convert to seconds then multiply by samples/sec
		durationSec := float64(m.DurationUs) / 1e6
		samples := durationSec * float64(samplesPerSec)

		stackCounts[fullStack] += samples
	}

	// Sort stacks for consistent output
	type stackEntry struct {
		stack string
		count float64
	}
	entries := make([]stackEntry, 0, len(stackCounts))
	for stack, count := range stackCounts {
		entries = append(entries, stackEntry{stack, count})
	}
	sort.Slice(entries, func(i, j int) bool {
		return entries[i].stack < entries[j].stack
	})

	// Write to file
	// Round float counts to integers for flamegraph compatibility
	// Ensure entries with any samples (even fractional) appear with at least 1 count
	totalSamples := int64(0)
	for _, entry := range entries {
		var roundedCount int64
		if entry.count > 0 {
			roundedCount = int64(entry.count + 0.5) // Round to nearest integer
			if roundedCount == 0 {
				roundedCount = 1 // Ensure at least 1 if any samples were recorded
			}
		}
		if roundedCount > 0 {
			fmt.Fprintf(file, "%s %d\n", entry.stack, roundedCount)
			totalSamples += roundedCount
		}
	}

	fmt.Printf("Wrote %d unique stacks (%d total samples) to %s\n", len(entries), totalSamples, filename)
	return nil
}

// WriteFoldedStacksFromMap writes CPU stacks from a map (for uprobe-only mode)
func WriteFoldedStacksFromMap(filename string, stackCounts map[string]int) error {
	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create output file: %w", err)
	}
	defer file.Close()

	// Sort stacks for consistent output
	type stackEntry struct {
		stack string
		count int
	}
	entries := make([]stackEntry, 0, len(stackCounts))
	for stack, count := range stackCounts {
		entries = append(entries, stackEntry{stack, count})
	}
	sort.Slice(entries, func(i, j int) bool {
		return entries[i].stack < entries[j].stack
	})

	// Write to file
	totalSamples := 0
	for _, entry := range entries {
		if entry.count > 0 {
			fmt.Fprintf(file, "%s %d\n", entry.stack, entry.count)
			totalSamples += entry.count
		}
	}

	fmt.Printf("Wrote %d unique stacks (%d total samples) to %s\n", len(entries), totalSamples, filename)
	return nil
}
