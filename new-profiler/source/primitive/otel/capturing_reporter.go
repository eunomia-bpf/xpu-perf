package otel

import (
	"fmt"
	"new-profiler/event"
	"new-profiler/symbolizer"
	"path/filepath"
	"strings"
	"sync"

	"go.opentelemetry.io/ebpf-profiler/libpf"
	"go.opentelemetry.io/ebpf-profiler/reporter"
	"go.opentelemetry.io/ebpf-profiler/reporter/samples"
	"go.opentelemetry.io/ebpf-profiler/support"
)

// CapturingReporter captures symbolized traces and converts them to events
type CapturingReporter struct {
	events chan<- *event.Event
	mu     sync.Mutex
}

func NewCapturingReporter(events chan<- *event.Event) *CapturingReporter {
	return &CapturingReporter{
		events: events,
	}
}

// SetEventsChannel updates the events channel (thread-safe)
func (r *CapturingReporter) SetEventsChannel(events chan<- *event.Event) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.events = events
}

// Debug counter for first few events
var eventDebugCount = 0

// ReportTraceEvent is called by the tracer with fully symbolized trace information
func (r *CapturingReporter) ReportTraceEvent(trace *libpf.Trace, meta *samples.TraceEventMeta) error {
	r.mu.Lock()
	events := r.events
	r.mu.Unlock()

	if events == nil {
		return nil
	}

	// Debug first few events
	eventDebugCount++
	if eventDebugCount <= 10 {
		fmt.Printf("[CapturingReporter DEBUG #%d] Origin=%d ContextValue=%d PID=%d TID=%d\n",
			eventDebugCount, meta.Origin, meta.ContextValue, meta.PID, meta.TID)
	}

	// Extract stack from symbolized trace
	stack := r.extractStackFromTrace(trace, meta)

	// Determine trace type and correlation ID
	isUprobe := meta.Origin == support.TraceOriginUProbe ||
		meta.Origin == support.TraceOriginCustom
	isSampling := meta.Origin == support.TraceOriginSampling

	correlationID := uint32(0)
	if isUprobe || meta.Origin == support.TraceOriginCustom {
		// Custom traces have correlation ID in ContextValue
		correlationID = uint32(meta.ContextValue)
		if correlationID != 0 {
			fmt.Printf("[CapturingReporter] Got correlation ID %d from Origin=%d\n", correlationID, meta.Origin)
		}
	}

	// Create event
	evt := &event.Event{
		Timestamp:     int64(meta.Timestamp),
		Source:        "otel",
		Type:          event.EventCPUTrace,
		CorrelationID: correlationID,
		ThreadID:      uint32(meta.TID),
		ProcessID:     uint32(meta.PID),
		Data: &event.CPUTraceData{
			Stack:      stack,
			CPU:        meta.CPU,
			Comm:       meta.Comm.String(),
			IsUprobe:   isUprobe,
			IsSampling: isSampling,
		},
	}

	// Send event (non-blocking)
	select {
	case events <- evt:
	default:
		// Channel full, drop event
	}

	return nil
}

// extractStackFromTrace extracts and symbolizes stack frames from a trace
// This uses the symbolized frame information provided by the tracer
func (r *CapturingReporter) extractStackFromTrace(trace *libpf.Trace, meta *samples.TraceEventMeta) []string {
	stack := make([]string, 0, len(trace.Frames))
	sym := symbolizer.GetSymbolizer()

	// Iterate in reverse order: frames are stored innermost-first,
	// but flamegraph format needs outermost-first (root to leaf)
	for i := len(trace.Frames) - 1; i >= 0; i-- {
		frame := trace.Frames[i].Value()

		functionName := frame.FunctionName.String()
		frameName := ""

		// Try to get a meaningful frame identifier
		if functionName != "" && functionName != "<unknown>" {
			// Demangle if needed
			frameName = sym.Demangle(functionName)
		} else if frame.MappingFile.Valid() {
			fileData := frame.MappingFile.Value()
			fileName := fileData.FileName.String()
			if fileName != "" {
				baseName := filepath.Base(fileName)
				fileOffset := uint64(frame.AddressOrLineno)

				// Resolve basename to full path using PID
				fullPath := resolveExecutablePath(int(meta.PID), baseName)

				// Try to resolve symbol from ELF using the file offset
				symbol := sym.Symbolize(fullPath, fileOffset)
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

// resolveExecutablePath tries to resolve a basename to a full executable path
func resolveExecutablePath(pid int, basename string) string {
	// Try /proc/pid/exe symlink
	exePath := fmt.Sprintf("/proc/%d/exe", pid)
	if target, err := filepath.EvalSymlinks(exePath); err == nil {
		if filepath.Base(target) == basename {
			return target
		}
	}

	// Try common library paths
	commonPaths := []string{
		"/lib/x86_64-linux-gnu",
		"/usr/lib/x86_64-linux-gnu",
		"/lib64",
		"/usr/lib64",
		"/usr/local/lib",
	}

	for _, dir := range commonPaths {
		fullPath := filepath.Join(dir, basename)
		if _, err := filepath.EvalSymlinks(fullPath); err == nil {
			return fullPath
		}
	}

	// Fall back to basename
	return basename
}

// ReportFramesForTrace is called to report frame information
func (r *CapturingReporter) ReportFramesForTrace(_ *libpf.Trace) {
	// We handle everything in ReportTraceEvent
}

// ReportCountForTrace is called to report sample counts
func (r *CapturingReporter) ReportCountForTrace(_ *libpf.Trace, _ uint16) {
	// We handle everything in ReportTraceEvent
}

// ExecutableKnown checks if an executable is known
func (r *CapturingReporter) ExecutableKnown(_ libpf.FileID, _ string) bool {
	return true
}

// ExecutableReports triggers reporting of executables
func (r *CapturingReporter) ExecutableReports() {
	// Not needed for our use case
}

// ReportExecutable reports executable metadata
func (r *CapturingReporter) ReportExecutable(_ *reporter.ExecutableMetadata) {
	// Not needed for our use case
}
