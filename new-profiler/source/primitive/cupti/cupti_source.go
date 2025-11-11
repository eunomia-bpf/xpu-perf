package cupti

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"new-profiler/event"
	"os"
)

// CUPTISource reads GPU events from CUPTI pipe
type CUPTISource struct {
	pipePath string
	pipe     *os.File
}

// JSONEvent represents a CUPTI trace event in JSON format
type JSONEvent struct {
	Type          string `json:"type"`
	Start         int64  `json:"start"`
	End           int64  `json:"end"`
	Duration      int64  `json:"duration"`
	Name          string `json:"name"`
	CorrelationID int    `json:"correlationId"`
	ProcessID     int    `json:"processId"`
	ThreadID      int    `json:"threadId"`
}

func NewCUPTISource(pipePath string) *CUPTISource {
	return &CUPTISource{
		pipePath: pipePath,
	}
}

func (s *CUPTISource) Name() string {
	return "cupti"
}

func (s *CUPTISource) Initialize() error {
	// Pipe will be opened when Start is called
	return nil
}

func (s *CUPTISource) Start(ctx context.Context, events chan<- *event.Event) error {
	// Start reading events in background
	// Note: Opening the pipe must happen in the goroutine to avoid blocking
	go func() {
		// Open the named pipe for reading (this will block until writer opens)
		pipe, err := os.OpenFile(s.pipePath, os.O_RDONLY, os.ModeNamedPipe)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Failed to open CUPTI pipe: %v\n", err)
			return
		}
		s.pipe = pipe
		defer pipe.Close()

		scanner := bufio.NewScanner(pipe)
		scanner.Buffer(make([]byte, 1024*1024), 1024*1024) // 1MB buffer for long lines

		for scanner.Scan() {
			select {
			case <-ctx.Done():
				return
			default:
				line := scanner.Text()
				if line == "" {
					continue
				}

				// Parse JSON event
				var jsonEvent JSONEvent
				if err := json.Unmarshal([]byte(line), &jsonEvent); err != nil {
					// Skip invalid JSON lines
					continue
				}

				// Convert to our Event type
				evt := s.convertJSONEvent(&jsonEvent)
				if evt != nil {
					// Debug: Log first few GPU events
					if evt.Type == event.EventGPUKernel {
						gpuData := evt.Data.(*event.GPUKernelData)
						fmt.Fprintf(os.Stderr, "[CUPTI] GPU kernel: %s, correlationID=%d\n",
							gpuData.KernelName, evt.CorrelationID)
					}
					if events != nil {
						events <- evt
					}
				}
			}
		}

		if err := scanner.Err(); err != nil {
			// Log error but don't crash
			fmt.Fprintf(os.Stderr, "CUPTI scanner error: %v\n", err)
		}
	}()

	return nil
}

func (s *CUPTISource) Stop() error {
	if s.pipe != nil {
		return s.pipe.Close()
	}
	return nil
}

// convertJSONEvent converts a CUPTI JSON event to our Event type
func (s *CUPTISource) convertJSONEvent(jsonEvent *JSONEvent) *event.Event {
	// Process CONCURRENT_KERNEL events
	if jsonEvent.Type == "CONCURRENT_KERNEL" {
		return &event.Event{
			Timestamp:     jsonEvent.Start,
			Source:        "cupti",
			Type:          event.EventGPUKernel,
			CorrelationID: uint32(jsonEvent.CorrelationID),
			ProcessID:     uint32(jsonEvent.ProcessID),
			ThreadID:      uint32(jsonEvent.ThreadID),
			Data: &event.GPUKernelData{
				KernelName: jsonEvent.Name,
				Duration:   jsonEvent.End - jsonEvent.Start,
			},
		}
	}

	// Process RUNTIME events (if needed for debugging)
	if jsonEvent.Type == "RUNTIME" {
		return &event.Event{
			Timestamp:     jsonEvent.Start,
			Source:        "cupti",
			Type:          event.EventGPURuntime,
			CorrelationID: uint32(jsonEvent.CorrelationID),
			ProcessID:     uint32(jsonEvent.ProcessID),
			ThreadID:      uint32(jsonEvent.ThreadID),
			Data: &event.GPURuntimeData{
				APIName:  jsonEvent.Name,
				Duration: jsonEvent.End - jsonEvent.Start,
			},
		}
	}

	return nil
}
