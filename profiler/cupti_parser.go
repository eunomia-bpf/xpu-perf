// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"
	"sync"
)

// GPUKernelEvent represents a kernel execution event from CUPTI
type GPUKernelEvent struct {
	Name          string
	StartNs       int64
	EndNs         int64
	CorrelationID int
}

// GPURuntimeEvent represents a CUDA runtime API call event
type GPURuntimeEvent struct {
	Name          string
	StartNs       int64
	EndNs         int64
	CorrelationID int
	ProcessID     int
	ThreadID      int
}

// CUPTIParser parses CUPTI trace output from a named pipe
type CUPTIParser struct {
	kernels  []GPUKernelEvent
	runtimes []GPURuntimeEvent
	mu       sync.Mutex
}

// NewCUPTIParser creates a new CUPTI parser
func NewCUPTIParser() *CUPTIParser {
	return &CUPTIParser{
		kernels:  make([]GPUKernelEvent, 0),
		runtimes: make([]GPURuntimeEvent, 0),
	}
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

// ParseStream reads and parses CUPTI trace from a reader (pipe or file)
func (p *CUPTIParser) ParseStream(reader io.Reader) error {
	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024) // 1MB buffer for long lines

	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			continue
		}

		// Parse JSON event
		var event JSONEvent
		if err := json.Unmarshal([]byte(line), &event); err != nil {
			// Skip invalid JSON lines
			continue
		}

		// Process CONCURRENT_KERNEL events
		if event.Type == "CONCURRENT_KERNEL" {
			kernel := GPUKernelEvent{
				Name:          event.Name,
				StartNs:       event.Start,
				EndNs:         event.End,
				CorrelationID: event.CorrelationID,
			}

			p.mu.Lock()
			p.kernels = append(p.kernels, kernel)
			p.mu.Unlock()
		}

		// Process RUNTIME events (especially cudaLaunchKernel)
		if event.Type == "RUNTIME" && strings.Contains(event.Name, "LaunchKernel") {
			runtime := GPURuntimeEvent{
				Name:          event.Name,
				StartNs:       event.Start,
				EndNs:         event.End,
				CorrelationID: event.CorrelationID,
				ProcessID:     event.ProcessID,
				ThreadID:      event.ThreadID,
			}

			p.mu.Lock()
			p.runtimes = append(p.runtimes, runtime)
			p.mu.Unlock()
		}
	}

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("error reading CUPTI trace: %v", err)
	}

	return nil
}

// ParseFile parses CUPTI trace from a file
func (p *CUPTIParser) ParseFile(filename string) error {
	file, err := os.Open(filename)
	if err != nil {
		return fmt.Errorf("failed to open file: %v", err)
	}
	defer file.Close()

	return p.ParseStream(file)
}

// GetKernels returns all parsed kernel events
func (p *CUPTIParser) GetKernels() []GPUKernelEvent {
	p.mu.Lock()
	defer p.mu.Unlock()
	return append([]GPUKernelEvent{}, p.kernels...)
}

// GetRuntimes returns all parsed runtime events
func (p *CUPTIParser) GetRuntimes() []GPURuntimeEvent {
	p.mu.Lock()
	defer p.mu.Unlock()
	return append([]GPURuntimeEvent{}, p.runtimes...)
}

// GetKernelByCorrelationID finds a kernel by correlation ID
func (p *CUPTIParser) GetKernelByCorrelationID(corrID int) *GPUKernelEvent {
	p.mu.Lock()
	defer p.mu.Unlock()

	for i := range p.kernels {
		if p.kernels[i].CorrelationID == corrID {
			return &p.kernels[i]
		}
	}
	return nil
}

// GetRuntimeByCorrelationID finds a runtime event by correlation ID
func (p *CUPTIParser) GetRuntimeByCorrelationID(corrID int) *GPURuntimeEvent {
	p.mu.Lock()
	defer p.mu.Unlock()

	for i := range p.runtimes {
		if p.runtimes[i].CorrelationID == corrID {
			return &p.runtimes[i]
		}
	}
	return nil
}
