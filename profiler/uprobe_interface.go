// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package main

import "go.opentelemetry.io/ebpf-profiler/tracer"

// UprobeAttacher defines the interface for attaching uprobes for GPU/CPU correlation
type UprobeAttacher interface {
	// Attach attaches the uprobe(s) and returns a state object that can be cleaned up
	Attach(trc *tracer.Tracer, cfg *Config) (UprobeState, error)
}

// UprobeState represents the state of attached uprobes
type UprobeState interface {
	// Close releases all resources associated with the uprobe(s)
	Close() error
}
