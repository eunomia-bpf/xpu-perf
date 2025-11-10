// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"fmt"
	"strings"

	"github.com/cilium/ebpf"
	"github.com/cilium/ebpf/link"
	log "github.com/sirupsen/logrus"
	"go.opentelemetry.io/ebpf-profiler/tracer"
)

// CorrelationUprobeState holds the state for the correlation uprobe
// We need to keep the collection alive to prevent prog_array from being destroyed
type CorrelationUprobeState struct {
	Link       link.Link
	Collection *ebpf.Collection
}

// Close releases all resources
func (s *CorrelationUprobeState) Close() error {
	if s.Link != nil {
		s.Link.Close()
	}
	if s.Collection != nil {
		s.Collection.Close()
	}
	return nil
}

// attachCorrelationUprobe attaches a uprobe to XpuPerfGetCorrelationId in the CUPTI library
// to capture correlation IDs and pass them to the custom trace handler
func attachCorrelationUprobe(trc *tracer.Tracer, cuptiLibPath string) (*CorrelationUprobeState, error) {
	// Get the custom__generic program FD from the tracer
	customProgFD := trc.GetCustomTraceProgramFD()
	if customProgFD < 0 {
		return nil, fmt.Errorf("custom__generic program not loaded")
	}
	log.Infof("Got custom__generic program FD: %d", customProgFD)

	// Get the custom_context_map FD for sharing context values
	customContextMapFD := trc.GetCustomContextMapFD()
	if customContextMapFD < 0 {
		return nil, fmt.Errorf("custom_context_map not loaded")
	}
	log.Infof("Got custom_context_map FD: %d", customContextMapFD)

	// Note: We create our own prog_array in the eBPF program, not reusing from profiler

	// Load our correlation uprobe program spec
	spec, err := LoadCorrelationUprobe()
	if err != nil {
		return nil, fmt.Errorf("failed to load correlation uprobe spec: %v", err)
	}

	// Create Map objects from FDs to reuse the profiler's maps
	customContextMap, err := ebpf.NewMapFromFD(customContextMapFD)
	if err != nil {
		return nil, fmt.Errorf("failed to create custom_context_map from FD: %v", err)
	}
	defer customContextMap.Close()

	// Create collection options to reuse the profiler's custom_context_map
	// Note: We don't use prog_array - instead the uprobe just stores the correlation ID
	// and the regular sampling profiler will pick it up
	opts := &ebpf.CollectionOptions{
		MapReplacements: map[string]*ebpf.Map{
			"custom_context_map": customContextMap,
		},
	}

	// Load the eBPF collection
	// IMPORTANT: Do NOT defer coll.Close() here! We need to keep it alive
	// for the uprobe to work. The collection contains prog_array which is needed
	// for tail calls. If we close it, tail calls will fail.
	coll, err := ebpf.NewCollectionWithOptions(spec, *opts)
	if err != nil {
		return nil, fmt.Errorf("failed to load correlation uprobe collection: %v", err)
	}

	// Add custom__generic to our prog_array at index 0 for tail calling
	// This allows the uprobe to trigger stack trace collection
	progArray := coll.Maps["prog_array"]
	if progArray == nil {
		return nil, fmt.Errorf("prog_array not found in correlation uprobe collection")
	}

	key := uint32(0)
	value := uint32(customProgFD)
	if err := progArray.Put(&key, &value); err != nil {
		return nil, fmt.Errorf("failed to add custom__generic to prog_array: %v", err)
	}
	log.Info("Successfully added custom__generic to prog_array at index 0")

	// Extract library path from cuptiLibPath (e.g., /path/to/libcupti_trace_injection.so)
	// We need to attach to the main CUPTI library, not the injection library
	// The XpuPerfGetCorrelationId function is in the injection library
	libPath := cuptiLibPath
	if strings.Contains(libPath, "libcupti_trace") {
		log.Infof("Attaching correlation uprobe to: %s:XpuPerfGetCorrelationId", libPath)
	}

	// Attach uprobe to XpuPerfGetCorrelationId
	prog := coll.Programs["capture_correlation_id"]
	if prog == nil {
		return nil, fmt.Errorf("capture_correlation_id program not found in collection")
	}

	ex, err := link.OpenExecutable(libPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open executable %s: %v", libPath, err)
	}

	uprobe, err := ex.Uprobe("XpuPerfGetCorrelationId", prog, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to attach uprobe to XpuPerfGetCorrelationId: %v", err)
	}

	log.Infof("Successfully attached correlation uprobe to %s:XpuPerfGetCorrelationId", libPath)
	log.Info("The uprobe will capture correlation IDs and tail call to custom__generic")

	// Debug: Print collection and program info
	log.Infof("DEBUG: Collection programs: %d", len(coll.Programs))
	for name, prog := range coll.Programs {
		log.Infof("DEBUG: Program '%s' FD=%d", name, prog.FD())
	}
	log.Infof("DEBUG: Collection maps: %d", len(coll.Maps))
	for name, m := range coll.Maps {
		log.Infof("DEBUG: Map '%s' FD=%d", name, m.FD())
	}

	// Return state that keeps both the uprobe link and collection alive
	return &CorrelationUprobeState{
		Link:       uprobe,
		Collection: coll,
	}, nil
}
