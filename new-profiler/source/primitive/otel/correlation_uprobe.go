// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package otel

import (
	"fmt"
	"strings"

	"github.com/cilium/ebpf"
	"github.com/cilium/ebpf/link"
	"go.opentelemetry.io/ebpf-profiler/tracer"
)

//go:generate go run github.com/cilium/ebpf/cmd/bpf2go -target amd64 -cflags "-I/usr/include -I/usr/include/x86_64-linux-gnu" CorrelationUprobe ebpf/correlation_uprobe.ebpf.c

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

// AttachCorrelationUprobe attaches a uprobe to XpuPerfGetCorrelationId in the CUPTI library
// to capture correlation IDs and pass them to the custom trace handler
func AttachCorrelationUprobe(trc *tracer.Tracer, cuptiLibPath string) (*CorrelationUprobeState, error) {
	// Get the custom__generic program FD from the tracer
	customProgFD := trc.GetCustomTraceProgramFD()
	if customProgFD < 0 {
		return nil, fmt.Errorf("custom__generic program not loaded")
	}
	fmt.Printf("Got custom__generic program FD: %d\n", customProgFD)

	// Get the custom_context_map FD for sharing context values
	customContextMapFD := trc.GetCustomContextMapFD()
	if customContextMapFD < 0 {
		return nil, fmt.Errorf("custom_context_map not loaded")
	}
	fmt.Printf("Got custom_context_map FD: %d\n", customContextMapFD)

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
	opts := &ebpf.CollectionOptions{
		MapReplacements: map[string]*ebpf.Map{
			"custom_context_map": customContextMap,
		},
	}

	// Load the eBPF collection
	// IMPORTANT: Do NOT defer coll.Close() here! We need to keep it alive
	// for the uprobe to work.
	coll, err := ebpf.NewCollectionWithOptions(spec, *opts)
	if err != nil {
		return nil, fmt.Errorf("failed to load correlation uprobe collection: %v", err)
	}

	// Add custom__generic to our prog_array at index 0 for tail calling
	progArray := coll.Maps["prog_array"]
	if progArray == nil {
		return nil, fmt.Errorf("prog_array not found in correlation uprobe collection")
	}

	key := uint32(0)
	value := uint32(customProgFD)
	if err := progArray.Put(&key, &value); err != nil {
		return nil, fmt.Errorf("failed to add custom__generic to prog_array: %v", err)
	}
	fmt.Println("Successfully added custom__generic to prog_array at index 0")

	// Extract library path
	libPath := cuptiLibPath
	if strings.Contains(libPath, "libcupti_trace") {
		fmt.Printf("Attaching correlation uprobe to: %s:XpuPerfGetCorrelationId\n", libPath)
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

	fmt.Printf("Successfully attached correlation uprobe to %s:XpuPerfGetCorrelationId\n", libPath)
	fmt.Println("The uprobe will capture correlation IDs and tail call to custom__generic")

	// Return state that keeps both the uprobe link and collection alive
	return &CorrelationUprobeState{
		Link:       uprobe,
		Collection: coll,
	}, nil
}
