// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"fmt"
	"time"

	log "github.com/sirupsen/logrus"
	"go.opentelemetry.io/ebpf-profiler/tracer"
)

// CudaKernelUprobeAttacher implements UprobeAttacher for CUDA kernel launch uprobes
type CudaKernelUprobeAttacher struct{}

// CudaKernelUprobeState holds the state for CUDA kernel uprobes
// The actual uprobe management is done by the tracer, so this is mostly a placeholder
type CudaKernelUprobeState struct {
	tracer *tracer.Tracer
}

// Close releases all resources (managed by tracer)
func (s *CudaKernelUprobeState) Close() error {
	// The tracer manages the uprobe lifecycle, nothing to do here
	return nil
}

// Attach attaches uprobes to CUDA kernel launch functions
func (a *CudaKernelUprobeAttacher) Attach(trc *tracer.Tracer, cfg *Config) (UprobeState, error) {
	// Determine uprobes based on mode
	var uprobes []string
	if cfg.gpuOnly || (!cfg.cpuOnly && !cfg.gpuOnly) {
		// GPU-only and merge mode use uprobes for CPU/GPU correlation
		// Default to CUDA runtime library for kernel launch tracing
		cudaLibPath := cfg.cudaLibPath
		if cudaLibPath == "" {
			// Try to detect CUDA library from target binary first
			cudaLibPath = detectCudaLibraryFromBinary(cfg.targetBinary)
			if cudaLibPath == "" {
				cudaLibPath = findCudaLibrary()
			}
		}
		// Attach to CUDA launch kernel symbols actually used by the target binary
		uprobes = buildUprobeList(cudaLibPath, cfg.targetBinary)
	}

	// Attach uprobes if any
	if len(uprobes) > 0 {
		log.Infof("Attaching %d CUDA kernel launch uprobes globally (will trigger on any process using CUDA library)", len(uprobes))
		for i, probe := range uprobes {
			log.Infof("  [%d] Attaching to: %s", i+1, probe)
		}
		if err := trc.AttachUProbes(uprobes); err != nil {
			return nil, fmt.Errorf("failed to attach CUDA kernel uprobes: %v", err)
		}
		log.Info("CUDA kernel uprobes attached successfully! They will fire when any process calls cudaLaunchKernel.")
		// Give uprobes time to become active in the kernel
		log.Info("Waiting 100ms for uprobes to become fully active...")
		time.Sleep(100 * time.Millisecond)
	} else {
		log.Info("No CUDA kernel uprobes to attach (CPU-only mode or no CUDA symbols found)")
	}

	return &CudaKernelUprobeState{
		tracer: trc,
	}, nil
}
