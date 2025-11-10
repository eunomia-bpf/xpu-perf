// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package main

//go:generate go run github.com/cilium/ebpf/cmd/bpf2go -target amd64 ExampleTailcall example_tailcall.c

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"time"

	"github.com/cilium/ebpf"
	"github.com/cilium/ebpf/link"
	log "github.com/sirupsen/logrus"
	"golang.org/x/sys/unix"

	"go.opentelemetry.io/ebpf-profiler/host"
	"go.opentelemetry.io/ebpf-profiler/libpf"
	"go.opentelemetry.io/ebpf-profiler/metrics"
	"go.opentelemetry.io/ebpf-profiler/times"
	"go.opentelemetry.io/ebpf-profiler/tracer"
	"go.opentelemetry.io/ebpf-profiler/tracer/types"
	"go.opentelemetry.io/otel/metric/noop"
)

func main() {
	if os.Geteuid() != 0 {
		fmt.Fprintf(os.Stderr, "Error: This profiler must be run as root\n")
		os.Exit(1)
	}

	if len(os.Args) < 2 {
		fmt.Fprintf(os.Stderr, "Usage: %s <executable:function>\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "\nExample:\n")
		fmt.Fprintf(os.Stderr, "  %s ./test_program:report_trace\n", os.Args[0])
		os.Exit(1)
	}

	targetFunction := os.Args[1]

	// Set up logging
	log.SetLevel(log.InfoLevel)
	log.SetFormatter(&log.TextFormatter{
		FullTimestamp: true,
	})

	log.Info("Starting custom trace example...")
	log.Infof("Will attach uprobe to: %s", targetFunction)

	// Create context with signal handling
	ctx, cancel := signal.NotifyContext(context.Background(),
		unix.SIGINT, unix.SIGTERM, unix.SIGABRT)
	defer cancel()

	// Initialize metrics
	metrics.Start(noop.Meter{})

	// Probe eBPF syscall support
	if err := tracer.ProbeBPFSyscall(); err != nil {
		log.Fatalf("Failed to probe eBPF syscall: %v", err)
	}

	// Create intervals configuration
	intervals := times.New(
		5*time.Second,  // Reporter interval
		5*time.Second,  // Monitor interval
		1*time.Minute,  // Probabilistic interval
	)

	// Start realtime clock synchronization
	times.StartRealtimeSync(ctx, 3*time.Minute)

	// Parse which tracers to include
	includeTracers, err := types.Parse("all")
	if err != nil {
		log.Fatalf("Failed to parse tracers: %v", err)
	}

	// Create our simple reporter
	reporter := NewSimpleReporter()

	// Configure the tracer
	tracerCfg := &tracer.Config{
		TraceReporter:          reporter,
		Intervals:              intervals,
		IncludeTracers:         includeTracers,
		SamplesPerSecond:       20,
		MapScaleFactor:         0,
		FilterErrorFrames:      true,
		KernelVersionCheck:     true,
		VerboseMode:            false,
		BPFVerifierLogLevel:    0,
		ProbabilisticInterval:  1 * time.Minute,
		ProbabilisticThreshold: 100,
		OffCPUThreshold:        0,
		IncludeEnvVars:         libpf.Set[string]{},
		LoadProbe:              true, // Enable probe loading
	}

	// Create and load the eBPF tracer
	log.Info("Loading eBPF tracer...")
	trc, err := tracer.NewTracer(ctx, tracerCfg)
	if err != nil {
		log.Fatalf("Failed to create tracer: %v", err)
	}
	defer trc.Close()

	// Get the custom__generic program FD
	customProgFD := trc.GetCustomTraceProgramFD()
	if customProgFD < 0 {
		log.Fatal("custom__generic program not loaded")
	}
	log.Infof("Got custom__generic program FD: %d", customProgFD)

	// Get the custom_context_map FD for sharing context_value
	customContextMapFD := trc.GetCustomContextMapFD()
	if customContextMapFD < 0 {
		log.Fatal("custom_context_map not loaded")
	}
	log.Infof("Got custom_context_map FD: %d", customContextMapFD)

	// Load our custom eBPF program spec
	spec, err := LoadExampleTailcall()
	if err != nil {
		log.Fatalf("Failed to load example eBPF spec: %v", err)
	}

	// Reuse the custom_context_map from the profiler
	// This allows both programs to share the same map for passing context_value
	spec.Maps["custom_context_map"].Pinning = 0 // Don't pin, just reuse FD

	// Create a Map object from the FD to reuse the profiler's custom_context_map
	customContextMap, err := ebpf.NewMapFromFD(customContextMapFD)
	if err != nil {
		log.Fatalf("Failed to create map from FD: %v", err)
	}
	defer customContextMap.Close()

	// Create RewriteOptions to reuse the profiler's custom_context_map
	opts := &ebpf.CollectionOptions{
		Maps: ebpf.MapOptions{
			PinPath: "", // Don't use pinning
		},
		MapReplacements: map[string]*ebpf.Map{
			"custom_context_map": customContextMap,
		},
	}

	objs := &ExampleTailcallObjects{}
	if err := spec.LoadAndAssign(objs, opts); err != nil {
		log.Fatalf("Failed to load example eBPF objects: %v", err)
	}
	defer objs.Close()

	log.Info("Successfully reused custom_context_map from profiler")

	// Add custom__generic to our prog_array at index 0
	// Both programs are uprobe type now, so cilium's Put should work
	if err := objs.ProgArray.Put(uint32(0), uint32(customProgFD)); err != nil {
		log.Fatalf("Failed to add custom__generic to prog_array: %v", err)
	}
	log.Info("Successfully added custom__generic to prog_array at index 0")

	// Attach our uprobe
	parts := splitFunctionSpec(targetFunction)
	if len(parts) != 2 {
		log.Fatalf("Invalid function spec: %s (expected format: executable:function)", targetFunction)
	}
	executable, function := parts[0], parts[1]

	ex, err := link.OpenExecutable(executable)
	if err != nil {
		log.Fatalf("Failed to open executable %s: %v", executable, err)
	}

	up, err := ex.Uprobe(function, objs.UprobeExample, nil)
	if err != nil {
		log.Fatalf("Failed to attach uprobe to %s: %v", function, err)
	}
	defer up.Close()

	log.Infof("Successfully attached uprobe to %s:%s", executable, function)
	log.Info("The uprobe will tail call to custom__generic with context_value")

	// Start PID event processor
	trc.StartPIDEventProcessor(ctx)

	// Attach tracer
	log.Info("Attaching eBPF programs...")
	if err := trc.AttachTracer(); err != nil {
		log.Fatalf("Failed to attach tracer: %v", err)
	}

	// Enable profiling
	if err := trc.EnableProfiling(); err != nil {
		log.Fatalf("Failed to enable profiling: %v", err)
	}

	// Attach scheduler monitor
	if err := trc.AttachSchedMonitor(); err != nil {
		log.Fatalf("Failed to attach scheduler monitor: %v", err)
	}

	// Start trace handling
	log.Info("Starting trace processing...")
	traceCh := make(chan *host.Trace)

	if err := trc.StartMapMonitors(ctx, traceCh); err != nil {
		log.Fatalf("Failed to start map monitors: %v", err)
	}

	// Process traces from the channel
	go func() {
		for {
			select {
			case trace := <-traceCh:
				if trace != nil {
					trc.HandleTrace(trace)
				}
			case <-ctx.Done():
				return
			}
		}
	}()

	log.Info("Profiler is running. Press Ctrl+C to stop...")
	log.Infof("Now run: %s", executable)
	log.Info("Traces will be collected when the function is called")

	// Periodic stats
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	go func() {
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				total, custom, sampling := reporter.GetStats()
				log.Infof("Traces - Total: %d, Custom: %d, Sampling: %d", total, custom, sampling)
			}
		}
	}()

	// Wait for signal
	<-ctx.Done()

	log.Info("Shutting down profiler...")

	total, custom, sampling := reporter.GetStats()
	log.Infof("Final stats - Total: %d, Custom: %d, Sampling: %d", total, custom, sampling)
}

func splitFunctionSpec(spec string) []string {
	for i := 0; i < len(spec); i++ {
		if spec[i] == ':' {
			return []string{spec[:i], spec[i+1:]}
		}
	}
	return []string{spec}
}
