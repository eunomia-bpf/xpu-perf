// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"time"

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
		fmt.Fprintf(os.Stderr, "Usage: %s <executable:symbol> [<executable:symbol>...]\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "\nExample:\n")
		fmt.Fprintf(os.Stderr, "  %s /bin/bash:readline\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  %s /usr/bin/python3.12:PyEval_EvalFrameDefault\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  %s /lib/x86_64-linux-gnu/libc.so.6:malloc /lib/x86_64-linux-gnu/libc.so.6:free\n", os.Args[0])
		os.Exit(1)
	}

	uprobes := os.Args[1:]

	// Set up logging
	log.SetLevel(log.InfoLevel)
	log.SetFormatter(&log.TextFormatter{
		FullTimestamp: true,
	})

	log.Info("Starting uprobe profiler...")
	log.Infof("Attaching uprobes to: %v", uprobes)

	// Create context with signal handling
	ctx, cancel := signal.NotifyContext(context.Background(),
		unix.SIGINT, unix.SIGTERM, unix.SIGABRT)
	defer cancel()

	// Initialize metrics (using noop meter)
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

	// Configure the tracer with LoadProbe enabled for uprobes
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
		UProbeLinks:            uprobes,
		LoadProbe:              true, // Important: must be true for uprobes
	}

	// Create and load the eBPF tracer
	log.Info("Loading eBPF tracer...")
	trc, err := tracer.NewTracer(ctx, tracerCfg)
	if err != nil {
		log.Fatalf("Failed to create tracer: %v", err)
	}

	// Start PID event processor
	trc.StartPIDEventProcessor(ctx)

	// Attach tracer to perf events for regular sampling (optional)
	log.Info("Attaching eBPF programs...")
	if err := trc.AttachTracer(); err != nil {
		log.Fatalf("Failed to attach tracer: %v", err)
	}

	// Enable profiling
	if err := trc.EnableProfiling(); err != nil {
		log.Fatalf("Failed to enable profiling: %v", err)
	}

	// Attach uprobes - This is the key part!
	log.Info("Attaching uprobes...")
	if err := trc.AttachUProbes(uprobes); err != nil {
		log.Fatalf("Failed to attach uprobes: %v", err)
	}
	log.Info("Uprobes attached successfully!")

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
	log.Info("Waiting for uprobe events...")

	// Periodic stats
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	go func() {
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				total, uprobe, sampling := reporter.GetStats()
				log.Infof("Traces - Total: %d, Uprobe: %d, Sampling: %d", total, uprobe, sampling)
			}
		}
	}()

	// Wait for signal
	<-ctx.Done()

	log.Info("Shutting down profiler...")
	trc.Close()

	total, uprobe, sampling := reporter.GetStats()
	log.Infof("Final stats - Total: %d, Uprobe: %d, Sampling: %d", total, uprobe, sampling)
}
