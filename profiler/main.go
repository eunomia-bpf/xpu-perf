// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package main

//go:generate go run github.com/cilium/ebpf/cmd/bpf2go -target amd64 -cflags "-I/usr/include -I/usr/include/x86_64-linux-gnu" CorrelationUprobe ebpf/correlation_uprobe.ebpf.c

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"os/signal"
	_ "strings" // Used in correlation_uprobe_attach.go
	"time"

	_ "github.com/cilium/ebpf"      // Used in correlation_uprobe_attach.go
	_ "github.com/cilium/ebpf/link" // Used in correlation_uprobe_attach.go
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

type Config struct {
	cuptiLibPath         string
	outputFile           string
	cudaLibPath          string
	cpuOnly              bool
	gpuOnly              bool // Now means: use uprobes for CPU/GPU correlation
	useCorrelationUprobe bool // Use XpuPerfGetCorrelationId uprobe for exact matching
	targetBinary         string
	targetArgs           []string
	debugDir             string // Directory to save debug output files
}

func main() {
	if os.Geteuid() != 0 {
		fmt.Fprintf(os.Stderr, "Error: This profiler must be run as root\n")
		os.Exit(1)
	}

	// Parse command-line flags
	cfg := parseFlags()

	// Extract embedded CUPTI library if not specified
	var extractedCuptiLib string
	if cfg.cuptiLibPath == "" {
		var err error
		extractedCuptiLib, err = ExtractEmbeddedCUPTILibrary()
		if err != nil {
			fmt.Fprintf(os.Stderr, "Failed to extract CUPTI library: %v\n", err)
			os.Exit(1)
		}
		cfg.cuptiLibPath = extractedCuptiLib
		defer CleanupEmbeddedLibrary(extractedCuptiLib)
	}

	// Determine uprobes based on mode
	var uprobes []string
	if cfg.gpuOnly || (!cfg.cpuOnly && !cfg.gpuOnly) {
		// GPU-only and merge mode use uprobes for CPU/GPU correlation
		// Default to CUDA runtime library for kernel launch tracing
		if cfg.cudaLibPath == "" {
			// Try to detect CUDA library from target binary first
			cfg.cudaLibPath = detectCudaLibraryFromBinary(cfg.targetBinary)
			if cfg.cudaLibPath == "" {
				cfg.cudaLibPath = findCudaLibrary()
			}
		}
		// Attach to CUDA launch kernel symbols actually used by the target binary
		uprobes = buildUprobeList(cfg.cudaLibPath, cfg.targetBinary)
	}

	// Set up logging
	log.SetLevel(log.InfoLevel)
	log.SetFormatter(&log.TextFormatter{
		FullTimestamp: true,
	})

	if cfg.cpuOnly {
		log.Info("Starting CPU-only Performance Profiler...")
	} else if cfg.gpuOnly {
		log.Info("Starting GPU-only Performance Profiler (CPU/GPU correlation via uprobes)...")
	} else {
		log.Info("Starting Full CPU+GPU Performance Profiler (merged sampling)...")
	}

	// Create named pipe for CUPTI trace
	var cuptiPipe string
	var cuptiParser *CUPTIParser
	var targetCmd *exec.Cmd

	// Always create CUPTI parser (even if nil, correlator handles it)
	cuptiParser = NewCUPTIParser()

	if !cfg.cpuOnly {
		var err error
		cuptiPipe, err = createNamedPipe()
		if err != nil {
			log.Fatalf("Failed to create named pipe: %v", err)
		}
		defer os.Remove(cuptiPipe)

		log.Infof("Created named pipe: %s", cuptiPipe)

		// Start CUPTI parser in background
		go func() {
			log.Info("Opening CUPTI pipe for reading...")
			if err := cuptiParser.ParseFile(cuptiPipe); err != nil {
				log.Errorf("CUPTI parser error: %v", err)
			}
		}()

		// Give pipe reader time to open
		time.Sleep(100 * time.Millisecond)

		// NOTE: Target process will be started AFTER uprobes are attached (or immediately in GPU-only mode)
	}

	// Create correlator (samplesPerSec=50 matches the CPU profiling frequency)
	const samplesPerSec = 50
	// Modes:
	// - cpuOnly=true, gpuOnly=false: Only CPU sampling
	// - cpuOnly=false, gpuOnly=true: CPU/GPU correlation via uprobes
	// - cpuOnly=false, gpuOnly=false: Merged CPU sampling + GPU samples
	mergeMode := !cfg.cpuOnly && !cfg.gpuOnly
	correlator := NewTraceCorrelator(cuptiParser, 10.0, samplesPerSec, cfg.cpuOnly, cfg.gpuOnly, mergeMode, cfg.useCorrelationUprobe, cfg.debugDir)

	// Create context with signal handling
	ctx, cancel := signal.NotifyContext(context.Background(),
		unix.SIGINT, unix.SIGTERM, unix.SIGABRT)
	defer cancel()

	var trc *tracer.Tracer
	var reporter *SimpleReporter

	// Always set up eBPF profiler (for sampling or uprobes)
	if true {
		if len(uprobes) > 0 {
			log.Infof("Will attach %d uprobes for CPU/GPU correlation:", len(uprobes))
			for _, probe := range uprobes {
				log.Infof("  - %s", probe)
			}
		} else if mergeMode {
			log.Info("Merged mode: using CPU sampling + GPU kernel sampling (no correlation)")
		} else {
			log.Info("CPU-only mode: using regular sampling (no GPU)")
		}

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

		// Create our simple reporter with correlator
		reporter = NewSimpleReporter(correlator)

		// Configure the tracer with LoadProbe enabled for uprobes
		tracerCfg := &tracer.Config{
			TraceReporter:          reporter,
			Intervals:              intervals,
			IncludeTracers:         includeTracers,
			SamplesPerSecond:       50,
			MapScaleFactor:         0,
			FilterErrorFrames:      true,
			KernelVersionCheck:     true,
			VerboseMode:            true, // Enable verbose mode for debugging
			BPFVerifierLogLevel:    1,   // Enable BPF verifier logs
			ProbabilisticInterval:  1 * time.Minute,
			ProbabilisticThreshold: 100,
			OffCPUThreshold:        0,
			IncludeEnvVars:         libpf.Set[string]{},
			UProbeLinks:            uprobes,
			LoadProbe:              true, // Important: must be true for uprobes
		}

		log.Infof("Tracer config: VerboseMode=%v BPFVerifierLogLevel=%d UProbeLinks=%d",
			tracerCfg.VerboseMode, tracerCfg.BPFVerifierLogLevel, len(tracerCfg.UProbeLinks))

		// Create and load the eBPF tracer
		log.Info("Loading eBPF tracer...")
		trc, err = tracer.NewTracer(ctx, tracerCfg)
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

		// Attach uprobes only in merged mode for correlation
		if len(uprobes) > 0 {
			log.Infof("Attaching %d uprobes globally (will trigger on any process using CUDA library)", len(uprobes))
			for i, probe := range uprobes {
				log.Infof("  [%d] Attaching to: %s", i+1, probe)
			}
			if err := trc.AttachUProbes(uprobes); err != nil {
				log.Fatalf("Failed to attach uprobes: %v", err)
			}
			log.Info("Uprobes attached successfully! They will fire when any process calls cudaLaunchKernel.")
			// Give uprobes time to become active in the kernel
			log.Info("Waiting 100ms for uprobes to become fully active...")
			time.Sleep(100 * time.Millisecond)
		}

		// Attach correlation uprobe if enabled
		var correlationUprobeState *CorrelationUprobeState
		if cfg.useCorrelationUprobe && !cfg.cpuOnly {
			log.Info("Attaching correlation uprobe to XpuPerfGetCorrelationId...")
			state, err := attachCorrelationUprobe(trc, cfg.cuptiLibPath)
			if err != nil {
				log.Fatalf("Failed to attach correlation uprobe: %v", err)
			}
			correlationUprobeState = state
			defer correlationUprobeState.Close()
			log.Info("Correlation uprobe attached successfully!")
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

		// Give the trace processing goroutine and eBPF map monitors time to fully start
		log.Info("Waiting 500ms for trace processing to fully initialize...")
		time.Sleep(500 * time.Millisecond)

		// Start the target process
		log.Infof("Starting target binary: %s %v", cfg.targetBinary, cfg.targetArgs)
		if cfg.cpuOnly {
			// CPU-only: no CUPTI injection
			targetCmd = exec.Command(cfg.targetBinary, cfg.targetArgs...)
			targetCmd.Stdout = os.Stdout
			targetCmd.Stderr = os.Stderr
		} else {
			// Merged mode: with CUPTI injection
			targetCmd = startTargetWithCUPTI(cfg, cuptiPipe)
		}
		if err := targetCmd.Start(); err != nil {
			log.Fatalf("Failed to start target process: %v", err)
		}
		log.Infof("Target process PID: %d", targetCmd.Process.Pid)

		// In merge mode, set target PID for filtering
		if mergeMode && reporter != nil {
			reporter.SetTargetPID(targetCmd.Process.Pid)
			log.Infof("Merge mode: filtering CPU samples for PID %d", targetCmd.Process.Pid)
		}

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
	} else {
		// GPU-only mode: start target process with CUPTI
		log.Info("GPU-only mode: starting target process...")
		log.Infof("Starting target binary: %s %v", cfg.targetBinary, cfg.targetArgs)
		targetCmd = startTargetWithCUPTI(cfg, cuptiPipe)
		if err := targetCmd.Start(); err != nil {
			log.Fatalf("Failed to start target process: %v", err)
		}
		log.Infof("Target process PID: %d", targetCmd.Process.Pid)
		log.Info("Waiting for target process to complete...")
	}

	// Wait for target process to complete or signal
	targetDone := make(chan error, 1)
	go func() {
		if targetCmd != nil {
			targetDone <- targetCmd.Wait()
		} else {
			<-ctx.Done()
			targetDone <- nil
		}
	}()

	select {
	case err := <-targetDone:
		if err != nil {
			log.Errorf("Target process exited with error: %v", err)
		} else {
			log.Info("Target process completed successfully")
			// Give eBPF time to process any pending events
			log.Info("Waiting 200ms for eBPF to process pending events...")
			time.Sleep(200 * time.Millisecond)
		}
		cancel()
	case <-ctx.Done():
		log.Info("Interrupted by signal")
		if targetCmd != nil && targetCmd.Process != nil {
			targetCmd.Process.Kill()
		}
	}

	// Give CUPTI time to flush
	time.Sleep(500 * time.Millisecond)

	log.Info("Shutting down profiler...")
	if trc != nil {
		trc.Close()

		total, uprobe, sampling := reporter.GetStats()
		log.Infof("Final stats - Total: %d, Uprobe: %d, Sampling: %d", total, uprobe, sampling)
	}

	// Close symbolizer
	GetSymbolizer().Close()

	// Correlate traces and write output
	log.Info("Correlating CPU and GPU traces...")
	if err := correlator.CorrelateTraces(); err != nil {
		log.Errorf("Correlation error: %v", err)
	}

	log.Infof("Writing folded output to: %s", cfg.outputFile)
	if err := correlator.WriteFoldedOutput(cfg.outputFile); err != nil {
		log.Fatalf("Failed to write output: %v", err)
	}

	log.Info("Profiling complete!")
	log.Infof("Generate flamegraph with: flamegraph.pl %s > flamegraph.svg", cfg.outputFile)
}

func parseFlags() *Config {
	cfg := &Config{}

	flag.StringVar(&cfg.cuptiLibPath, "cupti-lib", "", "Path to CUPTI trace injection library (uses embedded library if not specified)")
	flag.StringVar(&cfg.outputFile, "o", "merged_trace.folded", "Output file for folded stack traces")
	flag.StringVar(&cfg.cudaLibPath, "cuda-lib", "", "Path to CUDA runtime library (auto-detected if not specified)")
	flag.BoolVar(&cfg.cpuOnly, "cpu-only", false, "Only collect CPU sampling traces (no GPU profiling)")
	flag.BoolVar(&cfg.gpuOnly, "gpu-only", false, "Correlate CPU stacks with GPU kernels via uprobes")
	flag.BoolVar(&cfg.useCorrelationUprobe, "use-correlation-id", false, "Use XpuPerfGetCorrelationId uprobe for exact correlation (requires CUPTI_ENABLE_CORRELATION_UPROBE=1)")
	flag.StringVar(&cfg.debugDir, "debug-dir", "", "Directory to save debug output files (CUPTI, eBPF traces, correlation details)")

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: %s [options] <target_binary> [args...]\n\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "GPU+CPU Performance Profiler\n\n")
		fmt.Fprintf(os.Stderr, "Options:\n")
		flag.PrintDefaults()
		fmt.Fprintf(os.Stderr, "\nModes:\n")
		fmt.Fprintf(os.Stderr, "  Default: Merge CPU sampling + GPU time (converted to sample counts)\n")
		fmt.Fprintf(os.Stderr, "  -cpu-only: Only CPU sampling traces (no GPU)\n")
		fmt.Fprintf(os.Stderr, "  -gpu-only: CPU→GPU causality via uprobes (shows which CPU code launched which GPU kernel)\n")
		fmt.Fprintf(os.Stderr, "\nExamples:\n")
		fmt.Fprintf(os.Stderr, "  # Full CPU+GPU profiling (merged view with correct time proportions)\n")
		fmt.Fprintf(os.Stderr, "  %s -o trace.folded ./my_cuda_app\n\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  # CPU only sampling (no GPU)\n")
		fmt.Fprintf(os.Stderr, "  %s -cpu-only -o cpu_trace.folded ./my_cuda_app\n\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  # CPU/GPU correlation (shows full call path to GPU kernels)\n")
		fmt.Fprintf(os.Stderr, "  %s -gpu-only -o correlated_trace.folded ./my_cuda_app\n\n", os.Args[0])
	}

	flag.Parse()

	if flag.NArg() < 1 {
		flag.Usage()
		os.Exit(1)
	}

	cfg.targetBinary = flag.Arg(0)
	cfg.targetArgs = flag.Args()[1:]

	return cfg
}

func createNamedPipe() (string, error) {
	// Create a named pipe in /tmp
	pipePath := fmt.Sprintf("/tmp/cupti_trace_%d.pipe", os.Getpid())

	// Remove if exists
	os.Remove(pipePath)

	// Create named pipe
	if err := unix.Mkfifo(pipePath, 0600); err != nil {
		return "", fmt.Errorf("failed to create named pipe: %v", err)
	}

	return pipePath, nil
}

func startTargetWithCUPTI(cfg *Config, cuptiPipe string) *exec.Cmd {
	cmd := exec.Command(cfg.targetBinary, cfg.targetArgs...)

	// Set environment variables for CUPTI injection
	envVars := []string{
		fmt.Sprintf("CUDA_INJECTION64_PATH=%s", cfg.cuptiLibPath),
		fmt.Sprintf("CUPTI_TRACE_OUTPUT_FILE=%s", cuptiPipe),
		"CUPTI_ENABLE_MEMORY=1",
	}

	// Enable correlation uprobe if requested
	if cfg.useCorrelationUprobe {
		envVars = append(envVars, "CUPTI_ENABLE_CORRELATION_UPROBE=1")
	}

	cmd.Env = append(os.Environ(), envVars...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	return cmd
}
