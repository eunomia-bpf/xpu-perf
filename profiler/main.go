// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/exec"
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

type Config struct {
	cuptiLibPath string
	outputFile   string
	cudaLibPath  string
	cpuOnly      bool
	gpuOnly      bool
	mergeMode    bool
	targetBinary string
	targetArgs   []string
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
	if !cfg.gpuOnly {
		// Default to CUDA runtime library for kernel launch tracing
		if cfg.cudaLibPath == "" {
			cfg.cudaLibPath = findCudaLibrary()
		}
		uprobes = []string{cfg.cudaLibPath + ":cudaLaunchKernel"}
	}

	// Set up logging
	log.SetLevel(log.InfoLevel)
	log.SetFormatter(&log.TextFormatter{
		FullTimestamp: true,
	})

	log.Info("Starting GPU+CPU Performance Profiler...")

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

	// Create correlator
	correlator := NewTraceCorrelator(cuptiParser, 10.0, cfg.cpuOnly, cfg.gpuOnly)

	// Create context with signal handling
	ctx, cancel := signal.NotifyContext(context.Background(),
		unix.SIGINT, unix.SIGTERM, unix.SIGABRT)
	defer cancel()

	var trc *tracer.Tracer
	var reporter *SimpleReporter

	// Only set up eBPF profiler if not GPU-only mode
	if !cfg.gpuOnly {
		log.Infof("Attaching uprobes to: %v", uprobes)

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

		// NOW start the target process after uprobes are attached
		if !cfg.gpuOnly {
			log.Infof("Starting target binary: %s %v", cfg.targetBinary, cfg.targetArgs)
			if cfg.cpuOnly {
				targetCmd = exec.Command(cfg.targetBinary, cfg.targetArgs...)
				targetCmd.Stdout = os.Stdout
				targetCmd.Stderr = os.Stderr
			} else {
				targetCmd = startTargetWithCUPTI(cfg, cuptiPipe)
			}
			if err := targetCmd.Start(); err != nil {
				log.Fatalf("Failed to start target process: %v", err)
			}
			log.Infof("Target process PID: %d", targetCmd.Process.Pid)
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
	flag.BoolVar(&cfg.cpuOnly, "cpu-only", false, "Only collect CPU traces (no GPU profiling)")
	flag.BoolVar(&cfg.gpuOnly, "gpu-only", false, "Only collect GPU traces (no CPU profiling)")
	flag.BoolVar(&cfg.mergeMode, "merge", true, "Merge CPU and GPU traces (default: true)")

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: %s [options] <target_binary> [args...]\n\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "GPU+CPU Performance Profiler\n\n")
		fmt.Fprintf(os.Stderr, "Options:\n")
		flag.PrintDefaults()
		fmt.Fprintf(os.Stderr, "\nExamples:\n")
		fmt.Fprintf(os.Stderr, "  # Profile both GPU and CPU, merge output\n")
		fmt.Fprintf(os.Stderr, "  %s -o trace.folded ./my_cuda_app\n\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  # CPU only profiling\n")
		fmt.Fprintf(os.Stderr, "  %s -cpu-only -o cpu_trace.folded ./my_cuda_app\n\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  # GPU only profiling\n")
		fmt.Fprintf(os.Stderr, "  %s -gpu-only -o gpu_trace.folded ./my_cuda_app\n\n", os.Args[0])
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

func findCudaLibrary() string {
	// Search for CUDA runtime library in common locations
	cudaPaths := []string{
		"/usr/local/cuda-12.9/lib64/libcudart.so.12",
		"/usr/local/cuda-13.0/lib64/libcudart.so.12",
		"/usr/local/cuda/lib64/libcudart.so.12",
		"/usr/local/cuda-12.8/lib64/libcudart.so.12",
		"/usr/local/cuda/lib64/libcudart.so",
	}

	for _, path := range cudaPaths {
		if _, err := os.Stat(path); err == nil {
			return path
		}
	}

	log.Fatal("Could not find CUDA runtime library. Please specify with -cuda-lib")
	return ""
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
	cmd.Env = append(os.Environ(),
		fmt.Sprintf("CUDA_INJECTION64_PATH=%s", cfg.cuptiLibPath),
		fmt.Sprintf("CUPTI_TRACE_OUTPUT_FILE=%s", cuptiPipe),
		"CUPTI_ENABLE_MEMORY=1",
	)

	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	return cmd
}
