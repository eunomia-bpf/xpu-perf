package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"new-profiler/correlation"
	"new-profiler/event"
	"new-profiler/output"
	"new-profiler/source/composite"
	"new-profiler/source/primitive/cupti"
	"new-profiler/source/primitive/otel"
	"new-profiler/symbolizer"
	"os"
	"os/exec"
	"os/signal"
	"time"

	"github.com/cilium/ebpf"
	"github.com/cilium/ebpf/link"
	"go.opentelemetry.io/ebpf-profiler/host"
	"go.opentelemetry.io/ebpf-profiler/libpf"
	"go.opentelemetry.io/ebpf-profiler/metrics"
	"go.opentelemetry.io/ebpf-profiler/reporter"
	"go.opentelemetry.io/ebpf-profiler/reporter/samples"
	"go.opentelemetry.io/ebpf-profiler/times"
	"go.opentelemetry.io/ebpf-profiler/tracer"
	"go.opentelemetry.io/ebpf-profiler/tracer/types"
	"go.opentelemetry.io/otel/metric/noop"
	"golang.org/x/sys/unix"
)

type Config struct {
	cuptiLibPath        string
	outputFile          string
	testCuptiUprobeOnly bool
	uprobeOnly          bool // Test uprobe only (no CUPTI)
	gpuOnly             bool // Test correlation with CUPTI (uprobe traces + CUPTI)
	testUprobe          bool // Simple uprobe test with debugging
	debugDir            string
	targetBinary        string
	targetArgs          []string
	samplesPerSec       int
}

func main() {
	cfg := parseFlags()

	// Extract embedded CUPTI library if not specified
	if cfg.cuptiLibPath == "" {
		var err error
		cfg.cuptiLibPath, err = ExtractEmbeddedCUPTILibrary()
		if err != nil {
			fmt.Fprintf(os.Stderr, "Failed to extract CUPTI library: %v\n", err)
			os.Exit(1)
		}
		defer CleanupEmbeddedLibrary(cfg.cuptiLibPath)
		log.Printf("Using embedded CUPTI library")
	} else {
		log.Printf("Using CUPTI library from: %s", cfg.cuptiLibPath)
	}

	// Create context
	ctx, cancel := signal.NotifyContext(context.Background(),
		unix.SIGINT, unix.SIGTERM, unix.SIGABRT)
	defer cancel()

	// Handle Ctrl+C
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt)
	go func() {
		<-sigChan
		log.Println("Received interrupt, shutting down...")
		cancel()
	}()

	// Run profiling
	if err := runProfiling(ctx, cfg); err != nil {
		log.Fatalf("Profiling failed: %v", err)
	}

	log.Println("Profiling complete!")
	log.Printf("Generate flamegraph with: flamegraph.pl %s > flamegraph.svg", cfg.outputFile)
}

func runProfiling(ctx context.Context, cfg *Config) error {
	// Handle special test mode for CUPTI uprobe only
	if cfg.testCuptiUprobeOnly {
		return runCuptiUprobeTest(ctx, cfg)
	}

	// Handle simple uprobe test with debugging
	if cfg.testUprobe {
		return runSimpleUprobeTest(ctx, cfg)
	}

	// Handle uprobe-only mode (test uprobe without CUPTI)
	if cfg.uprobeOnly {
		log.Println("=== Uprobe-Only Mode ===")
		log.Println("Testing correlation uprobe without CUPTI")
		return runUprobeOnlyTest(ctx, cfg)
	}

	// Create CUPTI source (unless gpu-only mode where we don't need CPU sampling)
	var cuptiPipe string
	var cuptiSource *cupti.CUPTISource

	if !cfg.gpuOnly {
		var err error
		cuptiPipe, err = createNamedPipe()
		if err != nil {
			return fmt.Errorf("failed to create CUPTI pipe: %v", err)
		}
		defer os.Remove(cuptiPipe)
		log.Printf("Created CUPTI pipe: %s", cuptiPipe)

		cuptiSource = cupti.NewCUPTISource(cuptiPipe)
		log.Println("Created CUPTI source")
	} else {
		// GPU-only mode: still need CUPTI pipe for GPU events
		var err error
		cuptiPipe, err = createNamedPipe()
		if err != nil {
			return fmt.Errorf("failed to create CUPTI pipe: %v", err)
		}
		defer os.Remove(cuptiPipe)
		log.Printf("Created CUPTI pipe: %s", cuptiPipe)

		cuptiSource = cupti.NewCUPTISource(cuptiPipe)
		log.Println("Created CUPTI source (GPU-only mode)")
	}

	// Initialize metrics (using noop meter)
	metrics.Start(noop.Meter{})

	// Probe eBPF syscall support
	if err := tracer.ProbeBPFSyscall(); err != nil {
		return fmt.Errorf("failed to probe eBPF syscall: %v", err)
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
		return fmt.Errorf("failed to parse tracers: %v", err)
	}

	// Create OTel tracer configuration
	// Note: No cudaLaunchKernel uprobes - only using correlation ID uprobe
	otelCfg := &tracer.Config{
		TraceReporter:          &noopReporter{},
		ExecutableReporter:     &noopReporter{},
		Intervals:              intervals,
		IncludeTracers:         includeTracers,
		SamplesPerSecond:       cfg.samplesPerSec, // CPU sampling rate
		MapScaleFactor:         8,
		FilterErrorFrames:      true,
		VerboseMode:            false,
		BPFVerifierLogLevel:    0,
		ProbabilisticInterval:  1 * time.Minute,
		ProbabilisticThreshold: 100,
		IncludeEnvVars:         libpf.Set[string]{},
		UProbeLinks:            []string{}, // No uprobes
		LoadProbe:              true,
	}

	otelSource := otel.NewOTelSource(otelCfg)
	log.Println("Created OTel source")

	// Create correlation strategy (matches by correlation ID only)
	strategy := correlation.NewCorrelationIDStrategy()

	// Create correlated source
	correlatedSource := composite.NewCorrelateSource(cuptiSource, otelSource, strategy)
	log.Println("Created correlated source")

	// Initialize sources
	if err := correlatedSource.Initialize(); err != nil {
		return fmt.Errorf("failed to initialize source: %v", err)
	}
	log.Println("Initialized sources")

	// Always attach correlation uprobe to XpuPerfGetCorrelationId in CUPTI library
	log.Println("Attaching correlation uprobe to XpuPerfGetCorrelationId...")
	correlationUprobeState, err := otelSource.AttachCorrelationUprobeIfAvailable(cfg.cuptiLibPath)
	if err != nil {
		return fmt.Errorf("failed to attach correlation uprobe: %v", err)
	}
	defer correlationUprobeState.Close()
	log.Println("Correlation uprobe attached successfully!")

	// Start sources (events will be collected in background)
	if err := correlatedSource.Start(ctx, nil); err != nil {
		return fmt.Errorf("failed to start sources: %v", err)
	}
	log.Println("Started event collection")

	// Wait for everything to be ready
	time.Sleep(500 * time.Millisecond)

	// Start target process with CUPTI injection
	log.Printf("Starting target: %s %v", cfg.targetBinary, cfg.targetArgs)

	cmd := exec.Command(cfg.targetBinary, cfg.targetArgs...)
	cmd.Env = append(os.Environ(),
		fmt.Sprintf("CUDA_INJECTION64_PATH=%s", cfg.cuptiLibPath),
		fmt.Sprintf("CUPTI_TRACE_OUTPUT_FILE=%s", cuptiPipe),
		"CUPTI_ENABLE_MEMORY=1",
		"CUPTI_ENABLE_CORRELATION_UPROBE=1", // Always enable correlation uprobe
	)

	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("failed to start target: %v", err)
	}

	// Set target PID for filtering CPU sampling traces
	correlatedSource.SetTargetPID(uint32(cmd.Process.Pid))
	log.Printf("Filtering CPU sampling traces for PID %d", cmd.Process.Pid)

	// Wait for target to complete
	if err := cmd.Wait(); err != nil {
		log.Printf("Target exited with error: %v", err)
	}
	log.Println("Target process completed")

	// Wait for remaining events to be processed
	time.Sleep(1 * time.Second)

	// Stop profiling
	if err := correlatedSource.Stop(); err != nil {
		log.Printf("Error stopping sources: %v", err)
	}

	// Close symbolizer
	symbolizer.GetSymbolizer().Close()

	// Get merged events and CPU sampling stacks
	merged := correlatedSource.GetMergedEvents()
	cpuSamplingStacks := correlatedSource.GetCPUSamplingStacks()

	// Get statistics
	matched, unmatched, noCorrelation, cpuSampling := correlatedSource.GetStats()
	log.Printf("Statistics:")
	log.Printf("  - Matched CPU+GPU: %d", matched)
	log.Printf("  - Unmatched: %d", unmatched)
	log.Printf("  - No correlation ID: %d", noCorrelation)
	log.Printf("  - CPU sampling traces: %d", cpuSampling)
	log.Printf("Collected %d correlated CPU+GPU stacks", len(merged))
	log.Printf("Collected %d unique CPU sampling stacks", len(cpuSamplingStacks))

	// Write output
	if err := output.WriteFoldedStacks(cfg.outputFile, merged, cpuSamplingStacks, cfg.samplesPerSec); err != nil {
		return fmt.Errorf("failed to write output: %v", err)
	}

	log.Printf("Wrote folded stacks to: %s", cfg.outputFile)
	return nil
}

func runCuptiUprobeTest(ctx context.Context, cfg *Config) error {
	log.Println("=== CUPTI Correlation Uprobe Test Mode ===")
	log.Println("This mode tests the XpuPerfGetCorrelationId uprobe")
	log.Println("")

	// Initialize metrics
	metrics.Start(noop.Meter{})
	if err := tracer.ProbeBPFSyscall(); err != nil {
		return fmt.Errorf("failed to probe eBPF syscall: %v", err)
	}
	log.Println("✓ eBPF syscall probed")

	// Create intervals
	intervals := times.New(5*time.Second, 5*time.Second, 1*time.Minute)
	times.StartRealtimeSync(ctx, 3*time.Minute)

	// Parse tracers (disable sampling, only custom traces)
	includeTracers, _ := types.Parse("none")

	// Create minimal OTel tracer config (minimal sampling, mainly for custom traces)
	otelCfg := &tracer.Config{
		TraceReporter:          &noopReporter{},
		ExecutableReporter:     &noopReporter{},
		Intervals:              intervals,
		IncludeTracers:         includeTracers,
		SamplesPerSecond:       1, // Minimal sampling (required for buffer allocation)
		MapScaleFactor:         0, // Minimal map size
		FilterErrorFrames:      true,
		VerboseMode:            false,
		BPFVerifierLogLevel:    0,
		ProbabilisticInterval:  1 * time.Minute,
		ProbabilisticThreshold: 0, // Disable probabilistic profiling
		IncludeEnvVars:         libpf.Set[string]{},
		UProbeLinks:            []string{}, // No uprobes
		LoadProbe:              true,
	}

	otelSource := otel.NewOTelSource(otelCfg)
	log.Println("✓ Created minimal OTel source")

	// Initialize
	if err := otelSource.Initialize(); err != nil {
		return fmt.Errorf("failed to initialize OTel source: %v", err)
	}
	log.Println("✓ Initialized OTel source")

	// Attach correlation uprobe
	log.Println("Attaching correlation uprobe to XpuPerfGetCorrelationId...")
	correlationUprobeState, err := otelSource.AttachCorrelationUprobeIfAvailable(cfg.cuptiLibPath)
	if err != nil {
		return fmt.Errorf("failed to attach correlation uprobe: %v", err)
	}
	defer correlationUprobeState.Close()
	log.Println("✓ Correlation uprobe attached successfully!")

	// Create event channel to monitor correlation IDs
	events := make(chan *event.Event, 10000)

	// Start (this will begin monitoring)
	if err := otelSource.Start(ctx, events); err != nil {
		return fmt.Errorf("failed to start OTel source: %v", err)
	}
	log.Println("✓ Started monitoring")

	// Wait for eBPF to be ready
	time.Sleep(500 * time.Millisecond)

	// Start target process with CUPTI injection
	log.Printf("\n✓ Starting target: %s %v\n", cfg.targetBinary, cfg.targetArgs)

	cmd := exec.Command(cfg.targetBinary, cfg.targetArgs...)
	cmd.Env = append(os.Environ(),
		fmt.Sprintf("CUDA_INJECTION64_PATH=%s", cfg.cuptiLibPath),
		"CUPTI_ENABLE_CORRELATION_UPROBE=1", // Enable correlation uprobe in CUPTI
	)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("failed to start target: %v", err)
	}
	log.Printf("✓ Target PID: %d\n", cmd.Process.Pid)

	// Monitor correlation IDs
	correlationIDs := make(map[uint32]int)
	done := make(chan error, 1)
	go func() {
		done <- cmd.Wait()
	}()

	log.Println("\nMonitoring CUPTI correlation IDs...")
	monitoring := true
	for monitoring {
		select {
		case <-ctx.Done():
			monitoring = false
		case err := <-done:
			if err != nil {
				log.Printf("Target exited with error: %v\n", err)
			} else {
				log.Println("Target completed")
			}
			time.Sleep(500 * time.Millisecond) // Let remaining events arrive
			monitoring = false
		case evt := <-events:
			if evt != nil && evt.Type == event.EventCPUTrace {
				data := evt.Data.(*event.CPUTraceData)
				if evt.CorrelationID != 0 {
					correlationIDs[evt.CorrelationID]++
					log.Printf("  [Correlation ID] %d (count: %d, PID: %d, stack depth: %d)\n",
						evt.CorrelationID, correlationIDs[evt.CorrelationID],
						evt.ProcessID, len(data.Stack))
				}
			}
		}
	}

	// Stop
	if err := otelSource.Stop(); err != nil {
		log.Printf("Error stopping OTel source: %v", err)
	}

	log.Printf("\n\n=== Test Results ===\n")
	log.Printf("Total unique correlation IDs captured: %d\n", len(correlationIDs))
	for id, count := range correlationIDs {
		log.Printf("  - Correlation ID %d: %d occurrences\n", id, count)
	}

	if len(correlationIDs) > 0 {
		log.Println("\n✅ CUPTI correlation uprobe test PASSED")
		log.Println("   The XpuPerfGetCorrelationId uprobe is working correctly!")
	} else {
		log.Println("\n❌ CUPTI correlation uprobe test FAILED")
		log.Println("   No correlation IDs were captured.")
		log.Println("   Make sure CUPTI_ENABLE_CORRELATION_UPROBE=1 is set in the CUPTI library.")
		return fmt.Errorf("no correlation IDs captured")
	}

	return nil
}

func runSimpleUprobeTest(ctx context.Context, cfg *Config) error {
	log.Println("=== Simple Uprobe Test (Custom Tail Call Method) ===")
	log.Println("This test uses custom uprobes with tail call to custom__generic")
	log.Println("Sampling is DISABLED - only uprobe events will be captured")
	log.Println()

	// Extract embedded CUPTI library if needed
	cuptiLibPath := cfg.cuptiLibPath
	if cuptiLibPath == "" {
		var err error
		cuptiLibPath, err = ExtractEmbeddedCUPTILibrary()
		if err != nil {
			return fmt.Errorf("failed to extract embedded CUPTI library: %v", err)
		}
		defer os.Remove(cuptiLibPath)
		log.Printf("✓ Using embedded CUPTI library: %s\n", cuptiLibPath)
	} else {
		log.Printf("✓ Using CUPTI library: %s\n", cuptiLibPath)
	}

	// Initialize metrics
	metrics.Start(noop.Meter{})
	if err := tracer.ProbeBPFSyscall(); err != nil {
		return fmt.Errorf("failed to probe eBPF syscall: %v", err)
	}
	log.Println("✓ eBPF syscall probed")

	// Create intervals
	intervals := times.New(5*time.Second, 5*time.Second, 1*time.Minute)
	times.StartRealtimeSync(ctx, 3*time.Minute)

	// Parse tracers
	includeTracers, _ := types.Parse("all")

	// Create simple reporter to capture events
	reporter := &simpleUprobeReporter{
		correlationCount: make(map[uint32]int),
	}

	// Create tracer config with minimal sampling for perf buffer setup
	// Note: SamplesPerSecond must be > 0 for perf event array buffer initialization
	tracerCfg := &tracer.Config{
		TraceReporter:          reporter,
		ExecutableReporter:     reporter,
		Intervals:              intervals,
		IncludeTracers:         includeTracers,
		SamplesPerSecond:       20, // Same as custom-example
		MapScaleFactor:         0,  // Same as custom-example
		FilterErrorFrames:      true,
		VerboseMode:            false,
		BPFVerifierLogLevel:    0,
		ProbabilisticInterval:  1 * time.Minute,
		ProbabilisticThreshold: 100,
		IncludeEnvVars:         libpf.Set[string]{},
		UProbeLinks:            []string{},
		LoadProbe:              true,
	}

	log.Println("Creating eBPF tracer...")
	trc, err := tracer.NewTracer(ctx, tracerCfg)
	if err != nil {
		return fmt.Errorf("failed to create tracer: %v", err)
	}
	defer trc.Close()
	log.Println("✓ Created tracer")

	// Get custom__generic program FD
	customProgFD := trc.GetCustomTraceProgramFD()
	if customProgFD < 0 {
		return fmt.Errorf("custom__generic program not loaded")
	}
	log.Printf("✓ Got custom__generic program FD: %d\n", customProgFD)

	// Get custom_context_map FD
	customContextMapFD := trc.GetCustomContextMapFD()
	if customContextMapFD < 0 {
		return fmt.Errorf("custom_context_map not loaded")
	}
	log.Printf("✓ Got custom_context_map FD: %d\n", customContextMapFD)

	// Load correlation uprobe spec
	log.Println("Loading correlation uprobe eBPF program...")
	spec, err := otel.LoadCorrelationUprobe()
	if err != nil {
		return fmt.Errorf("failed to load correlation uprobe spec: %v", err)
	}

	// Create Map object from FD to reuse the profiler's custom_context_map
	customContextMap, err := ebpf.NewMapFromFD(customContextMapFD)
	if err != nil {
		return fmt.Errorf("failed to create map from FD: %v", err)
	}
	defer customContextMap.Close()

	// Load correlation uprobe with map reuse (same as custom-example)
	opts := &ebpf.CollectionOptions{
		MapReplacements: map[string]*ebpf.Map{
			"custom_context_map": customContextMap,
		},
	}

	objs := &otel.CorrelationUprobeObjects{}
	if err := spec.LoadAndAssign(objs, opts); err != nil {
		return fmt.Errorf("failed to load correlation uprobe: %v", err)
	}
	// DO NOT defer objs.Close() - keep it alive for tail calls
	log.Println("✓ Loaded correlation uprobe with shared custom_context_map")

	// Add custom__generic to prog_array at index 0 (same as custom-example)
	if err := objs.ProgArray.Put(uint32(0), uint32(customProgFD)); err != nil {
		return fmt.Errorf("failed to add custom__generic to prog_array: %v", err)
	}
	log.Println("✓ Added custom__generic to prog_array at index 0")

	// Attach uprobe to XpuPerfGetCorrelationId
	log.Printf("Attaching uprobe to %s:XpuPerfGetCorrelationId...\n", cuptiLibPath)
	ex, err := link.OpenExecutable(cuptiLibPath)
	if err != nil {
		return fmt.Errorf("failed to open executable: %v", err)
	}

	_, err = ex.Uprobe("XpuPerfGetCorrelationId", objs.CaptureCorrelationId, nil)
	if err != nil {
		return fmt.Errorf("failed to attach uprobe: %v", err)
	}
	// DO NOT defer Close() - keep uprobe alive for the entire session
	log.Println("✓ Uprobe attached successfully!")
	log.Println("  The uprobe will tail call to custom__generic with correlation ID")

	// Start PID event processor
	trc.StartPIDEventProcessor(ctx)

	// Attach tracer
	log.Println("Attaching eBPF programs...")
	if err := trc.AttachTracer(); err != nil {
		return fmt.Errorf("failed to attach tracer: %v", err)
	}

	// Enable profiling
	if err := trc.EnableProfiling(); err != nil {
		return fmt.Errorf("failed to enable profiling: %v", err)
	}

	// Attach scheduler monitor
	if err := trc.AttachSchedMonitor(); err != nil {
		return fmt.Errorf("failed to attach scheduler monitor: %v", err)
	}

	// Start trace handling
	log.Println("Starting trace processing...")
	traceCh := make(chan *host.Trace, 10000)

	if err := trc.StartMapMonitors(ctx, traceCh); err != nil {
		return fmt.Errorf("failed to start map monitors: %v", err)
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

	log.Println("✓ Profiler ready")

	// Wait for eBPF to be ready
	time.Sleep(500 * time.Millisecond)

	// Start target process with CUPTI injection
	log.Printf("\n✓ Starting target: %s %v\n", cfg.targetBinary, cfg.targetArgs)

	cmd := exec.Command(cfg.targetBinary, cfg.targetArgs...)
	cmd.Env = append(os.Environ(),
		fmt.Sprintf("CUDA_INJECTION64_PATH=%s", cuptiLibPath),
		"CUPTI_ENABLE_CORRELATION_UPROBE=1",
	)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("failed to start target: %v", err)
	}
	log.Printf("✓ Target PID: %d\n", cmd.Process.Pid)

	log.Println("\n=== Monitoring Uprobe Events ===")
	log.Println("(Sampling is disabled - only uprobe traces will appear)")
	log.Println()

	// Wait for target to complete
	if err := cmd.Wait(); err != nil {
		log.Printf("Target exited with error: %v\n", err)
	} else {
		log.Println("✓ Target completed")
	}

	// Wait for remaining events
	time.Sleep(500 * time.Millisecond)

	// Print results
	reporter.mu.Lock()
	defer reporter.mu.Unlock()

	log.Printf("\n\n=== Test Results ===\n")
	log.Printf("Total events: %d\n", reporter.totalEvents)
	log.Printf("Custom trace events (uprobes): %d\n", reporter.customTraceEvents)
	log.Printf("Events WITH correlation ID: %d\n", reporter.eventsWithCorrelation)
	log.Printf("Events WITHOUT correlation ID: %d\n", reporter.eventsWithoutCorrelation)
	log.Printf("Unique correlation IDs: %d\n", len(reporter.correlationCount))

	if len(reporter.correlationCount) > 0 {
		log.Println("\nCorrelation ID Distribution (first 20):")
		count := 0
		for id, occurrences := range reporter.correlationCount {
			log.Printf("  - Correlation ID %d: %d occurrences\n", id, occurrences)
			count++
			if count >= 20 {
				break
			}
		}
		log.Println("\n✅ SUCCESS: Uprobe captured correlation IDs!")
		return nil
	} else {
		log.Println("\n❌ FAILED: No correlation IDs captured")
		log.Println("\nDebugging info:")
		log.Printf("  - CUPTI library: %s\n", cuptiLibPath)
		log.Printf("  - Total events received: %d\n", reporter.totalEvents)
		log.Printf("  - Custom trace events: %d\n", reporter.customTraceEvents)
		return fmt.Errorf("no correlation IDs captured")
	}
}

func runUprobeOnlyTest(ctx context.Context, cfg *Config) error {
	log.Println("=== Uprobe-Only Test Mode ===")
	log.Println("This mode tests the XpuPerfGetCorrelationId uprobe without CUPTI")
	log.Println("")

	// Initialize metrics
	metrics.Start(noop.Meter{})
	if err := tracer.ProbeBPFSyscall(); err != nil {
		return fmt.Errorf("failed to probe eBPF syscall: %v", err)
	}
	log.Println("✓ eBPF syscall probed")

	// Create intervals
	intervals := times.New(5*time.Second, 5*time.Second, 1*time.Minute)
	times.StartRealtimeSync(ctx, 3*time.Minute)

	// Parse tracers
	includeTracers, _ := types.Parse("all")

	// Create OTel tracer config
	otelCfg := &tracer.Config{
		TraceReporter:          &noopReporter{},
		ExecutableReporter:     &noopReporter{},
		Intervals:              intervals,
		IncludeTracers:         includeTracers,
		SamplesPerSecond:       cfg.samplesPerSec,
		MapScaleFactor:         8,
		FilterErrorFrames:      true,
		VerboseMode:            false,
		BPFVerifierLogLevel:    0,
		ProbabilisticInterval:  1 * time.Minute,
		ProbabilisticThreshold: 100,
		IncludeEnvVars:         libpf.Set[string]{},
		UProbeLinks:            []string{},
		LoadProbe:              true,
	}

	otelSource := otel.NewOTelSource(otelCfg)
	log.Println("✓ Created OTel source")

	// Initialize
	if err := otelSource.Initialize(); err != nil {
		return fmt.Errorf("failed to initialize OTel source: %v", err)
	}
	log.Println("✓ Initialized OTel source")

	// Attach correlation uprobe
	log.Println("Attaching correlation uprobe to XpuPerfGetCorrelationId...")
	correlationUprobeState, err := otelSource.AttachCorrelationUprobeIfAvailable(cfg.cuptiLibPath)
	if err != nil {
		return fmt.Errorf("failed to attach correlation uprobe: %v", err)
	}
	defer correlationUprobeState.Close()
	log.Println("✓ Correlation uprobe attached successfully!")

	// Create event channel to monitor correlation IDs
	events := make(chan *event.Event, 10000)

	// Start (this will begin monitoring)
	if err := otelSource.Start(ctx, events); err != nil {
		return fmt.Errorf("failed to start OTel source: %v", err)
	}
	log.Println("✓ Started monitoring")

	// Wait for eBPF to be ready
	time.Sleep(500 * time.Millisecond)

	// Start target process with CUPTI injection
	log.Printf("\n✓ Starting target: %s %v\n", cfg.targetBinary, cfg.targetArgs)

	cmd := exec.Command(cfg.targetBinary, cfg.targetArgs...)
	cmd.Env = append(os.Environ(),
		fmt.Sprintf("CUDA_INJECTION64_PATH=%s", cfg.cuptiLibPath),
		"CUPTI_ENABLE_CORRELATION_UPROBE=1", // Enable correlation uprobe in CUPTI
	)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("failed to start target: %v", err)
	}
	log.Printf("✓ Target PID: %d\n", cmd.Process.Pid)

	// Monitor correlation IDs
	correlationIDs := make(map[uint32]int)
	cpuTraces := make(map[string]int)
	done := make(chan error, 1)
	go func() {
		done <- cmd.Wait()
	}()

	log.Println("\nMonitoring uprobe traces with correlation IDs...")
	monitoring := true
	for monitoring {
		select {
		case <-ctx.Done():
			monitoring = false
		case err := <-done:
			if err != nil {
				log.Printf("Target exited with error: %v\n", err)
			} else {
				log.Println("Target completed")
			}
			time.Sleep(500 * time.Millisecond) // Let remaining events arrive
			monitoring = false
		case evt := <-events:
			if evt != nil && evt.Type == event.EventCPUTrace {
				data := evt.Data.(*event.CPUTraceData)
				if evt.CorrelationID != 0 {
					correlationIDs[evt.CorrelationID]++
					// Convert stack to string
					stackStr := ""
					for i, frame := range data.Stack {
						if i > 0 {
							stackStr += ";"
						}
						stackStr += frame
					}
					cpuTraces[stackStr]++
				}
			}
		}
	}

	// Stop
	if err := otelSource.Stop(); err != nil {
		log.Printf("Error stopping OTel source: %v", err)
	}

	log.Printf("\n\n=== Test Results ===\n")
	log.Printf("Total unique correlation IDs captured: %d\n", len(correlationIDs))
	for id, count := range correlationIDs {
		log.Printf("  - Correlation ID %d: %d occurrences\n", id, count)
	}
	log.Printf("Total unique CPU stacks: %d\n", len(cpuTraces))

	// Write output
	if err := output.WriteFoldedStacksFromMap(cfg.outputFile, cpuTraces); err != nil {
		return fmt.Errorf("failed to write output: %v", err)
	}
	log.Printf("Wrote %d unique stacks to: %s\n", len(cpuTraces), cfg.outputFile)

	if len(correlationIDs) > 0 {
		log.Println("\n✅ Uprobe test PASSED")
		log.Println("   The XpuPerfGetCorrelationId uprobe is working correctly!")
	} else {
		log.Println("\n❌ Uprobe test FAILED")
		log.Println("   No correlation IDs were captured.")
		return fmt.Errorf("no correlation IDs captured")
	}

	return nil
}

func parseFlags() *Config {
	cfg := &Config{}

	flag.StringVar(&cfg.cuptiLibPath, "cupti-lib", "", "Path to CUPTI trace injection library (uses embedded library if not specified)")
	flag.StringVar(&cfg.outputFile, "o", "merged_trace.folded", "Output file for folded stack traces")
	flag.BoolVar(&cfg.testUprobe, "test-uprobe", false, "Test mode: simple uprobe test with detailed debugging output")
	flag.BoolVar(&cfg.testCuptiUprobeOnly, "test-cupti-uprobe-only", false, "Test mode: monitor CUPTI correlation uprobe only")
	flag.BoolVar(&cfg.uprobeOnly, "uprobe-only", false, "Test mode: test correlation uprobe without CUPTI (uprobe traces only)")
	flag.BoolVar(&cfg.gpuOnly, "gpu-only", false, "GPU correlation mode: correlate uprobe traces with CUPTI GPU events")
	flag.StringVar(&cfg.debugDir, "debug-dir", "", "Directory to save debug output files")
	flag.IntVar(&cfg.samplesPerSec, "samples-per-sec", 50, "CPU sampling frequency (Hz)")

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: %s [options] <target_binary> [args...]\n\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "GPU+CPU Performance Profiler\n\n")
		fmt.Fprintf(os.Stderr, "Correlates CPU sampling with GPU kernels using CUPTI correlation IDs.\n")
		fmt.Fprintf(os.Stderr, "Attaches eBPF uprobe to CUPTI's XpuPerfGetCorrelationId function to capture correlation IDs.\n\n")
		fmt.Fprintf(os.Stderr, "Options:\n")
		flag.PrintDefaults()
		fmt.Fprintf(os.Stderr, "\nModes:\n")
		fmt.Fprintf(os.Stderr, "  Default: CPU sampling + GPU events (merged view)\n")
		fmt.Fprintf(os.Stderr, "  -test-uprobe: Simple uprobe test with detailed debug output (recommended for testing)\n")
		fmt.Fprintf(os.Stderr, "  -uprobe-only: Test correlation uprobe only (no CUPTI GPU events)\n")
		fmt.Fprintf(os.Stderr, "  -gpu-only: Correlation mode (uprobe traces + CUPTI GPU events)\n")
		fmt.Fprintf(os.Stderr, "\nExamples:\n")
		fmt.Fprintf(os.Stderr, "  # Test if uprobe is capturing correlation IDs\n")
		fmt.Fprintf(os.Stderr, "  %s -test-uprobe ./my_cuda_app\n\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  # Default mode: merged CPU+GPU profiling\n")
		fmt.Fprintf(os.Stderr, "  %s -o trace.folded ./my_cuda_app\n\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  # GPU correlation mode\n")
		fmt.Fprintf(os.Stderr, "  %s -gpu-only -o correlated.folded ./my_cuda_app\n\n", os.Args[0])
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

// Simple reporter that does nothing (events go through correlation strategy)
type noopReporter struct{}

func (r *noopReporter) ReportTraceEvent(_ *libpf.Trace, _ *samples.TraceEventMeta) error {
	return nil
}

func (r *noopReporter) ReportFramesForTrace(_ *libpf.Trace) {
}

func (r *noopReporter) ReportCountForTrace(_ *libpf.Trace, _ uint16) {
}

func (r *noopReporter) ExecutableKnown(_ libpf.FileID, _ string) bool {
	return true
}

func (r *noopReporter) ExecutableReports() {
}

func (r *noopReporter) ReportExecutable(_ *reporter.ExecutableMetadata) {
}
