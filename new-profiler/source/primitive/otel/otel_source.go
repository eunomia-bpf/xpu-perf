package otel

import (
	"context"
	"fmt"
	"new-profiler/event"

	"go.opentelemetry.io/ebpf-profiler/host"
	"go.opentelemetry.io/ebpf-profiler/tracer"
)

// OTelSource wraps the OpenTelemetry eBPF profiler
type OTelSource struct {
	tracer    *tracer.Tracer
	tracerCfg *tracer.Config
	reporter  *CapturingReporter
	ctx       context.Context
}

func NewOTelSource(tracerCfg *tracer.Config) *OTelSource {
	// Create a capturing reporter that will be configured with events channel later
	reporter := NewCapturingReporter(nil)

	// Update tracer config to use our capturing reporter
	tracerCfg.TraceReporter = reporter
	tracerCfg.ExecutableReporter = reporter

	return &OTelSource{
		tracerCfg: tracerCfg,
		reporter:  reporter,
	}
}

func (s *OTelSource) Name() string {
	return "otel"
}

func (s *OTelSource) Initialize() error {
	s.ctx = context.Background()
	trc, err := tracer.NewTracer(s.ctx, s.tracerCfg)
	if err != nil {
		return fmt.Errorf("failed to create tracer: %w", err)
	}
	s.tracer = trc

	// Start PID event processor
	s.tracer.StartPIDEventProcessor(s.ctx)

	// Attach tracer
	if err := s.tracer.AttachTracer(); err != nil {
		return fmt.Errorf("failed to attach tracer: %w", err)
	}

	// Enable profiling
	if err := s.tracer.EnableProfiling(); err != nil {
		return fmt.Errorf("failed to enable profiling: %w", err)
	}

	// Attach uprobes if configured
	if len(s.tracerCfg.UProbeLinks) > 0 {
		if err := s.tracer.AttachUProbes(s.tracerCfg.UProbeLinks); err != nil {
			return fmt.Errorf("failed to attach uprobes: %w", err)
		}
		fmt.Printf("Attached %d uprobes\n", len(s.tracerCfg.UProbeLinks))
	}

	// Attach scheduler monitor
	if err := s.tracer.AttachSchedMonitor(); err != nil {
		return fmt.Errorf("failed to attach scheduler monitor: %w", err)
	}

	return nil
}

func (s *OTelSource) Start(ctx context.Context, events chan<- *event.Event) error {
	// Configure the reporter with the events channel
	if events != nil {
		s.reporter.SetEventsChannel(events)
	}

	// Start trace handling
	traceCh := make(chan *host.Trace, 10000)

	if err := s.tracer.StartMapMonitors(ctx, traceCh); err != nil {
		return fmt.Errorf("failed to start map monitors: %w", err)
	}

	// Handle traces - the reporter will do symbolization and send events
	go func() {
		for {
			select {
			case <-ctx.Done():
				return
			case trace := <-traceCh:
				if trace != nil {
					// HandleTrace will call our CapturingReporter which will
					// symbolize the frames and send events to the events channel
					s.tracer.HandleTrace(trace)
				}
			}
		}
	}()

	return nil
}

func (s *OTelSource) Stop() error {
	if s.tracer != nil {
		s.tracer.Close()
	}
	return nil
}

// GetTracer returns the underlying tracer for advanced operations
func (s *OTelSource) GetTracer() *tracer.Tracer {
	return s.tracer
}

// AttachCorrelationUprobeIfAvailable attempts to attach the correlation uprobe
// if the CUPTI library is available. Returns nil if successful or if CUPTI lib not found.
func (s *OTelSource) AttachCorrelationUprobeIfAvailable(cuptiLibPath string) (*CorrelationUprobeState, error) {
	// This should be called after Initialize()
	if s.tracer == nil {
		return nil, fmt.Errorf("tracer not initialized")
	}

	// Attach correlation uprobe
	state, err := AttachCorrelationUprobe(s.tracer, cuptiLibPath)
	if err != nil {
		return nil, err
	}

	return state, nil
}
