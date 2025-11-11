package composite

import (
	"context"
	"fmt"
	"new-profiler/correlation"
	"new-profiler/event"
)

// CorrelateSource combines two sources and correlates their events
type CorrelateSource struct {
	source1  event.EventSource
	source2  event.EventSource
	strategy correlation.CorrelationStrategy
}

func NewCorrelateSource(source1, source2 event.EventSource, strategy correlation.CorrelationStrategy) *CorrelateSource {
	return &CorrelateSource{
		source1:  source1,
		source2:  source2,
		strategy: strategy,
	}
}

func (s *CorrelateSource) Name() string {
	return fmt.Sprintf("correlate(%s,%s)", s.source1.Name(), s.source2.Name())
}

func (s *CorrelateSource) Initialize() error {
	if err := s.source1.Initialize(); err != nil {
		return fmt.Errorf("failed to initialize %s: %w", s.source1.Name(), err)
	}
	if err := s.source2.Initialize(); err != nil {
		return fmt.Errorf("failed to initialize %s: %w", s.source2.Name(), err)
	}
	return nil
}

func (s *CorrelateSource) Start(ctx context.Context, events chan<- *event.Event) error {
	// Create internal channel to receive from both sources
	internal := make(chan *event.Event, 10000)

	// Start both sources
	if err := s.source1.Start(ctx, internal); err != nil {
		return fmt.Errorf("failed to start %s: %w", s.source1.Name(), err)
	}
	if err := s.source2.Start(ctx, internal); err != nil {
		return fmt.Errorf("failed to start %s: %w", s.source2.Name(), err)
	}

	// Collect and correlate events
	go func() {
		for {
			select {
			case <-ctx.Done():
				return
			case evt := <-internal:
				if evt != nil {
					// Add to correlation strategy
					s.strategy.AddEvent(evt)

					// Also forward to output channel if provided
					if events != nil {
						events <- evt
					}
				}
			}
		}
	}()

	return nil
}

func (s *CorrelateSource) Stop() error {
	if err := s.source1.Stop(); err != nil {
		return err
	}
	if err := s.source2.Stop(); err != nil {
		return err
	}

	// Flush correlation strategy to finalize matching
	s.strategy.Flush()

	return nil
}

// GetMergedEvents returns the correlated CPU+GPU events
func (s *CorrelateSource) GetMergedEvents() []*correlation.MergedEvent {
	return s.strategy.GetMerged()
}

// GetCPUSamplingStacks returns pure CPU sampling stacks (no GPU correlation)
func (s *CorrelateSource) GetCPUSamplingStacks() map[string]int {
	return s.strategy.GetCPUSamplingStacks()
}

// GetStats returns correlation statistics
func (s *CorrelateSource) GetStats() (matched, unmatched, noCorrelation, cpuSampling int) {
	return s.strategy.GetStats()
}

// SetTargetPID sets the target PID for filtering sampling traces
func (s *CorrelateSource) SetTargetPID(pid uint32) {
	s.strategy.SetTargetPID(pid)
}
