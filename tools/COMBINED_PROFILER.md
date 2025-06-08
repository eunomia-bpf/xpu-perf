# Combined On-CPU and Off-CPU Profiler

This script combines both `profile` and `offcputime` BPF tools to create a unified view of application performance, showing both CPU usage and blocking behavior in a single flamegraph.

## Overview

The combined profiler addresses a common challenge in performance analysis: understanding both where your application spends CPU time (on-CPU) and where it waits or blocks (off-CPU). Traditional profilers typically show only one aspect, but modern applications often spend significant time in both states.

### What it does:

1. **Runs both tools simultaneously** - Executes `profile` and `offcputime` in parallel for the same duration
2. **Normalizes different metrics** - Converts off-CPU time (microseconds) to equivalent "samples" comparable with on-CPU sample counts
3. **Combines stack traces** - Merges results with clear prefixes (`oncpu:` and `offcpu:`) to distinguish behavior types
4. **Generates unified flamegraph data** - Outputs folded format suitable for FlameGraph visualization

## Key Features

- **Time-aware normalization**: Accounts for different sampling frequencies and time scales
- **Simultaneous profiling**: Captures both types of data for the exact same time period
- **Clear visualization**: Prefixes distinguish on-CPU from off-CPU activity in the flamegraph
- **Flexible filtering**: Supports minimum block time thresholds and sampling frequency control
- **Production-ready**: Handles edge cases, errors, and provides detailed progress information

## Usage

### Basic Usage
```bash
# Profile PID 1234 for 30 seconds (default)
python3 combined_profiler.py 1234

# Profile for 60 seconds with higher sampling frequency  
python3 combined_profiler.py 1234 -d 60 -f 99

# Save to file and filter small blocking events
python3 combined_profiler.py 1234 -o combined.folded -m 5000
```

### Command Line Options

- `pid` - Process ID to profile (required)
- `-d, --duration` - Duration in seconds (default: 30)
- `-f, --frequency` - On-CPU sampling frequency in Hz (default: 49)
- `-m, --min-block-us` - Minimum off-CPU block time in microseconds (default: 1000)
- `-o, --output` - Output file for flamegraph data (default: stdout)

### Test with Sample Program

```bash
bash tests/run_combined_test.sh 
```

## Understanding the Output

### Stack Trace Prefixes

The combined output uses prefixes to distinguish behavior types:

- **`oncpu:`** - Stack traces where CPU time was spent (from `profile` tool)
- **`offcpu:`** - Stack traces where blocking/waiting occurred (from `offcputime` tool)

### Example Output
```
oncpu:test_program;main;cpu_work 150
offcpu:test_program;main;blocking_work;nanosleep;sys_nanosleep 45
```

This shows:
- 150 "samples worth" of CPU time spent in `cpu_work` function
- 45 "samples worth" of blocking time in `nanosleep` system call

### Normalization Logic

The script normalizes off-CPU time to make it comparable with on-CPU samples:

1. **Calculate time per sample**: `1/frequency` seconds per on-CPU sample
2. **Convert to microseconds**: Time per sample × 1,000,000
3. **Normalize off-CPU values**: `off_cpu_microseconds / microseconds_per_sample`

This allows both metrics to be displayed proportionally in the same flamegraph.

## Interpreting Flamegraphs

### On-CPU Patterns (oncpu: prefix)
- **Wide towers**: Functions consuming significant CPU time
- **Deep stacks**: Complex call chains where CPU cycles are spent
- **Hot paths**: Frequently executed code requiring optimization

### Off-CPU Patterns (offcpu: prefix)  
- **Wide towers**: Long-running blocking operations
- **System call stacks**: I/O operations, network calls, synchronization
- **Wait patterns**: Lock contention, resource waiting

### Combined Analysis
- **Balance assessment**: Compare on-CPU vs off-CPU time distribution
- **Bottleneck identification**: Find whether issues are CPU-bound or I/O-bound
- **Optimization priorities**: Focus on the dominant behavior type first

## Technical Details

### Normalization Algorithm

The normalization aims to make off-CPU time visually comparable to on-CPU time:

```python
# Each on-CPU sample represents this much time
avg_oncpu_sample_us = (1.0 / sampling_frequency) * 1_000_000

# Convert off-CPU microseconds to equivalent samples
equivalent_samples = off_cpu_microseconds / avg_oncpu_sample_us
```

### Time Scale Considerations

- **On-CPU measurement**: Samples per second (frequency-based)
- **Off-CPU measurement**: Total blocking time in microseconds
- **Challenge**: Different units need normalization for meaningful comparison
- **Solution**: Convert both to "time-equivalent" values

### Threading and Synchronization

The script runs both profiling tools simultaneously using Python threading:
- Ensures identical profiling time windows
- Handles tool failures gracefully
- Captures stderr for debugging

## Prerequisites

- Linux kernel with BPF support (4.4+)
- Built `profile` and `offcputime` tools
- Root privileges (for BPF program loading)
- Python 3.6+

## Troubleshooting

### Common Issues

1. **"Tool not found"**: Ensure `profile` and `offcputime` binaries are built and in the same directory
2. **Permission denied**: Run with `sudo` for BPF access
3. **No data collected**: Check if target process is active and generating expected behavior
4. **Normalization warnings**: Occurs when one tool collects no data - this is normal for inactive processes

### Performance Considerations

- **Sampling frequency**: Higher frequencies increase overhead but improve resolution
- **Block time threshold**: Lower thresholds capture more events but increase data volume
- **Duration**: Longer profiling provides more stable results but consumes more resources

## Advanced Usage

### Custom Time Scales
```bash
# High-frequency profiling for CPU-intensive analysis
python3 combined_profiler.py 1234 -f 997 -d 10

# Focus on significant blocking events only
python3 combined_profiler.py 1234 -m 10000 -d 60
```

### Integration with Other Tools
```bash
# Combine with system monitoring
python3 combined_profiler.py 1234 -o profile.folded &
PROFILER_PID=$!
iostat 1 60 > iostat.log &
wait $PROFILER_PID
```

This combined approach provides a comprehensive view of application performance, revealing both computational hotspots and blocking bottlenecks in a single, unified visualization. 