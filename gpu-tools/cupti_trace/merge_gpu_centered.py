#!/usr/bin/env python3
"""
GPU-Centered Trace Merger

This script creates a GPU-centric view where:
1. GPU activity (kernels, memory ops) forms the base of the flamegraph
2. Virtual samples are generated based on CPU sample times
3. For each virtual sample, we check which GPU operations are active
4. CPU stack is added on top of GPU operations

The resulting flamegraph shows GPU activity as the foundation with CPU work layered above.
"""

import json
import re
from collections import defaultdict
from typing import List, Dict, Set
from pathlib import Path


class GPUActivity:
    """Represents a GPU activity (kernel or memory operation)"""
    def __init__(self, name: str, start_ns: int, end_ns: int, activity_type: str):
        self.name = name
        self.start_ns = start_ns
        self.end_ns = end_ns
        self.activity_type = activity_type  # 'kernel', 'memcpy', 'memory'

    def overlaps(self, timestamp_ns: int) -> bool:
        """Check if this activity is running at given timestamp"""
        return self.start_ns <= timestamp_ns < self.end_ns

    def __repr__(self):
        return f"GPU:{self.name}"


class CPUSample:
    """Represents a CPU stack sample"""
    def __init__(self, timestamp_ns: int, stack: List[str]):
        self.timestamp_ns = timestamp_ns
        self.stack = stack


class GPUCenteredMerger:
    """Merges traces with GPU activity as the base"""

    def __init__(self):
        self.gpu_activities = []
        self.cpu_samples = []
        self.merged_stacks = defaultdict(int)

    def parse_gpu_trace(self, gpu_json_file: str):
        """Parse GPU trace from Chrome JSON format"""
        print(f"Parsing GPU trace: {gpu_json_file}")

        with open(gpu_json_file, 'r') as f:
            data = json.load(f)

        events = data.get('traceEvents', [])
        kernel_count = 0
        memcpy_count = 0
        memory_count = 0

        for event in events:
            name = event.get('name', '')
            category = event.get('cat', '')

            # Skip overhead and API calls
            if category in ['CUDA_Driver', 'CUDA_Runtime', 'Overhead']:
                continue

            # Process GPU kernels
            if category == 'GPU_Kernel' or name.startswith('Kernel:'):
                kernel_name = name.replace('Kernel: ', '').strip()
                # Clean up mangled names
                if kernel_name.startswith('_Z'):
                    # Simple demangling for common patterns
                    kernel_name = self._simple_demangle(kernel_name)

                start_us = event.get('ts', 0)
                duration_us = event.get('dur', 0)

                if start_us > 0 and duration_us > 0:
                    start_ns = int(start_us * 1000)
                    end_ns = int((start_us + duration_us) * 1000)
                    self.gpu_activities.append(GPUActivity(
                        f"[GPU_Kernel] {kernel_name}",
                        start_ns,
                        end_ns,
                        'kernel'
                    ))
                    kernel_count += 1

            # Process memory copies
            elif category == 'MemCopy' or name.startswith('MemCopy:'):
                memcpy_type = name.replace('MemCopy: ', '').strip()
                start_us = event.get('ts', 0)
                duration_us = event.get('dur', 0)

                if start_us > 0 and duration_us > 0:
                    start_ns = int(start_us * 1000)
                    end_ns = int((start_us + duration_us) * 1000)
                    self.gpu_activities.append(GPUActivity(
                        f"[GPU_MemCopy] {memcpy_type}",
                        start_ns,
                        end_ns,
                        'memcpy'
                    ))
                    memcpy_count += 1

            # Process memory operations
            elif category == 'Memory':
                mem_type = name.replace('Memory: ', '').strip()
                start_us = event.get('ts', 0)
                duration_us = event.get('dur', 0)

                if start_us > 0 and duration_us > 0:
                    start_ns = int(start_us * 1000)
                    end_ns = int((start_us + duration_us) * 1000)
                    self.gpu_activities.append(GPUActivity(
                        f"[GPU_Memory] {mem_type}",
                        start_ns,
                        end_ns,
                        'memory'
                    ))
                    memory_count += 1

        # Sort by start time for efficient searching
        self.gpu_activities.sort(key=lambda x: x.start_ns)

        print(f"Parsed {kernel_count} kernel events, {memcpy_count} memcpy events, {memory_count} memory events")

    def _simple_demangle(self, mangled: str) -> str:
        """Simple demangling for common CUDA kernel names"""
        # Map of common mangled patterns to readable names
        patterns = {
            '_Z12attentionQKT': 'attentionQKT',
            '_Z7softmax': 'softmax',
            '_Z16attentionScoresV': 'attentionScoresV',
            '_Z11residualAdd': 'residualAdd',
            '_Z9layerNorm': 'layerNorm',
            '_Z9matmulFFN': 'matmulFFN',
            '_Z4gelu': 'gelu',
        }

        for pattern, readable in patterns.items():
            if mangled.startswith(pattern):
                return readable

        return mangled

    def parse_cpu_trace(self, cpu_file: str):
        """Parse CPU trace file"""
        print(f"Parsing CPU trace: {cpu_file}")

        with open(cpu_file, 'r') as f:
            lines = f.readlines()

        # Skip warning lines
        start_idx = 0
        for i, line in enumerate(lines):
            if not line.startswith('[') and not line.strip().startswith('WARN'):
                if re.match(r'^\d+\s+\w+', line):
                    start_idx = i
                    break

        sample_count = 0
        for line in lines[start_idx:]:
            line = line.strip()
            if not line or line.startswith('['):
                continue

            # Parse: timestamp process_name pid tid cpu stack_trace
            parts = line.split(None, 5)
            if len(parts) >= 6:
                try:
                    timestamp_ns = int(parts[0])
                    stack_str = parts[5]
                    stack = stack_str.split(';')

                    self.cpu_samples.append(CPUSample(timestamp_ns, stack))
                    sample_count += 1
                except (ValueError, IndexError):
                    continue

        print(f"Parsed {sample_count} CPU samples")

    def find_active_gpu_activities(self, timestamp_ns: int) -> List[GPUActivity]:
        """Find all GPU activities active at given timestamp"""
        active = []

        # Binary search for first activity that might overlap
        # Find the first activity where end_ns > timestamp_ns
        left, right = 0, len(self.gpu_activities) - 1
        start_idx = len(self.gpu_activities)

        while left <= right:
            mid = (left + right) // 2
            if self.gpu_activities[mid].end_ns <= timestamp_ns:
                left = mid + 1
            else:
                start_idx = mid
                right = mid - 1

        # Scan backwards to find all overlapping activities
        # (activities that started before but are still running)
        i = start_idx - 1
        while i >= 0 and self.gpu_activities[i].overlaps(timestamp_ns):
            active.append(self.gpu_activities[i])
            i -= 1

        # Scan forwards to find all overlapping activities
        # (activities that start at or after timestamp)
        i = start_idx
        while i < len(self.gpu_activities) and self.gpu_activities[i].overlaps(timestamp_ns):
            active.append(self.gpu_activities[i])
            i += 1

        return active

    def merge_traces(self):
        """Create GPU-centered merged trace"""
        print("\nMerging traces with GPU activity as base...")

        if not self.gpu_activities:
            print("Warning: No GPU activities found")
            return

        if not self.cpu_samples:
            print("Warning: No CPU samples found")
            return

        # Get GPU execution time range
        gpu_start_ns = min(a.start_ns for a in self.gpu_activities)
        gpu_end_ns = max(a.end_ns for a in self.gpu_activities)

        # Get CPU sample time range
        cpu_start_ns = min(s.timestamp_ns for s in self.cpu_samples)
        cpu_end_ns = max(s.timestamp_ns for s in self.cpu_samples)

        print(f"GPU execution range: {gpu_start_ns} to {gpu_end_ns} ns ({(gpu_end_ns - gpu_start_ns) / 1e9:.3f} s)")
        print(f"CPU sample range: {cpu_start_ns} to {cpu_end_ns} ns ({(cpu_end_ns - cpu_start_ns) / 1e9:.3f} s)")
        print(f"Total CPU samples: {len(self.cpu_samples)}")

        # Validate time synchronization
        time_offset_ns = abs(gpu_start_ns - cpu_start_ns)
        if time_offset_ns > 1_000_000_000:  # More than 1 second difference
            print(f"\nWARNING: GPU and CPU traces appear to have different time bases!")
            print(f"         Time offset: {time_offset_ns / 1e9:.3f} seconds")
            print(f"         Correlation may be incorrect. Ensure traces are time-synchronized.")

        # Calculate average CPU sample interval from actual samples
        if len(self.cpu_samples) < 2:
            print("Warning: Not enough CPU samples to calculate interval")
            return

        cpu_timestamps = sorted([s.timestamp_ns for s in self.cpu_samples])
        intervals = [cpu_timestamps[i+1] - cpu_timestamps[i] for i in range(len(cpu_timestamps)-1)]
        avg_interval_ns = sum(intervals) // len(intervals)

        print(f"CPU sample average interval: {avg_interval_ns} ns (~{1_000_000_000/avg_interval_ns:.0f} Hz)")

        # Create a map of CPU samples by timestamp for quick lookup
        cpu_sample_map = {s.timestamp_ns: s for s in self.cpu_samples}
        cpu_timestamp_set = set(cpu_timestamps)

        # Generate virtual samples to fill gaps between CPU samples
        # Only create virtual samples where there are NO actual CPU samples
        virtual_samples = []
        for i in range(len(cpu_timestamps) - 1):
            current_ts = cpu_timestamps[i] + avg_interval_ns
            next_cpu_ts = cpu_timestamps[i + 1]

            # Fill the gap between consecutive CPU samples
            while current_ts < next_cpu_ts:
                # Only add if within GPU execution range
                if gpu_start_ns <= current_ts <= gpu_end_ns:
                    virtual_samples.append(current_ts)
                current_ts += avg_interval_ns

        print(f"Generated {len(virtual_samples)} virtual samples to fill gaps between CPU samples")
        print(f"Using {len(cpu_timestamps)} actual CPU samples")

        # Combine real CPU samples and virtual samples
        all_samples = sorted(cpu_timestamps + virtual_samples)

        # Process each sample (real or virtual)
        on_gpu_samples = 0
        off_gpu_samples = 0
        cpu_samples_used = 0
        virtual_samples_used = 0

        for sample_ts in all_samples:
            # Find active GPU activities at this timestamp
            active_gpu = self.find_active_gpu_activities(sample_ts)

            # Check if this is a real CPU sample or virtual sample
            is_real_cpu_sample = sample_ts in cpu_timestamp_set

            cpu_stack = []
            if is_real_cpu_sample:
                # Use the exact CPU sample
                cpu_stack = cpu_sample_map[sample_ts].stack
                cpu_samples_used += 1
            else:
                # Virtual sample - no CPU stack unless we find an overlapping one
                virtual_samples_used += 1

            if active_gpu:
                # GPU is busy - build stack from GPU activities up
                on_gpu_samples += 1

                # Start with base GPU activity state
                stack = ["[ON_GPU]"]

                # Add all active GPU operations
                # Group by type: memory operations first, then memcpy, then kernels
                memories = [a for a in active_gpu if a.activity_type == 'memory']
                memcpys = [a for a in active_gpu if a.activity_type == 'memcpy']
                kernels = [a for a in active_gpu if a.activity_type == 'kernel']

                for activity in memories + memcpys + kernels:
                    stack.append(activity.name)

                # Only add CPU stack if this is a real CPU sample
                # Check if CPU is blocked (waiting for GPU)
                cpu_is_blocked = False
                if is_real_cpu_sample and cpu_stack:
                    top_frame = cpu_stack[-1] if cpu_stack else ""
                    if 'cudaStreamSynchronize' in top_frame or 'cudaDeviceSynchronize' in top_frame or 'cuStreamSynchronize' in top_frame:
                        cpu_is_blocked = True

                # Only add CPU stack if:
                # 1. This is a real CPU sample (timestamp matches exactly), AND
                # 2. CPU is not blocked waiting for GPU
                if is_real_cpu_sample and not cpu_is_blocked and cpu_stack:
                    for frame in cpu_stack:
                        stack.append(frame)

                # Record this stack
                stack_str = ';'.join(stack)
                self.merged_stacks[stack_str] += 1

            else:
                # GPU is idle - CPU only
                off_gpu_samples += 1

                stack = ["[OFF_GPU]"]

                # Only add CPU stack for real CPU samples
                if is_real_cpu_sample and cpu_stack:
                    for frame in cpu_stack:
                        stack.append(frame)

                stack_str = ';'.join(stack)
                self.merged_stacks[stack_str] += 1

        print(f"\nProcessed {len(all_samples)} total samples ({cpu_samples_used} real CPU + {virtual_samples_used} virtual):")
        print(f"  {on_gpu_samples} samples with GPU activity ({on_gpu_samples/len(all_samples)*100:.1f}%)")
        print(f"  {off_gpu_samples} samples with GPU idle ({off_gpu_samples/len(all_samples)*100:.1f}%)")
        print(f"  {len(self.merged_stacks)} unique stacks")

    def write_folded_output(self, output_file: str):
        """Write folded stack format"""
        print(f"\nWriting folded output to: {output_file}")

        with open(output_file, 'w') as f:
            for stack, count in sorted(self.merged_stacks.items(), key=lambda x: (-x[1], x[0])):
                f.write(f"{stack} {count}\n")

        total_samples = sum(self.merged_stacks.values())
        print(f"Wrote {len(self.merged_stacks)} unique stacks ({total_samples} total samples)")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate GPU-centered flamegraph from CPU and GPU traces'
    )
    parser.add_argument('-c', '--cpu-trace', required=True, help='CPU trace file')
    parser.add_argument('-g', '--gpu-trace', required=True, help='GPU trace JSON file')
    parser.add_argument('-o', '--output', default='gpu_centered.folded', help='Output folded file')

    args = parser.parse_args()

    # Check input files exist
    if not Path(args.cpu_trace).exists():
        print(f"Error: CPU trace file not found: {args.cpu_trace}")
        return 1

    if not Path(args.gpu_trace).exists():
        print(f"Error: GPU trace file not found: {args.gpu_trace}")
        return 1

    print("=" * 70)
    print("GPU-Centered Flamegraph Generator")
    print("=" * 70)

    # Create merger
    merger = GPUCenteredMerger()

    # Parse traces
    merger.parse_gpu_trace(args.gpu_trace)
    merger.parse_cpu_trace(args.cpu_trace)

    # Merge with GPU as base
    merger.merge_traces()

    # Write output
    merger.write_folded_output(args.output)

    print("\n" + "=" * 70)
    print("To generate flamegraph:")
    print(f"  perl combined_flamegraph.pl {args.output} > gpu_centered_flamegraph.svg")
    print("=" * 70)

    return 0


if __name__ == '__main__':
    exit(main())
