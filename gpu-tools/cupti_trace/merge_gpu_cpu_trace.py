#!/usr/bin/env python3
"""
Merge GPU and CPU traces into folded flamegraph format
Combines CPU stack traces with GPU kernel and memory operations based on timestamp alignment
"""

import json
import re
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Any
from collections import defaultdict


class GPUEvent:
    """Represents a GPU event (kernel or memory operation)"""
    def __init__(self, name: str, start_ns: int, end_ns: int, event_type: str, size: int = 0):
        self.name = name
        self.start_ns = start_ns
        self.end_ns = end_ns
        self.event_type = event_type  # 'kernel', 'memcpy', 'memory'
        self.size = size  # For memory operations
        
    def overlaps(self, timestamp_ns: int) -> bool:
        """Check if this event overlaps with a given timestamp"""
        return self.start_ns <= timestamp_ns <= self.end_ns
    
    def __repr__(self):
        return f"GPUEvent({self.name}, {self.start_ns}-{self.end_ns}, {self.event_type})"


class CPUSample:
    """Represents a CPU stack sample"""
    def __init__(self, timestamp_ns: int, pid: int, tid: int, cpu: int, stack: List[str]):
        self.timestamp_ns = timestamp_ns
        self.pid = pid
        self.tid = tid
        self.cpu = cpu
        self.stack = stack  # List of function names from bottom to top
        
    def __repr__(self):
        return f"CPUSample({self.timestamp_ns}, pid={self.pid}, stack_depth={len(self.stack)})"


class TraceMerger:
    """Merges GPU and CPU traces"""
    
    def __init__(self):
        self.gpu_events = []
        self.cpu_samples = []
        self.merged_stacks = defaultdict(int)  # stack_string -> count
        
    def parse_cpu_trace(self, cpu_file: str):
        """Parse CPU trace file from the profiler output"""
        print(f"Parsing CPU trace: {cpu_file}")
        
        with open(cpu_file, 'r') as f:
            lines = f.readlines()
        
        # Skip warning lines at the beginning
        start_idx = 0
        for i, line in enumerate(lines):
            if not line.startswith('[') and not line.strip().startswith('WARN'):
                # Found first actual sample line
                if re.match(r'^\d+\s+\w+', line):
                    start_idx = i
                    break
        
        sample_count = 0
        for line in lines[start_idx:]:
            line = line.strip()
            if not line:
                continue
                
            # Parse format: timestamp process_name pid tid cpu stack
            # Example: 1756392917740211444 complex_target 425433 425433 81 _start;__libc_start_main;...
            parts = line.split(None, 5)  # Split on whitespace, max 6 parts
            
            if len(parts) < 6:
                continue
                
            try:
                timestamp_ns = int(parts[0])
                process_name = parts[1]
                pid = int(parts[2])
                tid = int(parts[3])
                cpu = int(parts[4])
                stack_str = parts[5]
                
                # Parse stack (semicolon separated)
                stack = []
                for frame in stack_str.split(';'):
                    frame = frame.strip()
                    if frame and frame != '0x0':
                        # Clean up frame name
                        if frame.startswith('[k]'):
                            frame = frame[3:] + '_[kernel]'
                        elif frame.startswith('0x'):
                            continue  # Skip pure addresses
                        stack.append(frame)
                
                if stack:
                    self.cpu_samples.append(CPUSample(timestamp_ns, pid, tid, cpu, stack))
                    sample_count += 1
                    
            except (ValueError, IndexError) as e:
                # Skip malformed lines
                continue
        
        print(f"Parsed {sample_count} CPU samples")
    
    def parse_gpu_trace(self, gpu_json_file: str):
        """Parse GPU trace JSON file"""
        print(f"Parsing GPU trace: {gpu_json_file}")
        
        with open(gpu_json_file, 'r') as f:
            data = json.load(f)
        
        events = data.get('traceEvents', [])
        kernel_count = 0
        memory_count = 0
        
        for event in events:
            name = event.get('name', '')
            category = event.get('cat', '')
            
            # Skip driver and runtime API calls, only keep actual GPU operations
            if category in ['CUDA_Driver', 'CUDA_Runtime', 'Overhead']:
                continue
                
            # Process kernels
            if category == 'GPU_Kernel' or name.startswith('Kernel:'):
                kernel_name = name.replace('Kernel: ', '')
                start_us = event.get('ts', 0)
                duration_us = event.get('dur', 0)
                
                if start_us > 0 and duration_us > 0:
                    start_ns = int(start_us * 1000)
                    end_ns = int((start_us + duration_us) * 1000)
                    self.gpu_events.append(GPUEvent(
                        f"[GPU_Kernel]{kernel_name}",
                        start_ns,
                        end_ns,
                        'kernel'
                    ))
                    kernel_count += 1
            
            # Process memory operations
            elif category == 'MemCopy' or name.startswith('MemCopy:'):
                memcpy_type = name.replace('MemCopy: ', '')
                start_us = event.get('ts', 0)
                duration_us = event.get('dur', 0)
                size = event.get('args', {}).get('size', 0)
                
                if start_us > 0 and duration_us > 0:
                    start_ns = int(start_us * 1000)
                    end_ns = int((start_us + duration_us) * 1000)
                    
                    # Format memory operation with size
                    size_str = self._format_size(size)
                    mem_name = f"[GPU_MemCpy]{memcpy_type}({size_str})"
                    
                    self.gpu_events.append(GPUEvent(
                        mem_name,
                        start_ns,
                        end_ns,
                        'memcpy',
                        size
                    ))
                    memory_count += 1
                    
            elif category == 'Memory':
                # Instant memory events
                operation = event.get('args', {}).get('operation', 'unknown')
                memory_kind = event.get('args', {}).get('kind', 'unknown')
                size = event.get('args', {}).get('size', 0)
                timestamp_us = event.get('ts', 0)
                
                if timestamp_us > 0:
                    timestamp_ns = int(timestamp_us * 1000)
                    size_str = self._format_size(size)
                    mem_name = f"[GPU_Memory]{operation}_{memory_kind}({size_str})"
                    
                    # For instant events, create a small duration window
                    self.gpu_events.append(GPUEvent(
                        mem_name,
                        timestamp_ns - 1000,  # 1 microsecond window
                        timestamp_ns + 1000,
                        'memory',
                        size
                    ))
                    memory_count += 1
        
        # Sort GPU events by start time for efficient searching
        self.gpu_events.sort(key=lambda e: e.start_ns)
        
        print(f"Parsed {kernel_count} kernel events and {memory_count} memory operations")
    
    def _format_size(self, size_bytes: int) -> str:
        """Format size in human readable format"""
        if size_bytes < 1024:
            return f"{size_bytes}B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes/1024:.1f}KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes/(1024*1024):.1f}MB"
        else:
            return f"{size_bytes/(1024*1024*1024):.1f}GB"
    
    def find_overlapping_gpu_events(self, timestamp_ns: int) -> List[GPUEvent]:
        """Find all GPU events that overlap with the given timestamp"""
        overlapping = []
        
        # Binary search for potential starting point
        left, right = 0, len(self.gpu_events) - 1
        start_idx = 0
        
        while left <= right:
            mid = (left + right) // 2
            if self.gpu_events[mid].end_ns < timestamp_ns:
                left = mid + 1
            else:
                start_idx = mid
                right = mid - 1
        
        # Check events starting from start_idx
        for i in range(max(0, start_idx - 10), min(len(self.gpu_events), start_idx + 100)):
            event = self.gpu_events[i]
            if event.start_ns > timestamp_ns:
                break  # No more possible overlaps
            if event.overlaps(timestamp_ns):
                overlapping.append(event)
        
        return overlapping
    
    def merge_traces(self):
        """Merge GPU events into CPU stack traces based on timestamp alignment"""
        print("Merging GPU and CPU traces...")
        
        merged_count = 0
        for sample in self.cpu_samples:
            # Find overlapping GPU events
            gpu_events = self.find_overlapping_gpu_events(sample.timestamp_ns)
            
            # Build merged stack
            merged_stack = sample.stack.copy()
            
            # Add GPU events to the top of the stack
            if gpu_events:
                # Group by type
                kernels = [e for e in gpu_events if e.event_type == 'kernel']
                memcpys = [e for e in gpu_events if e.event_type == 'memcpy']
                memories = [e for e in gpu_events if e.event_type == 'memory']
                
                # Add memory operations first (lowest level GPU activity)
                for event in memories:
                    merged_stack.append(event.name)
                
                # Then memory copies
                for event in memcpys:
                    merged_stack.append(event.name)
                
                # Then kernels (highest level GPU activity)
                for event in kernels:
                    merged_stack.append(event.name)
                
                merged_count += 1
            
            # Create folded stack string
            if merged_stack:
                stack_str = ';'.join(merged_stack)
                self.merged_stacks[stack_str] += 1
        
        print(f"Merged {merged_count} samples with GPU events")
        print(f"Total unique stacks: {len(self.merged_stacks)}")
    
    def write_folded_output(self, output_file: str):
        """Write folded stack format for flamegraph generation"""
        print(f"Writing folded output to: {output_file}")
        
        with open(output_file, 'w') as f:
            for stack, count in sorted(self.merged_stacks.items()):
                # Folded format: stack_frame1;stack_frame2;... count
                f.write(f"{stack} {count}\n")
        
        total_samples = sum(self.merged_stacks.values())
        print(f"Wrote {len(self.merged_stacks)} unique stacks ({total_samples} total samples)")
        
    def generate_summary(self):
        """Generate summary statistics"""
        print("\n=== Summary Statistics ===")
        
        # CPU statistics
        if self.cpu_samples:
            cpu_start = min(s.timestamp_ns for s in self.cpu_samples)
            cpu_end = max(s.timestamp_ns for s in self.cpu_samples)
            cpu_duration_ms = (cpu_end - cpu_start) / 1_000_000
            print(f"CPU trace duration: {cpu_duration_ms:.2f} ms")
            print(f"CPU samples: {len(self.cpu_samples)}")
            print(f"CPU sampling rate: {len(self.cpu_samples) / (cpu_duration_ms / 1000):.1f} Hz")
        
        # GPU statistics
        if self.gpu_events:
            gpu_start = min(e.start_ns for e in self.gpu_events)
            gpu_end = max(e.end_ns for e in self.gpu_events)
            gpu_duration_ms = (gpu_end - gpu_start) / 1_000_000
            
            kernel_events = [e for e in self.gpu_events if e.event_type == 'kernel']
            memory_events = [e for e in self.gpu_events if e.event_type in ['memcpy', 'memory']]
            
            print(f"\nGPU trace duration: {gpu_duration_ms:.2f} ms")
            print(f"GPU kernel events: {len(kernel_events)}")
            print(f"GPU memory events: {len(memory_events)}")
            
            if kernel_events:
                total_kernel_time = sum(e.end_ns - e.start_ns for e in kernel_events) / 1_000_000
                print(f"Total kernel execution time: {total_kernel_time:.2f} ms")
            
            if memory_events:
                total_mem_size = sum(e.size for e in memory_events if e.size > 0)
                print(f"Total memory transferred: {self._format_size(total_mem_size)}")


def main():
    parser = argparse.ArgumentParser(
        description='Merge GPU and CPU traces into folded flamegraph format'
    )
    parser.add_argument(
        '-c', '--cpu',
        default='cpu_results.txt',
        help='CPU trace file (default: cpu_results.txt)'
    )
    parser.add_argument(
        '-g', '--gpu',
        default='gpu_results.json',
        help='GPU trace JSON file (default: gpu_results.json)'
    )
    parser.add_argument(
        '-o', '--output',
        default='merged_trace.folded',
        help='Output folded stack file (default: merged_trace.folded)'
    )
    parser.add_argument(
        '-s', '--summary',
        action='store_true',
        help='Print summary statistics'
    )
    
    args = parser.parse_args()
    
    # Check input files exist
    if not Path(args.cpu).exists():
        print(f"Error: CPU trace file not found: {args.cpu}", file=sys.stderr)
        sys.exit(1)
    
    if not Path(args.gpu).exists():
        print(f"Error: GPU trace file not found: {args.gpu}", file=sys.stderr)
        sys.exit(1)
    
    # Create merger and process traces
    merger = TraceMerger()
    
    # Parse inputs
    merger.parse_cpu_trace(args.cpu)
    merger.parse_gpu_trace(args.gpu)
    
    # Merge traces
    merger.merge_traces()
    
    # Write output
    merger.write_folded_output(args.output)
    
    # Print summary if requested
    if args.summary:
        merger.generate_summary()
    
    print(f"\nTo generate flamegraph:")
    print(f"  flamegraph.pl {args.output} > merged_flamegraph.svg")
    print(f"\nOr use online viewer:")
    print(f"  https://www.speedscope.app/ (upload {args.output})")


if __name__ == '__main__':
    main()