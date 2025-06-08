#!/usr/bin/env python3
"""
Combine on-CPU and off-CPU profiling data into a single flame graph.
This script normalizes the different units and adds color coding.
"""

import sys
import argparse
import re
from collections import defaultdict

def parse_folded_file(filename, data_type):
    """Parse folded stack trace file and return dict of stack -> value"""
    stacks = {}
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Split stack and value
            parts = line.rsplit(' ', 1)
            if len(parts) != 2:
                continue
                
            stack, value_str = parts
            try:
                value = int(value_str)
                stacks[stack] = value
            except ValueError:
                continue
    
    return stacks

def normalize_oncpu_to_time(sample_count, sample_freq_hz=49):
    """Convert on-CPU sample count to microseconds"""
    # Each sample represents 1/frequency seconds
    time_seconds = sample_count / sample_freq_hz
    return int(time_seconds * 1_000_000)  # Convert to microseconds

def combine_stacks(oncpu_stacks, offcpu_stacks, sample_freq=49):
    """Combine on-CPU and off-CPU stacks with proper weighting"""
    combined = {}
    
    # Process on-CPU stacks (convert samples to time)
    for stack, samples in oncpu_stacks.items():
        time_us = normalize_oncpu_to_time(samples, sample_freq)
        # Mark as on-CPU with prefix
        oncpu_stack = f"[ON-CPU] {stack}"
        combined[oncpu_stack] = time_us
    
    # Process off-CPU stacks (already in microseconds)
    for stack, time_us in offcpu_stacks.items():
        # Mark as off-CPU with prefix
        offcpu_stack = f"[OFF-CPU] {stack}"
        combined[offcpu_stack] = time_us
    
    return combined

def write_combined_folded(combined_stacks, output_file):
    """Write combined stacks to folded format"""
    with open(output_file, 'w') as f:
        for stack, time_us in sorted(combined_stacks.items(), 
                                   key=lambda x: x[1], reverse=True):
            f.write(f"{stack} {time_us}\n")

def create_flamegraph_colors():
    """Create color configuration for flame graph"""
    colors = """
# Color configuration for combined flame graph
# ON-CPU stacks in warm colors (red/orange)
/\\[ON-CPU\\]/ { color = "red" }

# OFF-CPU stacks in cool colors (blue/green)  
/\\[OFF-CPU\\]/ { color = "blue" }

# Specific patterns
/\\[ON-CPU\\].*nginx/ { color = "orange" }
/\\[OFF-CPU\\].*nginx/ { color = "lightblue" }
/\\[OFF-CPU\\].*recv/ { color = "cyan" }
/\\[OFF-CPU\\].*send/ { color = "green" }
/\\[OFF-CPU\\].*setsockopt/ { color = "purple" }
"""
    return colors

def main():
    parser = argparse.ArgumentParser(description='Combine on-CPU and off-CPU profiles')
    parser.add_argument('oncpu_file', help='On-CPU folded stack file')
    parser.add_argument('offcpu_file', help='Off-CPU folded stack file') 
    parser.add_argument('-o', '--output', default='combined.folded',
                       help='Output combined folded file')
    parser.add_argument('-f', '--freq', type=int, default=49,
                       help='On-CPU sampling frequency (default: 49Hz)')
    parser.add_argument('--colors', action='store_true',
                       help='Generate color configuration file')
    
    args = parser.parse_args()
    
    print(f"Reading on-CPU data from {args.oncpu_file}")
    oncpu_stacks = parse_folded_file(args.oncpu_file, 'oncpu')
    print(f"Found {len(oncpu_stacks)} on-CPU stack traces")
    
    print(f"Reading off-CPU data from {args.offcpu_file}")
    offcpu_stacks = parse_folded_file(args.offcpu_file, 'offcpu')
    print(f"Found {len(offcpu_stacks)} off-CPU stack traces")
    
    print(f"Combining stacks (sampling freq: {args.freq}Hz)")
    combined = combine_stacks(oncpu_stacks, offcpu_stacks, args.freq)
    
    print(f"Writing combined data to {args.output}")
    write_combined_folded(combined, args.output)
    
    if args.colors:
        color_file = args.output.replace('.folded', '_colors.txt')
        with open(color_file, 'w') as f:
            f.write(create_flamegraph_colors())
        print(f"Color configuration written to {color_file}")
    
    # Calculate statistics
    total_oncpu_time = sum(normalize_oncpu_to_time(v, args.freq) 
                          for v in oncpu_stacks.values())
    total_offcpu_time = sum(offcpu_stacks.values())
    total_time = total_oncpu_time + total_offcpu_time
    
    print("\n=== Analysis Summary ===")
    print(f"Total on-CPU time:  {total_oncpu_time:,} μs ({total_oncpu_time/1000:.1f} ms)")
    print(f"Total off-CPU time: {total_offcpu_time:,} μs ({total_offcpu_time/1000:.1f} ms)")
    print(f"Total time:         {total_time:,} μs ({total_time/1000:.1f} ms)")
    
    if total_time > 0:
        oncpu_pct = (total_oncpu_time / total_time) * 100
        offcpu_pct = (total_offcpu_time / total_time) * 100
        print(f"\nTime distribution:")
        print(f"  On-CPU:  {oncpu_pct:.1f}%")
        print(f"  Off-CPU: {offcpu_pct:.1f}%")
        
        if offcpu_pct > 50:
            print(f"\n⚠️  Application is I/O bound ({offcpu_pct:.1f}% waiting)")
        else:
            print(f"\n🔥 Application is CPU bound ({oncpu_pct:.1f}% computing)")

if __name__ == '__main__':
    main() 