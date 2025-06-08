#!/usr/bin/env python3
"""
Combined On-CPU and Off-CPU Profiler

This script runs both 'profile' and 'offcputime' tools simultaneously to capture
both on-CPU and off-CPU activity for a given process, then combines the results
into a unified flamegraph.

Usage:
    python3 combined_profiler.py <PID> [OPTIONS]
"""

import argparse
import subprocess
import sys
import os
import threading
import time
import tempfile
from pathlib import Path

class CombinedProfiler:
    def __init__(self, pid, duration=30, freq=49, min_block_us=1000):
        self.pid = pid
        self.duration = duration
        self.freq = freq
        self.min_block_us = min_block_us
        self.profile_output = []
        self.offcpu_output = []
        self.profile_error = None
        self.offcpu_error = None
        
        # Find tool paths
        self.script_dir = Path(__file__).parent
        self.profile_tool = self.script_dir / "profile"
        self.offcpu_tool = self.script_dir / "offcputime"
        
        # Check if tools exist
        if not self.profile_tool.exists():
            raise FileNotFoundError(f"Profile tool not found at {self.profile_tool}")
        if not self.offcpu_tool.exists():
            raise FileNotFoundError(f"Offcputime tool not found at {self.offcpu_tool}")

    def run_profile_tool(self):
        """Run the profile tool in a separate thread"""
        try:
            cmd = [
                str(self.profile_tool),
                "-p", str(self.pid),
                "-F", str(self.freq),
                "-f",  # Folded output format
                str(self.duration)
            ]
            
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.duration + 10)
            
            if result.returncode != 0:
                self.profile_error = f"Profile tool failed: {result.stderr}"
                return
                
            self.profile_output = result.stdout.strip().split('\n') if result.stdout.strip() else []
            
        except subprocess.TimeoutExpired:
            self.profile_error = "Profile tool timed out"
        except Exception as e:
            self.profile_error = f"Profile tool error: {str(e)}"

    def run_offcpu_tool(self):
        """Run the offcputime tool in a separate thread"""
        try:
            cmd = [
                str(self.offcpu_tool),
                "-p", str(self.pid),
                "-m", str(self.min_block_us),
                "-f",  # Folded output format
                str(self.duration)
            ]
            
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.duration + 10)
            
            if result.returncode != 0:
                self.offcpu_error = f"Offcputime tool failed: {result.stderr}"
                return
                
            self.offcpu_output = result.stdout.strip().split('\n') if result.stdout.strip() else []
            
        except subprocess.TimeoutExpired:
            self.offcpu_error = "Offcputime tool timed out"
        except Exception as e:
            self.offcpu_error = f"Offcputime tool error: {str(e)}"

    def run_profiling(self):
        """Run both profiling tools simultaneously"""
        print(f"Starting combined profiling for PID {self.pid} for {self.duration} seconds...")
        
        # Create threads for both tools
        profile_thread = threading.Thread(target=self.run_profile_tool)
        offcpu_thread = threading.Thread(target=self.run_offcpu_tool)
        
        # Start both threads
        profile_thread.start()
        offcpu_thread.start()
        
        # Wait for both to complete
        profile_thread.join()
        offcpu_thread.join()
        
        # Check for errors
        if self.profile_error:
            print(f"Profile tool error: {self.profile_error}", file=sys.stderr)
        if self.offcpu_error:
            print(f"Offcpu tool error: {self.offcpu_error}", file=sys.stderr)
            
        if self.profile_error and self.offcpu_error:
            raise RuntimeError("Both profiling tools failed")

    def parse_folded_line(self, line):
        """Parse a folded format line into stack trace and value"""
        if not line.strip():
            return None, None
            
        parts = line.rsplit(' ', 1)
        if len(parts) != 2:
            return None, None
            
        stack_trace = parts[0]
        try:
            value = int(parts[1])
            return stack_trace, value
        except ValueError:
            return None, None

    def normalize_and_combine_stacks(self):
        """Combine and normalize stack traces from both tools"""
        oncpu_stacks = {}
        offcpu_stacks = {}
        
        # Process on-CPU data (profile tool)
        print(f"Processing {len(self.profile_output)} on-CPU stack traces...")
        oncpu_total_samples = 0
        for line in self.profile_output:
            stack, value = self.parse_folded_line(line)
            if stack and value:
                oncpu_total_samples += value
                # remove the first part of the stack trace and add annotation
                stack_parts = stack.split(";")[1:]
                # Add _[c] annotation for CPU-intensive (on-CPU) stacks
                annotated_stack = ";".join(stack_parts) + "_[c]"
                oncpu_stacks[annotated_stack] = oncpu_stacks.get(annotated_stack, 0) + value
        
        # Process off-CPU data (offcputime tool) 
        print(f"Processing {len(self.offcpu_output)} off-CPU stack traces...")
        offcpu_total_us = 0
        for line in self.offcpu_output:
            stack, value = self.parse_folded_line(line)
            if stack and value:
                offcpu_total_us += value
                # remove the first part of the stack trace and add annotation
                stack_parts = stack.split(";")[1:]
                # Add _[o] annotation for off-CPU (I/O/blocking) stacks
                annotated_stack = ";".join(stack_parts) + "_[o]"
                offcpu_stacks[annotated_stack] = offcpu_stacks.get(annotated_stack, 0) + value
        
        # Store counts for summary
        self.oncpu_count = len(oncpu_stacks)
        self.offcpu_count = len(offcpu_stacks)
        
        # Combine stacks with annotations
        combined_stacks = {}
        
        # Add on-CPU stacks directly
        for stack, value in oncpu_stacks.items():
            combined_stacks[stack] = combined_stacks.get(stack, 0) + value
        
        # Normalize and add off-CPU stacks
        if offcpu_total_us > 0 and oncpu_total_samples > 0:
            # Calculate normalization factor
            # Assume each on-CPU sample represents 1/freq seconds of CPU time
            avg_oncpu_sample_us = (1.0 / self.freq) * 1_000_000  # microseconds per sample
            normalization_factor = avg_oncpu_sample_us
            
            print(f"On-CPU: {oncpu_total_samples} samples ({oncpu_total_samples/self.freq:.2f} seconds)")
            print(f"Off-CPU: {offcpu_total_us} microseconds ({offcpu_total_us/1_000_000:.2f} seconds)")
            print(f"Normalization factor: {normalization_factor:.0f} us/sample")
            
            # Add normalized off-CPU stacks
            for stack, value in offcpu_stacks.items():
                # Convert microseconds to equivalent samples
                normalized_value = int(value / normalization_factor)
                if normalized_value > 0:  # Only include if it results in at least 1 equivalent sample
                    combined_stacks[stack] = combined_stacks.get(stack, 0) + normalized_value
        else:
            # If no normalization needed, just add off-CPU stacks as-is
            for stack, value in offcpu_stacks.items():
                combined_stacks[stack] = combined_stacks.get(stack, 0) + value
        
        return combined_stacks

    def setup_flamegraph_tools(self):
        """Ensure FlameGraph tools are available and create custom color palette"""
        flamegraph_dir = self.script_dir / "FlameGraph"
        flamegraph_script = flamegraph_dir / "flamegraph.pl"
        
        if flamegraph_script.exists():
            # Create a custom flamegraph script with our color palette
            custom_script = self.script_dir / "combined_flamegraph.pl"
            self.create_custom_flamegraph_script(flamegraph_script, custom_script)
            return custom_script
        
        print("FlameGraph tools not found, cloning repository...")
        try:
            result = subprocess.run([
                "git", "clone", 
                "https://github.com/brendangregg/FlameGraph.git",
                str(flamegraph_dir), "--depth=1"
            ], capture_output=True, text=True, cwd=self.script_dir)
            
            if result.returncode != 0:
                print(f"Failed to clone FlameGraph: {result.stderr}")
                return None
                
            if flamegraph_script.exists():
                # Make it executable
                os.chmod(flamegraph_script, 0o755)
                print("FlameGraph tools cloned successfully")
                # Create custom script
                custom_script = self.script_dir / "combined_flamegraph.pl"
                self.create_custom_flamegraph_script(flamegraph_script, custom_script)
                return custom_script
            else:
                print("FlameGraph script not found after cloning")
                return None
                
        except Exception as e:
            print(f"Error setting up FlameGraph tools: {e}")
            return None

    def create_custom_flamegraph_script(self, original_script, custom_script):
        """Create a custom flamegraph script with our color palette"""
        try:
            with open(original_script, 'r') as f:
                content = f.read()
            
            # Add our custom color palette for combined profiling
            # Insert after the existing "chain" palette logic
            custom_palette = '''
	if (defined $type and $type eq "combined") {
		if ($name =~ m:_\\[c\\]$:) {	# CPU annotation (on-CPU)
			$type = "red";
		} elsif ($name =~ m:_\\[o\\]$:) {	# off-CPU annotation (I/O/blocking)
			$type = "blue";
		} else {			# default
			$type = "yellow";
		}
		# fall-through to color palettes
	}'''
            
            # Find the insertion point after the chain palette
            insertion_point = content.find('	if (defined $type and $type eq "chain") {')
            if insertion_point != -1:
                # Find the end of the chain block
                end_point = content.find('\t# color palettes', insertion_point)
                if end_point != -1:
                    # Insert our custom palette before the color palettes section
                    content = content[:end_point] + custom_palette + '\n\n\t' + content[end_point:]
            
            with open(custom_script, 'w') as f:
                f.write(content)
            
            # Make it executable
            os.chmod(custom_script, 0o755)
            print("Custom flamegraph script created with combined color palette")
            
        except Exception as e:
            print(f"Error creating custom flamegraph script: {e}")
            # Fall back to original script
            return original_script

    def generate_flamegraph_data(self, output_prefix=None):
        """Generate combined flamegraph data and SVG"""
        if output_prefix is None:
            output_prefix = f"combined_profile_pid{self.pid}_{int(time.time())}"
        
        folded_file = f"{output_prefix}.folded"
        svg_file = f"{output_prefix}.svg"
        
        combined_stacks = self.normalize_and_combine_stacks()
        
        if not combined_stacks:
            print("No stack traces collected from either tool")
            return None, None
        
        # Sort by value for better visualization
        sorted_stacks = sorted(combined_stacks.items(), key=lambda x: x[1], reverse=True)
        
        # Generate folded output
        output_lines = []
        for stack, value in sorted_stacks:
            output_lines.append(f"{stack} {value}")
        
        # Write folded data to file
        try:
            with open(folded_file, 'w') as f:
                f.write('\n'.join(output_lines))
            print(f"Combined flamegraph data written to: {folded_file}")
        except Exception as e:
            print(f"Error writing folded data: {e}")
            return None, None
        
        # Generate SVG flamegraph
        flamegraph_script = self.setup_flamegraph_tools()
        if flamegraph_script:
            try:
                print("Generating flamegraph SVG...")
                
                # Generate flamegraph with our custom combined color palette
                result = subprocess.run([
                    "perl", str(flamegraph_script), 
                    "--colors", "combined",  # Use our custom color palette
                    "--title", "Combined On-CPU and Off-CPU Profile",
                    folded_file
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    with open(svg_file, 'w') as f:
                        f.write(result.stdout)
                    print(f"Flamegraph SVG generated: {svg_file}")
                else:
                    print(f"Error generating flamegraph: {result.stderr}")
                    svg_file = None
            except Exception as e:
                print(f"Error running flamegraph.pl: {e}")
                svg_file = None
        else:
            print("FlameGraph tools not available, skipping SVG generation")
            svg_file = None
        
        # Print summary
        print(f"\nSummary:")
        print(f"Total unique stack traces: {len(sorted_stacks)}")
        oncpu_stacks = sum(1 for stack, _ in sorted_stacks if stack.startswith("oncpu:"))
        offcpu_stacks = sum(1 for stack, _ in sorted_stacks if stack.startswith("offcpu:"))
        print(f"On-CPU stack traces: {oncpu_stacks}")
        print(f"Off-CPU stack traces: {offcpu_stacks}")
        
        return folded_file, svg_file

def main():
    parser = argparse.ArgumentParser(
        description="Combined On-CPU and Off-CPU Profiler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Profile PID 1234 for 30 seconds (default)
  python3 combined_profiler.py 1234
  
  # Profile for 60 seconds with custom sampling frequency
  python3 combined_profiler.py 1234 -d 60 -f 99
  
  # Use custom output prefix for generated files
  python3 combined_profiler.py 1234 -o myapp_profile -m 5000
  
  # Build and run test program first:
  gcc -o test_program test_program.c
  ./test_program &
  python3 combined_profiler.py $!
        """
    )
    
    parser.add_argument("pid", type=int, help="Process ID to profile")
    parser.add_argument("-d", "--duration", type=int, default=30,
                        help="Duration to profile in seconds (default: 30)")
    parser.add_argument("-f", "--frequency", type=int, default=49,
                        help="On-CPU sampling frequency in Hz (default: 49)")
    parser.add_argument("-m", "--min-block-us", type=int, default=1000,
                        help="Minimum off-CPU block time in microseconds (default: 1000)")
    parser.add_argument("-o", "--output", type=str,
                        help="Output file prefix for generated files (default: combined_profile_pid<PID>_<timestamp>)")
    
    args = parser.parse_args()
    
    # Check if running as root
    if os.geteuid() != 0:
        print("Warning: This script typically requires root privileges to access BPF features", 
              file=sys.stderr)
    
    # Check if PID exists
    try:
        os.kill(args.pid, 0)
    except OSError:
        print(f"Error: Process {args.pid} does not exist", file=sys.stderr)
        sys.exit(1)
    
    try:
        profiler = CombinedProfiler(
            pid=args.pid,
            duration=args.duration,
            freq=args.frequency,
            min_block_us=args.min_block_us
        )
        
        profiler.run_profiling()
        folded_file, svg_file = profiler.generate_flamegraph_data(args.output)
        
        print(f"\n" + "="*60)
        print("PROFILING COMPLETE")
        print("="*60)
        if folded_file:
            print(f"📊 Folded data: {folded_file}")
        if svg_file:
            print(f"🔥 Flamegraph:  {svg_file}")
            print(f"   Open {svg_file} in a web browser to view the interactive flamegraph")
        else:
            print("⚠️  SVG flamegraph generation failed")
            if folded_file:
                print(f"   You can manually generate it with:")
                print(f"   ./FlameGraph/flamegraph.pl {folded_file} > flamegraph.svg")
        
        print("\n📝 Interpretation guide:")
        print("   • Red frames show CPU-intensive code paths (on-CPU) marked with _[c]")
        print("   • Blue frames show blocking/waiting operations (off-CPU) marked with _[o]")
        print("   • Wider sections represent more time spent in those functions")
        print("   • Values are normalized to make on-CPU and off-CPU time comparable")
        
    except KeyboardInterrupt:
        print("\nProfiling interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 