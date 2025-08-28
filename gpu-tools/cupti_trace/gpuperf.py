#!/usr/bin/env python3

import os
import sys
import argparse
import subprocess
import tempfile
import json
import re
import signal
import atexit
import threading
import time
from pathlib import Path

class GPUPerf:
    def __init__(self):
        self.script_dir = Path(__file__).parent.absolute()
        self.injection_lib = self.script_dir / "libcupti_trace_injection.so"
        self.output_file = None
        self.temp_trace_file = None
        self.profiler_proc = None
        self.profiler_output = None
        
        # Path to CPU profiler
        self.cpu_profiler = Path("/root/yunwei37/systemscope/profiler/target/release/profile")
        if not self.cpu_profiler.exists():
            print(f"Warning: CPU profiler not found at {self.cpu_profiler}", file=sys.stderr)
            self.cpu_profiler = None
        
        # Find CUPTI library path
        cuda_paths = [
            "/usr/local/cuda-13.0/extras/CUPTI/lib64",
            "/usr/local/cuda/extras/CUPTI/lib64",
            "/usr/local/cuda-12.0/extras/CUPTI/lib64",
        ]
        
        self.cupti_lib = None
        for path in cuda_paths:
            cupti_path = Path(path) / "libcupti.so"
            if cupti_path.exists():
                self.cupti_lib = str(cupti_path)
                self.cupti_lib_dir = str(Path(path))
                break
                
        if not self.cupti_lib:
            print("Warning: Could not find CUPTI library. NVTX annotations may not work.", file=sys.stderr)
    
    def parse_cupti_trace(self, filename):
        """Parse CUPTI trace data and convert to Chrome Trace Format"""
        
        events = []
        
        # Regular expressions for different trace line formats
        runtime_pattern = r'RUNTIME \[ (\d+), (\d+) \] duration (\d+), "([^"]+)", cbid (\d+), processId (\d+), threadId (\d+), correlationId (\d+)'
        driver_pattern = r'DRIVER \[ (\d+), (\d+) \] duration (\d+), "([^"]+)", cbid (\d+), processId (\d+), threadId (\d+), correlationId (\d+)'
        kernel_pattern = r'CONCURRENT_KERNEL \[ (\d+), (\d+) \] duration (\d+), "([^"]+)", correlationId (\d+)'
        overhead_pattern = r'OVERHEAD ([A-Z_]+) \[ (\d+), (\d+) \] duration (\d+), (\w+), id (\d+), correlation id (\d+)'
        memory_pattern = r'MEMORY2 \[ (\d+) \] memoryOperationType (\w+), memoryKind (\w+), size (\d+), address (\d+)'
        memcpy_pattern = r'MEMCPY "([^"]+)" \[ (\d+), (\d+) \] duration (\d+), size (\d+), copyCount (\d+), srcKind (\w+), dstKind (\w+), correlationId (\d+)'
        grid_pattern = r'\s+grid \[ (\d+), (\d+), (\d+) \], block \[ (\d+), (\d+), (\d+) \]'
        device_pattern = r'\s+deviceId (\d+), contextId (\d+), streamId (\d+)'
        
        with open(filename, 'r') as f:
            lines = f.readlines()
            
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if not line or line.startswith('Calling CUPTI') or line.startswith('Enabling') or \
               line.startswith('Disabling') or line.startswith('Found') or \
               line.startswith('Configuring') or line.startswith('It took') or \
               line.startswith('Activity buffer') or line.startswith('CUPTI trace output'):
                i += 1
                continue
                
            # Parse RUNTIME events
            match = re.search(runtime_pattern, line)
            if match:
                start_time = int(match.group(1))
                duration = int(match.group(3))
                name = match.group(4)
                cbid = match.group(5)
                process_id = int(match.group(6))
                thread_id = int(match.group(7))
                correlation_id = int(match.group(8))
                
                events.append({
                    "name": f"Runtime: {name}",
                    "ph": "X",
                    "ts": start_time / 1000,
                    "dur": duration / 1000,
                    "tid": thread_id,
                    "pid": process_id,
                    "cat": "CUDA_Runtime",
                    "args": {
                        "cbid": cbid,
                        "correlationId": correlation_id
                    }
                })
                i += 1
                continue
                
            # Parse DRIVER events
            match = re.search(driver_pattern, line)
            if match:
                start_time = int(match.group(1))
                duration = int(match.group(3))
                name = match.group(4)
                cbid = match.group(5)
                process_id = int(match.group(6))
                thread_id = int(match.group(7))
                correlation_id = int(match.group(8))
                
                events.append({
                    "name": f"Driver: {name}",
                    "ph": "X",
                    "ts": start_time / 1000,
                    "dur": duration / 1000,
                    "tid": thread_id,
                    "pid": process_id,
                    "cat": "CUDA_Driver",
                    "args": {
                        "cbid": cbid,
                        "correlationId": correlation_id
                    }
                })
                i += 1
                continue
                
            # Parse CONCURRENT_KERNEL events
            match = re.search(kernel_pattern, line)
            if match:
                start_time = int(match.group(1))
                duration = int(match.group(3))
                name = match.group(4)
                correlation_id = int(match.group(5))
                
                kernel_info = {
                    "name": f"Kernel: {name}",
                    "ph": "X",
                    "ts": start_time / 1000,
                    "dur": duration / 1000,
                    "cat": "GPU_Kernel",
                    "args": {
                        "correlationId": correlation_id
                    }
                }
                
                # Check next lines for additional kernel info
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    grid_match = re.search(grid_pattern, next_line)
                    if grid_match:
                        kernel_info["args"]["grid"] = [
                            int(grid_match.group(1)),
                            int(grid_match.group(2)),
                            int(grid_match.group(3))
                        ]
                        kernel_info["args"]["block"] = [
                            int(grid_match.group(4)),
                            int(grid_match.group(5)),
                            int(grid_match.group(6))
                        ]
                        i += 1
                        
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    device_match = re.search(device_pattern, next_line)
                    if device_match:
                        device_id = int(device_match.group(1))
                        context_id = int(device_match.group(2))
                        stream_id = int(device_match.group(3))
                        
                        kernel_info["tid"] = f"GPU{device_id}_Stream{stream_id}"
                        kernel_info["pid"] = f"Device_{device_id}"
                        kernel_info["args"]["deviceId"] = device_id
                        kernel_info["args"]["contextId"] = context_id
                        kernel_info["args"]["streamId"] = stream_id
                        i += 1
                        
                events.append(kernel_info)
                i += 1
                continue
                
            # Parse OVERHEAD events
            match = re.search(overhead_pattern, line)
            if match:
                overhead_type = match.group(1)
                start_time = int(match.group(2))
                duration = int(match.group(4))
                overhead_target = match.group(5)
                overhead_id = int(match.group(6))
                correlation_id = int(match.group(7))
                
                events.append({
                    "name": f"Overhead: {overhead_type}",
                    "ph": "X",
                    "ts": start_time / 1000,
                    "dur": duration / 1000,
                    "tid": overhead_id,
                    "pid": "CUPTI_Overhead",
                    "cat": "Overhead",
                    "args": {
                        "type": overhead_type,
                        "target": overhead_target,
                        "correlationId": correlation_id
                    }
                })
                i += 1
                continue
                
            # Parse MEMCPY events
            match = re.search(memcpy_pattern, line)
            if match:
                copy_type = match.group(1)
                start_time = int(match.group(2))
                duration = int(match.group(4))
                size = int(match.group(5))
                copy_count = int(match.group(6))
                src_kind = match.group(7)
                dst_kind = match.group(8)
                correlation_id = int(match.group(9))
                
                memcpy_info = {
                    "name": f"MemCopy: {copy_type}",
                    "ph": "X",
                    "ts": start_time / 1000,
                    "dur": duration / 1000,
                    "cat": "MemCopy",
                    "args": {
                        "type": copy_type,
                        "size": size,
                        "copyCount": copy_count,
                        "srcKind": src_kind,
                        "dstKind": dst_kind,
                        "correlationId": correlation_id
                    }
                }
                
                # Check next line for device info
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    device_match = re.search(device_pattern, next_line)
                    if device_match:
                        device_id = int(device_match.group(1))
                        context_id = int(device_match.group(2))
                        stream_id = int(device_match.group(3))
                        
                        memcpy_info["tid"] = f"GPU{device_id}_Stream{stream_id}"
                        memcpy_info["pid"] = f"Device_{device_id}"
                        memcpy_info["args"]["deviceId"] = device_id
                        memcpy_info["args"]["contextId"] = context_id
                        memcpy_info["args"]["streamId"] = stream_id
                        i += 1
                    else:
                        memcpy_info["tid"] = "MemCopy_Operations"
                        memcpy_info["pid"] = "MemCopy"
                        
                events.append(memcpy_info)
                i += 1
                continue
                
            # Parse MEMORY2 events
            match = re.search(memory_pattern, line)
            if match:
                timestamp = int(match.group(1))
                operation = match.group(2)
                memory_kind = match.group(3)
                size = int(match.group(4))
                address = int(match.group(5))
                
                events.append({
                    "name": f"Memory: {operation} ({memory_kind})",
                    "ph": "i",
                    "ts": timestamp / 1000,
                    "tid": "Memory_Operations",
                    "pid": "Memory",
                    "cat": "Memory",
                    "s": "g",
                    "args": {
                        "operation": operation,
                        "kind": memory_kind,
                        "size": size,
                        "address": hex(address)
                    }
                })
                i += 1
                continue
                
            i += 1
        
        return events
    
    def start_cpu_profiler(self, pid, cpu_output_file=None):
        """Start CPU profiler in background for given PID"""
        if not self.cpu_profiler:
            return None
            
        if not cpu_output_file:
            cpu_output_file = f"cpu_profile_{pid}.txt"
            
        self.profiler_output = cpu_output_file
        print(f"Starting CPU profiler for PID {pid}, output: {cpu_output_file}")
        
        try:
            self.profiler_proc = subprocess.Popen(
                [str(self.cpu_profiler), "-p", str(pid), "-E"],
                stdout=open(cpu_output_file, 'w'),
                stderr=subprocess.DEVNULL
            )
            return self.profiler_proc
        except Exception as e:
            print(f"Warning: Failed to start CPU profiler: {e}", file=sys.stderr)
            return None
    
    def stop_cpu_profiler(self):
        """Stop the CPU profiler gracefully"""
        if self.profiler_proc and self.profiler_proc.poll() is None:
            print("Stopping CPU profiler...")
            self.profiler_proc.terminate()
            try:
                self.profiler_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.profiler_proc.kill()
                self.profiler_proc.wait()
            
            if self.profiler_output and os.path.exists(self.profiler_output):
                print(f"CPU profile saved to: {self.profiler_output}")
    
    def run_with_trace(self, command, output_trace=None, chrome_trace=None, cpu_profile=None):
        """Run a command with CUPTI tracing and optional CPU profiling enabled"""
        
        # Check if injection library exists
        if not self.injection_lib.exists():
            print(f"Error: CUPTI injection library not found at {self.injection_lib}", file=sys.stderr)
            print("Please build it first using 'make' in the cupti_trace directory", file=sys.stderr)
            return 1
        
        # Set up trace output file
        if output_trace:
            trace_file = output_trace
        else:
            # Create temporary file for trace output
            fd, trace_file = tempfile.mkstemp(suffix=".txt", prefix="gpuperf_trace_")
            os.close(fd)
            self.temp_trace_file = trace_file
            atexit.register(self.cleanup_temp_files)
        
        # Set up environment variables
        env = os.environ.copy()
        env['CUDA_INJECTION64_PATH'] = str(self.injection_lib)
        env['CUPTI_TRACE_OUTPUT_FILE'] = trace_file
        
        if self.cupti_lib:
            env['NVTX_INJECTION64_PATH'] = self.cupti_lib
            if 'LD_LIBRARY_PATH' in env:
                env['LD_LIBRARY_PATH'] = f"{self.cupti_lib_dir}:{env['LD_LIBRARY_PATH']}"
            else:
                env['LD_LIBRARY_PATH'] = self.cupti_lib_dir
        
        print(f"Running command with GPU profiling: {' '.join(command)}")
        print(f"Trace output: {trace_file}")
        
        # Start the target process
        target_proc = None
        
        try:
            # Start the target process
            target_proc = subprocess.Popen(command, env=env)
            target_pid = target_proc.pid
            print(f"Started target process with PID: {target_pid}")
            
            # Start CPU profiler if available and requested
            if cpu_profile and self.cpu_profiler:
                # Give the process a moment to start
                time.sleep(0.1)
                self.start_cpu_profiler(target_pid, cpu_profile)
            
            # Wait for the target process to complete
            return_code = target_proc.wait()
            
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            if target_proc:
                target_proc.terminate()
                try:
                    target_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    target_proc.kill()
            return_code = 130
        except Exception as e:
            print(f"Error running command: {e}", file=sys.stderr)
            return_code = 1
        finally:
            # Stop CPU profiler if running
            self.stop_cpu_profiler()
        
        # Convert to Chrome trace if requested
        if chrome_trace and os.path.exists(trace_file):
            print(f"\nConverting trace to Chrome format: {chrome_trace}")
            try:
                events = self.parse_cupti_trace(trace_file)
                print(f"Parsed {len(events)} events")
                
                trace_data = {
                    "traceEvents": events,
                    "displayTimeUnit": "ms",
                    "metadata": {
                        "tool": "gpuperf - GPU Performance Profiler",
                        "format": "Chrome Trace Format",
                        "command": ' '.join(command)
                    }
                }
                
                with open(chrome_trace, 'w') as f:
                    json.dump(trace_data, f, indent=2)
                
                print(f"\nChrome trace file written to: {chrome_trace}")
                print("\nTo visualize the trace:")
                print("1. Open Chrome or Edge browser")
                print("2. Navigate to chrome://tracing or edge://tracing")
                print("3. Click 'Load' and select the generated JSON file")
                print("\nAlternatively, visit https://ui.perfetto.dev/ and drag the JSON file there")
            except Exception as e:
                print(f"Error converting trace: {e}", file=sys.stderr)
        
        # Clean up temporary file if not keeping raw trace
        if not output_trace and self.temp_trace_file:
            try:
                os.unlink(self.temp_trace_file)
            except:
                pass
        
        return return_code
    
    def cleanup_temp_files(self):
        """Clean up temporary files"""
        if self.temp_trace_file and os.path.exists(self.temp_trace_file):
            try:
                os.unlink(self.temp_trace_file)
            except:
                pass
    
    def convert_trace(self, input_file, output_file):
        """Convert existing CUPTI trace to Chrome format"""
        
        if not os.path.exists(input_file):
            print(f"Error: Input file '{input_file}' not found", file=sys.stderr)
            return 1
        
        print(f"Converting CUPTI trace to Chrome format...")
        print(f"Input: {input_file}")
        print(f"Output: {output_file}")
        
        try:
            events = self.parse_cupti_trace(input_file)
            print(f"Parsed {len(events)} events")
            
            trace_data = {
                "traceEvents": events,
                "displayTimeUnit": "ms",
                "metadata": {
                    "tool": "gpuperf - GPU Performance Profiler",
                    "format": "Chrome Trace Format"
                }
            }
            
            with open(output_file, 'w') as f:
                json.dump(trace_data, f, indent=2)
            
            print(f"\nChrome trace file written to: {output_file}")
            print("\nTo visualize the trace:")
            print("1. Open Chrome or Edge browser")
            print("2. Navigate to chrome://tracing or edge://tracing")
            print("3. Click 'Load' and select the generated JSON file")
            print("\nAlternatively, visit https://ui.perfetto.dev/ and drag the JSON file there")
            
            return 0
        except Exception as e:
            print(f"Error converting trace: {e}", file=sys.stderr)
            return 1

def main():
    # Check if first argument is 'convert' for conversion mode
    if len(sys.argv) > 1 and sys.argv[1] == 'convert':
        parser = argparse.ArgumentParser(
            prog='gpuperf convert',
            description='Convert existing CUPTI trace to Chrome format'
        )
        parser.add_argument('mode', help='Operation mode')  # This will be 'convert'
        parser.add_argument('-i', '--input', required=True, help='Input CUPTI trace file')
        parser.add_argument('-o', '--output', default='trace.json', help='Output Chrome trace JSON file')
        args = parser.parse_args()
        
        profiler = GPUPerf()
        return profiler.convert_trace(args.input, args.output)
    
    # Regular run mode
    parser = argparse.ArgumentParser(
        description='gpuperf - GPU and CPU Performance Profiler',
        usage='gpuperf [options] command [args...]\n       gpuperf convert -i input.txt -o output.json'
    )
    
    parser.add_argument('-o', '--output', help='Save raw CUPTI trace to file')
    parser.add_argument('-c', '--chrome', help='Convert trace to Chrome format and save to file')
    parser.add_argument('-p', '--cpu-profile', help='Also capture CPU profile and save to file')
    parser.add_argument('--cpu-only', action='store_true', help='Only run CPU profiler without GPU tracing')
    parser.add_argument('command', nargs=argparse.REMAINDER, help='Command to run with profiling')
    
    args = parser.parse_args()
    
    profiler = GPUPerf()
    
    # Handle run mode
    if not args.command:
        parser.print_help()
        return 1
    
    # Use the command directly from REMAINDER
    full_command = args.command
    
    # CPU-only mode
    if args.cpu_only:
        if not profiler.cpu_profiler:
            print("Error: CPU profiler not available", file=sys.stderr)
            return 1
        
        # Start the process and immediately profile it
        try:
            target_proc = subprocess.Popen(full_command)
            target_pid = target_proc.pid
            print(f"Started target process with PID: {target_pid}")
            
            cpu_output = args.cpu_profile or f"cpu_profile_{target_pid}.txt"
            profiler.start_cpu_profiler(target_pid, cpu_output)
            
            return_code = target_proc.wait()
            profiler.stop_cpu_profiler()
            return return_code
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
    
    # Combined GPU and CPU profiling
    return profiler.run_with_trace(
        full_command, 
        output_trace=args.output, 
        chrome_trace=args.chrome,
        cpu_profile=args.cpu_profile
    )

if __name__ == '__main__':
    sys.exit(main())