#!/usr/bin/env python3
"""
Run all NVIDIA bpftrace tests and log results
"""

import subprocess
import time
import os
import sys
from datetime import datetime

# Configuration
SCRIPT_DIR = "/home/yunwei37/workspace/xpu-perf/tools/bpftrace-script"
TEST_APP = "/home/yunwei37/workspace/xpu-perf/test/mock-app/vectorAdd"
LOG_FILE = "/home/yunwei37/workspace/xpu-perf/tools/nvidia_test_results.log"

# List of test scripts to run (split version - default)
TEST_SCRIPTS_SPLIT = [
    "test_nv_drm.bt",
    "test_nv_1.bt",
    "test_nv_2.bt",
    "test_nvidia.bt",
    "test_uvm_1.bt",
    "test_uvm_2.bt",
    "test_uvm_3.bt",
    "test_uvm_4.bt",
    "test_nvkms.bt",
    "test_nvuvm.bt",
]

# Monolithic versions (use with --monolithic flag)
TEST_SCRIPTS_MONOLITHIC = [
    "test_nv_drm.bt",
    "test_nv.bt",        # Monolithic nv_* test
    "test_nvidia.bt",
    "test_uvm.bt",       # Monolithic uvm_* test
    "test_nvkms.bt",
    "test_nvuvm.bt",
]

def run_test(script_name):
    """Run a single bpftrace test"""
    script_path = os.path.join(SCRIPT_DIR, script_name)

    if not os.path.exists(script_path):
        return None, f"Script not found: {script_path}"

    print(f"Running {script_name}...")

    # Determine environment variables based on script
    env = os.environ.copy()
    # Monolithic scripts need higher limits
    if script_name in ["test_nv.bt", "test_uvm.bt"]:
        env["BPFTRACE_MAX_BPF_PROGS"] = "3000"
        env["BPFTRACE_MAX_PROBES"] = "3000"
    elif "uvm" in script_name or "nv_" in script_name:
        env["BPFTRACE_MAX_BPF_PROGS"] = "1000"
        env["BPFTRACE_MAX_PROBES"] = "1000"

    start_time = time.time()

    try:
        # Monolithic scripts need more time
        timeout = 300 if script_name in ["test_nv.bt", "test_uvm.bt"] else 120

        # Build command with env vars for sudo
        cmd = ["sudo"]
        if script_name in ["test_nv.bt", "test_uvm.bt"]:
            cmd.extend([
                "BPFTRACE_MAX_BPF_PROGS=3000",
                "BPFTRACE_MAX_PROBES=3000"
            ])
        elif "uvm" in script_name or "nv_" in script_name:
            cmd.extend([
                "BPFTRACE_MAX_BPF_PROGS=1000",
                "BPFTRACE_MAX_PROBES=1000"
            ])
        cmd.extend(["bpftrace", script_path, "-c", TEST_APP])

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )

        elapsed = time.time() - start_time

        return {
            "script": script_name,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "elapsed": elapsed,
            "success": result.returncode == 0
        }, None

    except subprocess.TimeoutExpired:
        return None, f"Timeout after 120 seconds"
    except Exception as e:
        return None, f"Error: {str(e)}"

def format_result(result):
    """Format test result for logging"""
    if result is None:
        return ""

    output = []
    output.append("=" * 80)
    output.append(f"TEST: {result['script']}")
    output.append(f"Status: {'PASSED' if result['success'] else 'FAILED'}")
    output.append(f"Elapsed: {result['elapsed']:.2f} seconds")
    output.append("=" * 80)

    # Only include relevant output (skip warnings)
    stdout_lines = result['stdout'].split('\n')
    in_results = False
    for line in stdout_lines:
        if '===' in line or 'Tracing' in line or '@calls' in line or 'Vector' in line or 'Test PASSED' in line:
            in_results = True
        if in_results:
            output.append(line)

    output.append("")
    return "\n".join(output)

def main():
    # Show help
    if "--help" in sys.argv or "-h" in sys.argv:
        print("Usage: python3 run_all_nvidia_tests.py [OPTIONS]")
        print()
        print("Options:")
        print("  -m, --monolithic    Use monolithic test scripts (test_nv.bt, test_uvm.bt)")
        print("                      WARNING: May produce many warnings but covers all functions")
        print("  -h, --help          Show this help message")
        print()
        print("Default: Uses split test scripts (recommended, cleaner output)")
        return

    # Parse command line arguments
    use_monolithic = "--monolithic" in sys.argv or "-m" in sys.argv

    # Select test scripts based on flag
    if use_monolithic:
        TEST_SCRIPTS = TEST_SCRIPTS_MONOLITHIC
        test_mode = "MONOLITHIC (may have warnings)"
    else:
        TEST_SCRIPTS = TEST_SCRIPTS_SPLIT
        test_mode = "SPLIT (recommended)"

    print("=" * 80)
    print("NVIDIA Kernel Function Test Suite")
    print("=" * 80)
    print(f"Test Mode: {test_mode}")
    print(f"Test Application: {TEST_APP}")
    print(f"Number of Tests: {len(TEST_SCRIPTS)}")
    print(f"Log File: {LOG_FILE}")
    print("=" * 80)
    print()

    # Open log file
    with open(LOG_FILE, 'w') as log:
        # Write header
        log.write("=" * 80 + "\n")
        log.write("NVIDIA Kernel Function Test Results\n")
        log.write(f"Test Mode: {test_mode}\n")
        log.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write(f"Test Application: {TEST_APP}\n")
        log.write("=" * 80 + "\n\n")

        results = []
        errors = []

        # Run all tests
        for script in TEST_SCRIPTS:
            result, error = run_test(script)

            if error:
                print(f"  ❌ {script}: {error}")
                errors.append((script, error))
                log.write(f"FAILED: {script}\n")
                log.write(f"Error: {error}\n\n")
            else:
                # Monolithic scripts work but produce warnings in stderr
                # Check if they actually completed successfully
                is_monolithic = script in ["test_nv.bt", "test_uvm.bt"]
                actual_success = result['returncode'] == 0 and "Test PASSED" in result['stdout']

                if is_monolithic and actual_success:
                    result['success'] = True

                status = "✓" if result['success'] else "✗"
                print(f"  {status} {script}: {result['elapsed']:.2f}s")
                results.append(result)
                log.write(format_result(result))

        # Write summary
        print()
        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Total Tests: {len(TEST_SCRIPTS)}")
        print(f"Passed: {len([r for r in results if r['success']])}")
        print(f"Failed: {len([r for r in results if not r['success']])}")
        print(f"Errors: {len(errors)}")

        log.write("\n" + "=" * 80 + "\n")
        log.write("SUMMARY\n")
        log.write("=" * 80 + "\n")
        log.write(f"Total Tests: {len(TEST_SCRIPTS)}\n")
        log.write(f"Passed: {len([r for r in results if r['success']])}\n")
        log.write(f"Failed: {len([r for r in results if not r['success']])}\n")
        log.write(f"Errors: {len(errors)}\n")

        if errors:
            log.write("\nERRORS:\n")
            for script, error in errors:
                log.write(f"  {script}: {error}\n")

    print(f"\nResults saved to: {LOG_FILE}")

if __name__ == '__main__':
    main()
