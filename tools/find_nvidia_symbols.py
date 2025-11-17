#!/usr/bin/env python3
"""
find_nvidia_symbols.py - Find all traceable NVIDIA kernel module symbols

This script:
1. Reads /proc/kallsyms to find ALL NVIDIA module symbols
2. Uses bpftrace -l to find ALL available kprobes
3. Compares them to find which symbols are actually traceable

Only includes modules with 'nvidia' in the name (nvidia, nvidia_uvm, nvidia_drm, nvidia_modeset).
Excludes nvme/nvmem/nvdimm which are not NVIDIA GPU modules.
"""

import subprocess
import re
from collections import defaultdict
import sys

def get_kallsyms_nvidia_symbols():
    """Read /proc/kallsyms and extract ALL NVIDIA module symbols"""
    print("[1/3] Reading /proc/kallsyms for NVIDIA symbols...")

    nvidia_symbols = defaultdict(list)

    try:
        with open('/proc/kallsyms', 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    addr, sym_type, name = parts[0], parts[1], parts[2]

                    # Check if there's a module name
                    module = None
                    if len(parts) == 4:
                        module = parts[3].strip('[]')

                    # Include only NVIDIA GPU modules - 'nvidia' prefix only
                    # nvme, nvmem, nvdimm are NOT nvidia modules
                    if module and 'nvidia' in module.lower():
                        nvidia_symbols[module].append({
                            'name': name,
                            'type': sym_type,
                            'addr': addr
                        })
    except IOError as e:
        print(f"Error reading /proc/kallsyms: {e}")
        sys.exit(1)

    total = sum(len(syms) for syms in nvidia_symbols.values())
    print(f"  Found {total} symbols across {len(nvidia_symbols)} modules")

    return nvidia_symbols

def get_bpftrace_kprobes():
    """Get all available kprobes from bpftrace"""
    print("[2/3] Querying bpftrace for all available kprobes...")

    try:
        result = subprocess.run(
            ['bpftrace', '-l', 'kprobe:*'],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            print(f"Error running bpftrace: {result.stderr}")
            sys.exit(1)

        kprobes = set()
        for line in result.stdout.splitlines():
            if line.startswith('kprobe:'):
                # Extract function name
                func_name = line.replace('kprobe:', '')
                kprobes.add(func_name)

        print(f"  Found {len(kprobes)} total kprobes available")
        return kprobes

    except subprocess.TimeoutExpired:
        print("Error: bpftrace command timed out")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: bpftrace not found. Please install bpftrace.")
        sys.exit(1)

def match_symbols_to_kprobes(kallsyms, kprobes):
    """Match kallsyms symbols with available kprobes - NO FILTERING"""
    print("[3/3] Matching symbols to kprobes (no filtering)...")

    matched = defaultdict(lambda: defaultdict(list))
    unmatched = defaultdict(list)

    for module, symbols in kallsyms.items():
        for sym in symbols:
            name = sym['name']

            if name in kprobes:
                matched[module][sym['type']].append(name)
            else:
                # Symbol exists but not traceable
                unmatched[module].append({
                    'name': name,
                    'type': sym['type']
                })

    return matched, unmatched


def main():
    print("=" * 70)
    print("NVIDIA Kernel Module Symbol Tracer (UNFILTERED)")
    print("=" * 70)
    print()

    # Step 1: Get ALL symbols from kallsyms for nv* modules
    kallsyms = get_kallsyms_nvidia_symbols()

    # Step 2: Get ALL available kprobes
    all_kprobes = get_bpftrace_kprobes()

    # Step 3: Match symbols to kprobes (NO FILTERING)
    matched, unmatched = match_symbols_to_kprobes(kallsyms, all_kprobes)

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()

    # Print matched symbols (traceable)
    print("TRACEABLE SYMBOLS BY MODULE:")
    print("-" * 70)

    total_traceable = 0
    for module in sorted(matched.keys()):
        all_funcs = []
        for sym_type, funcs in matched[module].items():
            all_funcs.extend(funcs)

        total_traceable += len(all_funcs)
        print(f"{module}: {len(all_funcs)} traceable functions")

    print(f"\nTOTAL TRACEABLE: {total_traceable} functions")

    # Print summary of untraceable symbols
    print()
    print("-" * 70)
    print("UNTRACEABLE SYMBOLS (exist but cannot kprobe):")
    print("-" * 70)

    total_untraceable = 0
    for module in sorted(unmatched.keys()):
        # Count by type
        type_counts = defaultdict(int)
        for sym in unmatched[module]:
            type_counts[sym['type']] += 1

        total = len(unmatched[module])
        total_untraceable += total
        print(f"\n{module}: {total} untraceable symbols")
        for sym_type, count in sorted(type_counts.items()):
            print(f"  type '{sym_type}': {count}")

    print(f"\nTOTAL UNTRACEABLE: {total_untraceable} symbols")

    # Generate output files
    print()
    print("=" * 70)
    print("GENERATING OUTPUT FILES")
    print("=" * 70)

    # Save full list of traceable functions
    output_file = '/tmp/nvidia_traceable_functions.txt'
    with open(output_file, 'w') as f:
        f.write("# NVIDIA Traceable Functions\n")
        f.write("# Generated by find_nvidia_symbols.py\n\n")

        for module in sorted(matched.keys()):
            f.write(f"\n### {module} ###\n")

            all_funcs = []
            for funcs in matched[module].values():
                all_funcs.extend(funcs)

            for func in sorted(all_funcs):
                f.write(f"{func}\n")

    print(f"Saved traceable functions to: {output_file}")

    # Save all functions in a simple list
    all_funcs_file = '/tmp/nvidia_all_traceable.txt'
    with open(all_funcs_file, 'w') as f:
        f.write("# All NVIDIA Traceable Functions\n")
        f.write("# Generated by find_nvidia_symbols.py\n\n")

        all_funcs = []
        for module_funcs in matched.values():
            for funcs in module_funcs.values():
                all_funcs.extend(funcs)

        for func in sorted(all_funcs):
            f.write(f"{func}\n")

    print(f"Saved all traceable functions to: {all_funcs_file}")

    # Save bpftrace template
    template_file = '/tmp/nvidia_trace_template.bt'
    with open(template_file, 'w') as f:
        f.write("#!/usr/bin/env bpftrace\n")
        f.write("/*\n")
        f.write(" * NVIDIA GPU Tracing Template\n")
        f.write(" * Auto-generated by find_nvidia_symbols.py\n")
        f.write(" */\n\n")
        f.write("BEGIN { printf(\"Tracing NVIDIA operations...\\n\"); }\n\n")

        # Add some key probes as examples
        f.write("/* Device Operations */\n")
        if 'nvidia_open' in all_funcs:
            f.write("kprobe:nvidia_open { @opens = count(); }\n")
        if 'nvidia_close' in all_funcs:
            f.write("kprobe:nvidia_close { @closes = count(); }\n")
        if 'nvidia_unlocked_ioctl' in all_funcs:
            f.write("kprobe:nvidia_unlocked_ioctl { @ioctls = count(); }\n")

        f.write("\n/* Add more probes here */\n\n")
        f.write("END {\n")
        f.write("  print(@opens); print(@closes); print(@ioctls);\n")
        f.write("}\n")

    print(f"Saved bpftrace template to: {template_file}")

    print()
    print("Done!")

if __name__ == '__main__':
    main()
