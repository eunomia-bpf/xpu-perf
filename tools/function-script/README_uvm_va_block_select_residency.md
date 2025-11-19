# Tracing uvm_va_block_select_residency

## Overview

This bpftrace script traces the `uvm_va_block_select_residency` function in the NVIDIA UVM kernel module, capturing performance metrics and argument summaries.

## Function Signature

```c
uvm_processor_id_t uvm_va_block_select_residency(
    uvm_va_block_t *va_block,                      // arg0
    uvm_va_block_context_t *va_block_context,      // arg1
    uvm_page_index_t page_index,                   // arg2
    uvm_processor_id_t processor_id,               // arg3 - requesting processor
    NvU32 access_type_mask,                        // arg4 - access type bitmap
    const uvm_va_policy_t *policy,                 // arg5
    const uvm_perf_thrashing_hint_t *thrashing_hint, // arg6
    uvm_service_operation_t operation,             // arg7 - operation type
    const bool hmm_migratable,                     // arg8
    bool *read_duplicate)                          // arg9
```

## Captured Metrics

The script captures and aggregates:

1. **Call Count**: Total number of times the function is called
2. **Latency Distribution**: Histogram of function execution times (in nanoseconds)
3. **Processor ID Distribution**: Which processors are requesting residency selection
4. **New Residency Distribution**: Where pages are being placed
5. **Access Type Mask Distribution**: Types of memory access patterns
   - `0x1` = READ
   - `0x2` = WRITE
   - `0x4` = ATOMIC
   - `0x8` = PREFETCH
6. **Page Index Distribution**: Distribution of page indices being processed
7. **Migration Patterns**: Tracks when processor_id != new_residency (actual migrations)

## Usage

### Basic Usage

```bash
sudo bpftrace trace_uvm_va_block_select_residency.bt
```

### With Timeout

```bash
sudo timeout 10 bpftrace trace_uvm_va_block_select_residency.bt
```

### Run in Background

```bash
sudo bpftrace trace_uvm_va_block_select_residency.bt > output.txt 2>&1 &
```

## Detailed Tracing Mode

To enable detailed per-call tracing, edit the script and uncomment the printf section around line 81:

```bpftrace
printf("%lu: page_idx=%d proc_id=%d->%d access=0x%x lat=%dus\n",
       elapsed / 1000000,
       $page_index,
       $processor_id,
       $new_residency,
       $access_type_mask,
       $duration_ns / 1000);
```

## Example Output

```
=== Summary Statistics ===
Total calls:
@call_count: 1510559

=== Overall Latency Distribution (nanoseconds) ===
@latency_ns:
[128, 256)       1401077 |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
[256, 512)        103029 |@@@                                                 |
[512, 1K)           6283 |                                                    |

=== Processor ID Distribution (Requester) ===
@processor_id_dist[1]: 1510559

=== New Residency Distribution (Destination) ===
@new_residency_dist[1]: 1510559

=== Access Type Mask Distribution ===
@access_type_mask_dist[2]: 768408   # WRITE
@access_type_mask_dist[4]: 742151   # ATOMIC

=== No Migration (same processor) ===
@no_migration: 1510559
```

## Interpretation

From the example output above:

- **1.5M calls** to select_residency in 10 seconds
- **Latency**: Most calls (93%) complete in 128-256ns
- **Single GPU**: All requests from processor ID 1 (GPU)
- **No Migrations**: All pages stay on the requesting processor
- **Access Pattern**: Nearly equal split between WRITE (51%) and ATOMIC (49%) operations
- **Page Distribution**: Most activity on higher page indices (256-512 range)

## Use Cases

1. **Performance Analysis**: Identify latency bottlenecks in residency selection
2. **Migration Patterns**: Understand memory migration behavior between CPU/GPU
3. **Access Pattern Analysis**: See what types of memory operations trigger residency selection
4. **Debugging**: Trace specific page migration decisions

## Notes

- Requires root/sudo privileges
- Works with NVIDIA UVM driver (nvidia-uvm module must be loaded)
- Low overhead tracing suitable for production systems
- Args beyond arg5 (arg7, arg8) cannot be directly captured on x86_64 due to bpftrace limitations
