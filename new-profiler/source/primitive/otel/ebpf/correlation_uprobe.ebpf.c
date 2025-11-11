// eBPF program to capture CUPTI correlation IDs from XpuPerfGetCorrelationId
// This program attaches to the XpuPerfGetCorrelationId function in the CUPTI library
// and captures the correlation ID parameter, passing it to the custom trace handler.

#include <linux/bpf.h>
#include <linux/ptrace.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

// Forward declaration of the custom_context_map from the profiler
// This map is shared with the main profiler to pass correlation IDs
struct {
	__uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
	__type(key, __u32);
	__type(value, __u64);
	__uint(max_entries, 1);
} custom_context_map SEC(".maps");

// Program array map for tail calling to custom__generic
// This will be populated with the custom__generic program FD from Go
struct {
	__uint(type, BPF_MAP_TYPE_PROG_ARRAY);
	__type(key, __u32);
	__type(value, __u32);
	__uint(max_entries, 1);
} prog_array SEC(".maps");

// Capture correlation ID from XpuPerfGetCorrelationId function
// Function signature: uint32_t XpuPerfGetCorrelationId(uint32_t correlationId)
// The correlation ID is passed as the first parameter (RDI on x86_64)
//
// This uprobe stores the correlation ID and tail calls to custom__generic
// which will collect the stack trace with the correlation ID attached.
SEC("uprobe/XpuPerfGetCorrelationId")
int capture_correlation_id(struct pt_regs *ctx)
{
	// Read correlation_id from function's first argument
	// For x86_64: first arg is in RDI (PT_REGS_PARM1)
	__u64 correlation_id = (__u64)PT_REGS_PARM1(ctx);

	// Store context_value in the shared per-CPU map for custom__generic to retrieve
	__u32 key = 0;
	bpf_map_update_elem(&custom_context_map, &key, &correlation_id, BPF_ANY);

	// Tail call to custom__generic program
	// Index 0 in our prog_array will contain the custom__generic FD
	bpf_tail_call(ctx, &prog_array, 0);

	// If tail call fails, return
	return 0;
}

char LICENSE[] SEC("license") = "Dual BSD/GPL";
