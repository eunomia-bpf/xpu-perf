// Example eBPF program that demonstrates tail calling into custom__generic
// Compile: clang -O2 -target bpf -c example_tailcall.bpf.c -o example_tailcall.bpf.o

#include "vmlinux.h"
#include <bpf/bpf_helpers.h>

// Program array map for tail calls
// This will be populated with the custom__generic program FD from Go
struct {
    __uint(type, BPF_MAP_TYPE_PROG_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, __u32);
} prog_array SEC(".maps");

// External reference to custom_context_map from the profiler
// This map is shared with custom__generic and managed by the Go side
// The Go code will reuse the map FD from the profiler
struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, __u64);
} custom_context_map SEC(".maps");

// Uprobe entry point that captures context_id and tail calls to custom__generic
SEC("uprobe/example_function")
int uprobe_example(struct pt_regs *ctx)
{
    // Read context_id from function's first argument
    // For x86_64: first arg is in RDI (ctx->di)
    __u64 context_id = (__u64)ctx->di;

    // Store context_value in the shared per-CPU map for custom__generic to retrieve
    __u32 key = 0;
    bpf_map_update_elem(&custom_context_map, &key, &context_id, BPF_ANY);

    // Tail call to custom__generic program
    // Index 0 in our prog_array will contain the custom__generic FD
    bpf_tail_call(ctx, &prog_array, 0);

    // If tail call fails, return
    return 0;
}

char LICENSE[] SEC("license") = "GPL";
