//go:build ignore

#include <linux/bpf.h>
#include <bpf/bpf_helpers.h>

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 1024);
    __type(key, __u32);
    __type(value, __u64);
} function_calls SEC(".maps");

// Uprobe - attached to function entry
SEC("uprobe/bash_readline")
int uprobe_bash_readline(struct pt_regs *ctx) {
    __u32 pid = bpf_get_current_pid_tgid() >> 32;
    __u64 *count = bpf_map_lookup_elem(&function_calls, &pid);

    if (count) {
        __sync_fetch_and_add(count, 1);
    } else {
        __u64 initial = 1;
        bpf_map_update_elem(&function_calls, &pid, &initial, BPF_ANY);
    }

    return 0;
}

// USDT probe
SEC("usdt/libc:memory:malloc")
int usdt_malloc(struct pt_regs *ctx) {
    __u32 pid = bpf_get_current_pid_tgid() >> 32;
    __u64 *count = bpf_map_lookup_elem(&function_calls, &pid);

    if (count) {
        __sync_fetch_and_add(count, 1);
    } else {
        __u64 initial = 1;
        bpf_map_update_elem(&function_calls, &pid, &initial, BPF_ANY);
    }

    return 0;
}

char _license[] SEC("license") = "GPL";
