//go:build ignore

#include <linux/bpf.h>
#include <bpf/bpf_helpers.h>

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 1024);
    __type(key, __u32);
    __type(value, __u64);
} uprobe_calls SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 1024);
    __type(key, __u32);
    __type(value, __u64);
} usdt_calls SEC(".maps");

// Uprobe - attached to add_numbers function
SEC("uprobe/add_numbers")
int uprobe_add_numbers(struct pt_regs *ctx) {
    __u32 pid = bpf_get_current_pid_tgid() >> 32;
    __u64 *count = bpf_map_lookup_elem(&uprobe_calls, &pid);

    if (count) {
        __sync_fetch_and_add(count, 1);
    } else {
        __u64 initial = 1;
        bpf_map_update_elem(&uprobe_calls, &pid, &initial, BPF_ANY);
    }

    return 0;
}

// USDT probe for add_operation
SEC("usdt")
int usdt_add_operation(struct pt_regs *ctx) {
    __u32 pid = bpf_get_current_pid_tgid() >> 32;
    __u64 *count = bpf_map_lookup_elem(&usdt_calls, &pid);

    if (count) {
        __sync_fetch_and_add(count, 1);
    } else {
        __u64 initial = 1;
        bpf_map_update_elem(&usdt_calls, &pid, &initial, BPF_ANY);
    }

    return 0;
}

char _license[] SEC("license") = "GPL";
