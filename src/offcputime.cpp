// SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause)
// Copyright (c) 2021 Wenbo Zhang
//
// Based on offcputime(8) from BCC by Brendan Gregg.
// 19-Mar-2021   Wenbo Zhang   Created this.

#ifdef __cplusplus
extern "C" {
#endif

#include <argp.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <bpf/libbpf.h>
#include <bpf/bpf.h>

#ifdef __cplusplus
}
#endif

#include "offcputime.h"
#include "../build/offcputime.skel.h"
#include "arg_parse.h"
#include "offcputime.hpp"
#include "utils.hpp"
#include <sstream>

static int libbpf_print_fn(enum libbpf_print_level level, const char *format, va_list args)
{
	if (level == LIBBPF_DEBUG && !env.verbose)
		return 0;
	return vfprintf(stderr, format, args);
}

// Custom deleter implementations
void OffCPUBPFDeleter::operator()(struct offcputime_bpf* obj) const {
    if (obj) {
        offcputime_bpf__destroy(obj);
    }
}

static bool probe_tp_btf(const char *name)
{
	LIBBPF_OPTS(bpf_prog_load_opts, opts, .expected_attach_type = BPF_TRACE_RAW_TP);
	struct bpf_insn insns[] = {
		{ .code = BPF_ALU64 | BPF_MOV | BPF_K, .dst_reg = BPF_REG_0, .imm = 0 },
		{ .code = BPF_JMP | BPF_EXIT },
	};
	int fd, insn_cnt = sizeof(insns) / sizeof(struct bpf_insn);

	opts.attach_btf_id = libbpf_find_vmlinux_btf_id(name, BPF_TRACE_RAW_TP);
	fd = bpf_prog_load(BPF_PROG_TYPE_TRACING, NULL, "GPL", insns, insn_cnt, &opts);
	if (fd >= 0)
		close(fd);
	return fd >= 0;
}

static bool print_header_threads()
{
	int i;
	bool printed = false;

	if (env.pids[0]) {
		printf(" PID [");
		for (i = 0; i < MAX_PID_NR && env.pids[i]; i++)
			printf("%d%s", env.pids[i], (i < MAX_PID_NR - 1 && env.pids[i + 1]) ? ", " : "]");
		printed = true;
	}

	if (env.tids[0]) {
		printf(" TID [");
		for (i = 0; i < MAX_TID_NR && env.tids[i]; i++)
			printf("%d%s", env.tids[i], (i < MAX_TID_NR - 1 && env.tids[i + 1]) ? ", " : "]");
		printed = true;
	}

	return printed;
}

static void print_headers()
{
	if (env.folded)
		return;  // Don't print headers in folded format

	printf("Tracing off-CPU time (us) of");

	if (!print_header_threads())
		printf(" all threads");

	if (env.duration < 99999999)
		printf(" for %d secs.\n", env.duration);
	else
		printf("... Hit Ctrl-C to end.\n");
}

// OffCPUTimeCollector implementation
OffCPUTimeCollector::OffCPUTimeCollector() : obj(nullptr), running(false) {}

std::string OffCPUTimeCollector::get_name() const {
    return "offcputime";
}

bool OffCPUTimeCollector::start() {
    if (running) {
        return true;
    }
    
    int err, i;
    __u8 val = 0;

    libbpf_set_print(libbpf_print_fn);

    obj.reset(offcputime_bpf__open());
    if (!obj) {
        fprintf(stderr, "failed to open BPF object\n");
        return false;
    }

    /* initialize global data (filtering options) */
    obj->rodata->user_threads_only = env.user_threads_only;
    obj->rodata->kernel_threads_only = env.kernel_threads_only;
    obj->rodata->state = env.state;
    obj->rodata->min_block_ns = env.min_block_time;
    obj->rodata->max_block_ns = env.max_block_time;

    /* User space PID and TID correspond to TGID and PID in the kernel, respectively */
    if (env.pids[0])
        obj->rodata->filter_by_tgid = true;
    if (env.tids[0])
        obj->rodata->filter_by_pid = true;

    bpf_map__set_value_size(obj->maps.stackmap,
                env.perf_max_stack_depth * sizeof(unsigned long));
    bpf_map__set_max_entries(obj->maps.stackmap, env.stack_storage_size);

    if (!probe_tp_btf("sched_switch"))
        bpf_program__set_autoload(obj->progs.sched_switch, false);
    else
        bpf_program__set_autoload(obj->progs.sched_switch_raw, false);

    err = offcputime_bpf__load(obj.get());
    if (err) {
        fprintf(stderr, "failed to load BPF programs\n");
        goto cleanup;
    }

    if (env.pids[0]) {
        /* User pids_fd points to the tgids map in the BPF program */
        int pids_fd = bpf_map__fd(obj->maps.tgids);
        for (i = 0; i < MAX_PID_NR && env.pids[i]; i++) {
            if (bpf_map_update_elem(pids_fd, &(env.pids[i]), &val, BPF_ANY) != 0) {
                fprintf(stderr, "failed to init pids map: %s\n", strerror(errno));
                goto cleanup;
            }
        }
    }
    if (env.tids[0]) {
        /* User tids_fd points to the pids map in the BPF program */
        int tids_fd = bpf_map__fd(obj->maps.pids);
        for (i = 0; i < MAX_TID_NR && env.tids[i]; i++) {
            if (bpf_map_update_elem(tids_fd, &(env.tids[i]), &val, BPF_ANY) != 0) {
                fprintf(stderr, "failed to init tids map: %s\n", strerror(errno));
                goto cleanup;
            }
        }
    }

    err = offcputime_bpf__attach(obj.get());
    if (err) {
        fprintf(stderr, "failed to attach BPF programs\n");
        goto cleanup;
    }

    symbolizer.reset(blazesym_new());
    if (!symbolizer) {
        fprintf(stderr, "Failed to create a symbolizer\n");
        goto cleanup;
    }

    running = true;
    return true;

cleanup:
    obj.reset();
    return false;
}

OffCPUData OffCPUTimeCollector::collect_data() {
    OffCPUData data;
    
    if (!running || !obj) {
        return data;
    }
    
    struct offcpu_key_t lookup_key = {}, next_key;
    int err, fd_stackid, fd_info;
    struct offcpu_val_t val;

    fd_info = bpf_map__fd(obj->maps.info);
    fd_stackid = bpf_map__fd(obj->maps.stackmap);
    
    while (!bpf_map_get_next_key(fd_info, &lookup_key, &next_key)) {
        err = bpf_map_lookup_elem(fd_info, &next_key, &val);
        if (err < 0) {
            fprintf(stderr, "failed to lookup info: %d\n", err);
            break;
        }
        lookup_key = next_key;
        if (val.delta == 0)
            continue;

        OffCPUEntry entry;
        entry.key = next_key;
        entry.val = val;
        entry.has_kernel_stack = next_key.kern_stack_id != -1;
        entry.has_user_stack = next_key.user_stack_id != -1;
        
        // Collect stack traces
        entry.user_stack.resize(env.perf_max_stack_depth);
        entry.kernel_stack.resize(env.perf_max_stack_depth);
        
        if (entry.has_user_stack) {
            if (bpf_map_lookup_elem(fd_stackid, &next_key.user_stack_id, entry.user_stack.data()) != 0) {
                entry.has_user_stack = false;
                entry.user_stack.clear();
            }
        }
        
        if (entry.has_kernel_stack) {
            if (bpf_map_lookup_elem(fd_stackid, &next_key.kern_stack_id, entry.kernel_stack.data()) != 0) {
                entry.has_kernel_stack = false;
                entry.kernel_stack.clear();
            }
        }
        
        data.entries.push_back(std::move(entry));
    }
    
    return data;
}

std::string OffCPUTimeCollector::format_data(const OffCPUData& data) {
    // For the return value, we just return a summary since the actual output
    // is printed directly to stdout by the show_stack_trace functions
    std::ostringstream oss;
    oss << "Collected " << data.entries.size() << " off-CPU entries";
    return oss.str();
}

void OffCPUTimeCollector::print_data(const OffCPUData& data) {
    if (!symbolizer) {
        return;
    }
    
    for (const auto& entry : data.entries) {
        if (env.folded) {
            /* folded stack output format */
            printf("%s", entry.val.comm);
            
            /* Print user stack first for folded format */
            if (entry.has_user_stack && !env.kernel_threads_only) {
                if (entry.user_stack.empty()) {
                    printf(";[Missed User Stack]");
                } else {
                    printf(";");
                    show_stack_trace_folded(symbolizer.get(), 
                        const_cast<__u64 *>(reinterpret_cast<const __u64 *>(entry.user_stack.data())), 
                        env.perf_max_stack_depth, entry.key.tgid, ';', true);
                }
            }
            
            /* Then print kernel stack if it exists */
            if (entry.has_kernel_stack && !env.user_threads_only) {
                /* Add delimiter between user and kernel stacks if needed */
                if (entry.has_user_stack && env.delimiter && !env.kernel_threads_only)
                    printf("-");
                    
                if (entry.kernel_stack.empty()) {
                    printf(";[Missed Kernel Stack]");
                } else {
                    printf(";");
                    show_stack_trace_folded(symbolizer.get(), 
                        const_cast<__u64 *>(reinterpret_cast<const __u64 *>(entry.kernel_stack.data())), 
                        env.perf_max_stack_depth, 0, ';', true);
                }
            }
            
            printf(" %lld\n", entry.val.delta);
        } else {
            /* standard multi-line output format */
            if (entry.has_kernel_stack && !env.user_threads_only) {
                if (entry.kernel_stack.empty()) {
                    fprintf(stderr, "    [Missed Kernel Stack]\n");
                } else {
                    show_stack_trace(symbolizer.get(), 
                        const_cast<__u64 *>(reinterpret_cast<const __u64 *>(entry.kernel_stack.data())), 
                        env.perf_max_stack_depth, 0);
                }
            }

            /* Add delimiter between kernel and user stacks if both exist and delimiter is requested */
            if (env.delimiter && entry.has_kernel_stack && entry.has_user_stack && 
                !env.user_threads_only && !env.kernel_threads_only) {
                printf("    --\n");
            }

            if (entry.has_user_stack && !env.kernel_threads_only) {
                if (entry.user_stack.empty()) {
                    fprintf(stderr, "    [Missed User Stack]\n");
                } else {
                    show_stack_trace(symbolizer.get(), 
                        const_cast<__u64 *>(reinterpret_cast<const __u64 *>(entry.user_stack.data())), 
                        env.perf_max_stack_depth, entry.key.tgid);
                }
            }

            printf("    %-16s %s (%d)\n", "-", entry.val.comm, entry.key.pid);
            printf("        %lld\n\n", entry.val.delta);
        }
    }
}

CollectorData OffCPUTimeCollector::get_data() {
    if (!running || !obj) {
        return CollectorData("offcputime", "", false);
    }
    
    // Print headers first (if not in folded mode)
    print_headers();
    
    // Collect the data from BPF maps
    OffCPUData data = collect_data();
    
    // Print the data directly to stdout
    print_data(data);
    
    // Also format as string for return value
    std::string formatted = format_data(data);
    
    return CollectorData("offcputime", formatted, true);
}

bool OffCPUTimeCollector::probe_tp_btf(const char *name) {
    LIBBPF_OPTS(bpf_prog_load_opts, opts, .expected_attach_type = BPF_TRACE_RAW_TP);
    struct bpf_insn insns[] = {
        { .code = BPF_ALU64 | BPF_MOV | BPF_K, .dst_reg = BPF_REG_0, .imm = 0 },
        { .code = BPF_JMP | BPF_EXIT },
    };
    int fd, insn_cnt = sizeof(insns) / sizeof(struct bpf_insn);

    opts.attach_btf_id = libbpf_find_vmlinux_btf_id(name, BPF_TRACE_RAW_TP);
    fd = bpf_prog_load(BPF_PROG_TYPE_TRACING, NULL, "GPL", insns, insn_cnt, &opts);
    if (fd >= 0)
        close(fd);
    return fd >= 0;
} 