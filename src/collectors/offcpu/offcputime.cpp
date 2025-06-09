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

#include "collectors/bpf_event.h"
#include "offcputime.skel.h"
#include "offcputime.hpp"
#include "collectors/utils.hpp"
#include "collectors/sampling_printer.hpp"
#include <sstream>

static int libbpf_print_fn(enum libbpf_print_level level, const char *format, va_list args)
{
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

static bool print_header_threads(const Config& config)
{
	int i;
	bool printed = false;

	if (config.pids[0]) {
		printf(" PID [");
		for (i = 0; i < MAX_PID_NR && config.pids[i]; i++)
			printf("%d%s", config.pids[i], (i < MAX_PID_NR - 1 && config.pids[i + 1]) ? ", " : "]");
		printed = true;
	}

	if (config.tids[0]) {
		printf(" TID [");
		for (i = 0; i < MAX_TID_NR && config.tids[i]; i++)
			printf("%d%s", config.tids[i], (i < MAX_TID_NR - 1 && config.tids[i + 1]) ? ", " : "]");
		printed = true;
	}

	return printed;
}

static void print_headers(const Config& config)
{
	if (config.folded)
		return;  // Don't print headers in folded format

	printf("Tracing off-CPU time (us) of");

	if (!print_header_threads(config))
		printf(" all threads");

	if (config.duration < 99999999)
		printf(" for %d secs.\n", config.duration);
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
    obj->rodata->user_threads_only = config.user_threads_only;
    obj->rodata->kernel_threads_only = config.kernel_threads_only;
    obj->rodata->state = config.state;
    obj->rodata->min_block_ns = config.min_block_time;
    obj->rodata->max_block_ns = config.max_block_time;

    /* User space PID and TID correspond to TGID and PID in the kernel, respectively */
    if (config.pids[0])
        obj->rodata->filter_by_tgid = true;
    if (config.tids[0])
        obj->rodata->filter_by_pid = true;

    bpf_map__set_value_size(obj->maps.stackmap,
                config.perf_max_stack_depth * sizeof(unsigned long));
    bpf_map__set_max_entries(obj->maps.stackmap, config.stack_storage_size);

    if (!probe_tp_btf("sched_switch"))
        bpf_program__set_autoload(obj->progs.sched_switch, false);
    else
        bpf_program__set_autoload(obj->progs.sched_switch_raw, false);

    err = offcputime_bpf__load(obj.get());
    if (err) {
        fprintf(stderr, "failed to load BPF programs\n");
        goto cleanup;
    }

    if (config.pids[0]) {
        /* User pids_fd points to the tgids map in the BPF program */
        int pids_fd = bpf_map__fd(obj->maps.tgids);
        for (i = 0; i < MAX_PID_NR && config.pids[i]; i++) {
            if (bpf_map_update_elem(pids_fd, &(config.pids[i]), &val, BPF_ANY) != 0) {
                fprintf(stderr, "failed to init pids map: %s\n", strerror(errno));
                goto cleanup;
            }
        }
    }
    if (config.tids[0]) {
        /* User tids_fd points to the pids map in the BPF program */
        int tids_fd = bpf_map__fd(obj->maps.pids);
        for (i = 0; i < MAX_TID_NR && config.tids[i]; i++) {
            if (bpf_map_update_elem(tids_fd, &(config.tids[i]), &val, BPF_ANY) != 0) {
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
    
    struct sample_key_t lookup_key = {}, next_key;
    int err, fd_stackid, fd_info;
    unsigned long long val;

    fd_info = bpf_map__fd(obj->maps.info);
    fd_stackid = bpf_map__fd(obj->maps.stackmap);
    
    while (!bpf_map_get_next_key(fd_info, &lookup_key, &next_key)) {
        err = bpf_map_lookup_elem(fd_info, &next_key, &val);
        if (err < 0) {
            fprintf(stderr, "failed to lookup info: %d\n", err);
            break;
        }
        lookup_key = next_key;
        if (val == 0)
            continue;

        SamplingEntry entry;
        entry.key = next_key;
        entry.value = val;
        entry.has_kernel_stack = next_key.kern_stack_id != -1;
        entry.has_user_stack = next_key.user_stack_id != -1;
        
        // Collect stack traces
        entry.user_stack.resize(config.perf_max_stack_depth);
        entry.kernel_stack.resize(config.perf_max_stack_depth);
        
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
    return SamplingPrinter::format_data(data, "off-CPU");
}

void OffCPUTimeCollector::print_data(const OffCPUData& data) {
    SamplingPrinter::print_data(data, symbolizer.get(), config, "us");
}

CollectorData OffCPUTimeCollector::get_data() {
    if (!running || !obj) {
        return CollectorData("offcputime", "", false);
    }
    
    // Print headers first (if not in folded mode)
    print_headers(config);
    
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