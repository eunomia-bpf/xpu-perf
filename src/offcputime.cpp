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
#include "offcputime.skel.h"
#include "arg_parse.h"
#include "collector_interface.hpp"
#include <sstream>

static int libbpf_print_fn(enum libbpf_print_level level, const char *format, va_list args)
{
	if (level == LIBBPF_DEBUG && !env.verbose)
		return 0;
	return vfprintf(stderr, format, args);
}

static void sig_handler(int sig)
{
}

static struct blazesym *symbolizer;

static void print_map(struct offcputime_bpf *obj)
{
	struct offcpu_key_t lookup_key = {}, next_key;
	int err, fd_stackid, fd_info;
	unsigned long *ip;
	struct offcpu_val_t val;
	int idx;
	bool has_kernel_stack, has_user_stack;

	ip = static_cast<unsigned long*>(calloc(env.perf_max_stack_depth, sizeof(*ip)));
	if (!ip) {
		fprintf(stderr, "failed to alloc ip\n");
		return;
	}

	fd_info = bpf_map__fd(obj->maps.info);
	fd_stackid = bpf_map__fd(obj->maps.stackmap);
	while (!bpf_map_get_next_key(fd_info, &lookup_key, &next_key)) {
		idx = 0;

		err = bpf_map_lookup_elem(fd_info, &next_key, &val);
		if (err < 0) {
			fprintf(stderr, "failed to lookup info: %d\n", err);
			goto cleanup;
		}
		lookup_key = next_key;
		if (val.delta == 0)
			continue;

		has_kernel_stack = next_key.kern_stack_id != -1;
		has_user_stack = next_key.user_stack_id != -1;

		if (env.folded) {
			/* folded stack output format */
			printf("%s", val.comm);
			
			/* Print user stack first for folded format */
			if (has_user_stack && !env.kernel_threads_only) {
				if (bpf_map_lookup_elem(fd_stackid, &next_key.user_stack_id, ip) != 0) {
					printf(";[Missed User Stack]");
				} else {
					printf(";");
					show_stack_trace_folded(symbolizer, reinterpret_cast<__u64 *>(ip), env.perf_max_stack_depth, next_key.tgid, ';', true);
				}
			}
			
			/* Then print kernel stack if it exists */
			if (has_kernel_stack && !env.user_threads_only) {
				/* Add delimiter between user and kernel stacks if needed */
				if (has_user_stack && env.delimiter && !env.kernel_threads_only)
					printf("-");
					
				if (bpf_map_lookup_elem(fd_stackid, &next_key.kern_stack_id, ip) != 0) {
					printf(";[Missed Kernel Stack]");
				} else {
					printf(";");
					show_stack_trace_folded(symbolizer, reinterpret_cast<__u64 *>(ip), env.perf_max_stack_depth, 0, ';', true);
				}
			}
			
			printf(" %lld\n", val.delta);
		} else {
			/* standard multi-line output format */
			if (has_kernel_stack && !env.user_threads_only) {
				if (bpf_map_lookup_elem(fd_stackid, &next_key.kern_stack_id, ip) != 0) {
					fprintf(stderr, "    [Missed Kernel Stack]\n");
				} else {
					show_stack_trace(symbolizer, reinterpret_cast<__u64 *>(ip), env.perf_max_stack_depth, 0);
				}
			}

			/* Add delimiter between kernel and user stacks if both exist and delimiter is requested */
			if (env.delimiter && has_kernel_stack && has_user_stack && 
				!env.user_threads_only && !env.kernel_threads_only) {
				printf("    --\n");
			}

			if (has_user_stack && !env.kernel_threads_only) {
				if (bpf_map_lookup_elem(fd_stackid, &next_key.user_stack_id, ip) != 0) {
					fprintf(stderr, "    [Missed User Stack]\n");
				} else {
					show_stack_trace(symbolizer, reinterpret_cast<__u64 *>(ip), env.perf_max_stack_depth, next_key.tgid);
				}
			}

			printf("    %-16s %s (%d)\n", "-", val.comm, next_key.pid);
			printf("        %lld\n\n", val.delta);
		}
	}

cleanup:
	free(ip);
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

class OffCPUTimeCollector : public ICollector {
private:
    struct offcputime_bpf *obj;
    bool running;
    
public:
    OffCPUTimeCollector() : obj(nullptr), running(false) {}
    
    ~OffCPUTimeCollector() {
        if (obj) {
            blazesym_free(symbolizer);
            offcputime_bpf__destroy(obj);
        }
    }
    
    std::string get_name() const override {
        return "offcputime";
    }
    
    bool start() override {
        if (running) {
            return true;
        }
        
        int err, i;
        __u8 val = 0;

        libbpf_set_print(libbpf_print_fn);

        obj = offcputime_bpf__open();
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

        err = offcputime_bpf__load(obj);
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

        err = offcputime_bpf__attach(obj);
        if (err) {
            fprintf(stderr, "failed to attach BPF programs\n");
            goto cleanup;
        }

        symbolizer = blazesym_new();
        if (!symbolizer) {
            fprintf(stderr, "Failed to create a symbolizer\n");
            goto cleanup;
        }

        running = true;
        return true;

cleanup:
        if (obj) {
            offcputime_bpf__destroy(obj);
            obj = nullptr;
        }
        return false;
    }
    
    CollectorData get_data() override {
        if (!running || !obj) {
            return CollectorData("offcputime", "", false);
        }
        
        // Capture output to string
        std::ostringstream output;
        FILE* old_stdout = stdout;
        
        // Create a pipe to capture stdout
        int pipefd[2];
        if (pipe(pipefd) == -1) {
            return CollectorData("offcputime", "Failed to create pipe", false);
        }
        
        // Redirect stdout to pipe
        fflush(stdout);
        dup2(pipefd[1], STDOUT_FILENO);
        close(pipefd[1]);
        
        // Print data
        print_map(obj);
        
        // Restore stdout
        fflush(stdout);
        dup2(fileno(old_stdout), STDOUT_FILENO);
        
        // Read from pipe
        char buffer[4096];
        std::string result;
        ssize_t count;
        while ((count = read(pipefd[0], buffer, sizeof(buffer))) > 0) {
            result.append(buffer, count);
        }
        close(pipefd[0]);
        
        return CollectorData("offcputime", result, true);
    }
};

int main(int argc, char **argv)
{
	int err;

	err = parse_common_args(argc, argv, TOOL_OFFCPUTIME);
	if (err)
		return err;

	err = validate_common_args();
	if (err)
		return err;

	OffCPUTimeCollector collector;
	
	if (!collector.start()) {
		fprintf(stderr, "Failed to start offcputime collector\n");
		return 1;
	}

	signal(SIGINT, sig_handler);

	print_headers();

	sleep(env.duration);

	/* Get traces from info map and print them to stdout */
	CollectorData data = collector.get_data();
	if (data.success) {
		printf("%s", data.output.c_str());
	}

	return 0;
} 