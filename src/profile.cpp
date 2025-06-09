// SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause)
/*
 * profile    Profile CPU usage by sampling stack traces at a timed interval.
 * Copyright (c) 2022 LG Electronics
 *
 * Based on profile from BCC by Brendan Gregg and others.
 * 28-Dec-2021   Eunseon Lee   Created this.
 */

#ifdef __cplusplus
extern "C" {
#endif

#include <argp.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <unistd.h>
#include <time.h>
#include <linux/perf_event.h>
#include <asm/unistd.h>
#include <bpf/libbpf.h>
#include <bpf/bpf.h>
#include <sys/stat.h>
#include <string.h>

#ifdef __cplusplus
}
#endif

#include "profile.h"
#include "../build/profile.skel.h"
#include "arg_parse.h"
#include "profile.hpp"
#include "common.h"
#include <sstream>
#include <algorithm>

#define SYM_INFO_LEN			2048

/*
 * -EFAULT in get_stackid normally means the stack-trace is not available,
 * such as getting kernel stack trace in user mode
 */
#define STACK_ID_EFAULT(stack_id)	(stack_id == -EFAULT)

#define STACK_ID_ERR(stack_id)		((stack_id < 0) && !STACK_ID_EFAULT(stack_id))

/* hash collision (-EEXIST) suggests that stack map size may be too small */
#define CHECK_STACK_COLLISION(ustack_id, kstack_id)	\
	(kstack_id == -EEXIST || ustack_id == -EEXIST)

#define MISSING_STACKS(ustack_id, kstack_id)	\
	(!env.user_stacks_only && STACK_ID_ERR(kstack_id)) + (!env.kernel_stacks_only && STACK_ID_ERR(ustack_id))

/* This structure combines profile_key_t and count which should be sorted together */
struct key_ext_t {
	struct profile_key_t k;
	__u64 v;
};

static int nr_cpus;

// Custom deleter implementations
void ProfileBPFDeleter::operator()(struct profile_bpf* obj) const {
    if (obj) {
        profile_bpf__destroy(obj);
    }
}

void BlazesymDeleter::operator()(struct blazesym* sym) const {
    if (sym) {
        blazesym_free(sym);
    }
}

void BPFLinkDeleter::operator()(struct bpf_link* link) const {
    if (link) {
        bpf_link__destroy(link);
    }
}

static int libbpf_print_fn(enum libbpf_print_level level, const char *format, va_list args)
{
	if (level == LIBBPF_DEBUG && !env.verbose)
		return 0;

	return vfprintf(stderr, format, args);
}

static int cmp_counts(const void *a, const void *b)
{
	const __u64 x = (static_cast<const struct key_ext_t *>(a))->v;
	const __u64 y = (static_cast<const struct key_ext_t *>(b))->v;

	/* descending order */
	return y - x;
}

static int read_counts_map(int fd, struct key_ext_t *items, __u32 *count)
{
	struct profile_key_t empty = {};
	struct profile_key_t *lookup_key = &empty;
	int i = 0;
	int err;

	while (bpf_map_get_next_key(fd, lookup_key, &items[i].k) == 0) {
		err = bpf_map_lookup_elem(fd, &items[i].k, &items[i].v);
		if (err < 0) {
			fprintf(stderr, "failed to lookup counts: %d\n", err);
			return -err;
		}

		if (items[i].v == 0)
			continue;

		lookup_key = &items[i].k;
		i++;
	}

	*count = i;
	return 0;
}

static int print_count(struct profile_key_t *event, __u64 count, int stack_map, struct blazesym *symbolizer)
{
	unsigned long *ip;
	bool has_kernel_stack, has_user_stack;

	ip = static_cast<unsigned long*>(calloc(env.perf_max_stack_depth, sizeof(unsigned long)));
	if (!ip) {
		fprintf(stderr, "failed to alloc ip\n");
		return -ENOMEM;
	}

	has_kernel_stack = !STACK_ID_EFAULT(event->kern_stack_id);
	has_user_stack = !STACK_ID_EFAULT(event->user_stack_id);

	if (!env.folded) {
		/* multi-line stack output */
		/* Show kernel stack first */
		if (!env.user_stacks_only && has_kernel_stack) {
			if (bpf_map_lookup_elem(stack_map, &event->kern_stack_id, ip) != 0) {
				fprintf(stderr, "    [Missed Kernel Stack]\n");
			} else {
				show_stack_trace(symbolizer, reinterpret_cast<__u64 *>(ip), env.perf_max_stack_depth, 0);
			}
		}

		if (env.delimiter && !env.user_stacks_only && !env.kernel_stacks_only &&
		    has_user_stack && has_kernel_stack) {
			printf("    --\n");
		}

		/* Then show user stack */
		if (!env.kernel_stacks_only && has_user_stack) {
			if (bpf_map_lookup_elem(stack_map, &event->user_stack_id, ip) != 0) {
				fprintf(stderr, "    [Missed User Stack]\n");
			} else {
				show_stack_trace(symbolizer, reinterpret_cast<__u64 *>(ip), env.perf_max_stack_depth, event->pid);
			}
		}

		printf("    %-16s %s (%d)\n", "-", event->name, event->pid);
		printf("        %lld\n", count);
	} else {
		/* folded stack output */
		printf("%s", event->name);
		
		/* Print user stack first for folded format */
		if (has_user_stack && !env.kernel_stacks_only) {
			if (bpf_map_lookup_elem(stack_map, &event->user_stack_id, ip) != 0) {
				printf(";[Missed User Stack]");
			} else {
				printf(";");
				show_stack_trace_folded(symbolizer, reinterpret_cast<__u64 *>(ip), env.perf_max_stack_depth, event->pid, ';', true);
			}
		}
		
		/* Then print kernel stack if it exists */
		if (has_kernel_stack && !env.user_stacks_only) {
			/* Add delimiter between user and kernel stacks if needed */
			if (has_user_stack && env.delimiter && !env.kernel_stacks_only)
				printf("-");
				
			if (bpf_map_lookup_elem(stack_map, &event->kern_stack_id, ip) != 0) {
				printf(";[Missed Kernel Stack]");
			} else {
				printf(";");
				show_stack_trace_folded(symbolizer, reinterpret_cast<__u64 *>(ip), env.perf_max_stack_depth, 0, ';', true);
			}
		}
		
		printf(" %lld\n", count);
	}

	free(ip);

	return 0;
}

static int print_counts(int counts_map, int stack_map, struct blazesym *symbolizer)
{
	struct key_ext_t *counts;
	struct profile_key_t *event;
	__u64 count;
	__u32 nr_count = MAX_ENTRIES;
	size_t nr_missing_stacks = 0;
	bool has_collision = false;
	int i, ret = 0;

	counts = static_cast<struct key_ext_t*>(calloc(MAX_ENTRIES, sizeof(struct key_ext_t)));
	if (!counts) {
		fprintf(stderr, "Out of memory\n");
		return -ENOMEM;
	}

	ret = read_counts_map(counts_map, counts, &nr_count);
	if (ret)
		goto cleanup;

	qsort(counts, nr_count, sizeof(struct key_ext_t), cmp_counts);

	for (i = 0; i < static_cast<int>(nr_count); i++) {
		event = &counts[i].k;
		count = counts[i].v;

		print_count(event, count, stack_map, symbolizer);
		
		/* Add a newline between stack traces for better readability */
		if (!env.folded && i < static_cast<int>(nr_count) - 1)
			printf("\n");

		/* handle stack id errors */
		nr_missing_stacks += MISSING_STACKS(event->user_stack_id, event->kern_stack_id);
		has_collision = CHECK_STACK_COLLISION(event->user_stack_id, event->kern_stack_id);
	}

	if (nr_missing_stacks > 0) {
		fprintf(stderr, "WARNING: %zu stack traces could not be displayed.%s\n",
			nr_missing_stacks, has_collision ?
			" Consider increasing --stack-storage-size.":"");
	}

cleanup:
	free(counts);

	return ret;
}

static void print_headers()
{
	int i;

	if (env.folded)
		return;  // Don't print headers in folded format

	printf("Sampling at %d Hertz of", env.sample_freq);

	if (env.pids[0]) {
		printf(" PID [");
		for (i = 0; i < MAX_PID_NR && env.pids[i]; i++)
			printf("%d%s", env.pids[i], (i < MAX_PID_NR - 1 && env.pids[i + 1]) ? ", " : "]");
	} else if (env.tids[0]) {
		printf(" TID [");
		for (i = 0; i < MAX_TID_NR && env.tids[i]; i++)
			printf("%d%s", env.tids[i], (i < MAX_TID_NR - 1 && env.tids[i + 1]) ? ", " : "]");
	} else {
		printf(" all threads");
	}

	if (env.user_stacks_only)
		printf(" by user");
	else if (env.kernel_stacks_only)
		printf(" by kernel");
	else
		printf(" by user + kernel");

	if (env.cpu != -1)
		printf(" on CPU#%d", env.cpu);

	if (env.duration < INT_MAX)
		printf(" for %d secs.\n", env.duration);
	else
		printf("... Hit Ctrl-C to end.\n");
}

// ProfileCollector implementation
ProfileCollector::ProfileCollector() : obj(nullptr), running(false) {
    nr_cpus = 0;
}

std::string ProfileCollector::get_name() const {
    return "profile";
}

bool ProfileCollector::start() {
    if (running) {
        return true;
    }
    
    int err, i;
    __u8 val = 0;

    libbpf_set_print(libbpf_print_fn);

    nr_cpus = libbpf_num_possible_cpus();
    if (nr_cpus < 0) {
        printf("failed to get # of possible cpus: '%s'!\n",
               strerror(-nr_cpus));
        return false;
    }
    if (nr_cpus > MAX_CPU_NR) {
        fprintf(stderr, "the number of cpu cores is too big, please "
            "increase MAX_CPU_NR's value and recompile");
        return false;
    }

    symbolizer.reset(blazesym_new());
    if (!symbolizer) {
        fprintf(stderr, "Failed to create a symbolizer\n");
        return false;
    }

    obj.reset(profile_bpf__open());
    if (!obj) {
        fprintf(stderr, "failed to open BPF object\n");
        return false;
    }

    /* initialize global data (filtering options) */
    obj->rodata->user_stacks_only = env.user_stacks_only;
    obj->rodata->kernel_stacks_only = env.kernel_stacks_only;
    obj->rodata->include_idle = env.include_idle;
    if (env.pids[0])
        obj->rodata->filter_by_pid = true;
    else if (env.tids[0])
        obj->rodata->filter_by_tid = true;

    bpf_map__set_value_size(obj->maps.stackmap,
                env.perf_max_stack_depth * sizeof(unsigned long));
    bpf_map__set_max_entries(obj->maps.stackmap, env.stack_storage_size);

    err = profile_bpf__load(obj.get());
    if (err) {
        fprintf(stderr, "failed to load BPF programs\n");
        goto cleanup;
    }

    if (env.pids[0]) {
        int pids_fd = bpf_map__fd(obj->maps.pids);
        for (i = 0; i < MAX_PID_NR && env.pids[i]; i++) {
            if (bpf_map_update_elem(pids_fd, &(env.pids[i]), &val, BPF_ANY) != 0) {
                fprintf(stderr, "failed to init pids map: %s\n", strerror(errno));
                goto cleanup;
            }
        }
    }
    else if (env.tids[0]) {
        int tids_fd = bpf_map__fd(obj->maps.tids);
        for (i = 0; i < MAX_TID_NR && env.tids[i]; i++) {
            if (bpf_map_update_elem(tids_fd, &(env.tids[i]), &val, BPF_ANY) != 0) {
                fprintf(stderr, "failed to init tids map: %s\n", strerror(errno));
                goto cleanup;
            }
        }
    }

    err = open_and_attach_perf_event(obj->progs.do_perf_event);
    if (err)
        goto cleanup;

    running = true;
    return true;

cleanup:
    obj.reset();
    return false;
}

ProfileData ProfileCollector::collect_data() {
    ProfileData data;
    
    if (!running || !obj) {
        return data;
    }
    
    struct profile_key_t empty = {};
    struct profile_key_t *lookup_key = &empty;
    __u64 count;
    int err;
    int counts_fd = bpf_map__fd(obj->maps.counts);
    int stack_fd = bpf_map__fd(obj->maps.stackmap);
    
    // Collect all entries from the counts map
    std::vector<key_ext_t> items;
    
    struct profile_key_t key;
    while (bpf_map_get_next_key(counts_fd, lookup_key, &key) == 0) {
        err = bpf_map_lookup_elem(counts_fd, &key, &count);
        if (err < 0) {
            fprintf(stderr, "failed to lookup counts: %d\n", err);
            break;
        }

        if (count == 0) {
            lookup_key = &key;
            continue;
        }

        key_ext_t item;
        item.k = key;
        item.v = count;
        items.push_back(item);
        
        lookup_key = &key;
    }
    
    // Sort by count (descending)
    std::sort(items.begin(), items.end(), [](const key_ext_t& a, const key_ext_t& b) {
        return a.v > b.v;
    });
    
    // Convert to ProfileEntry format with stack traces
    for (const auto& item : items) {
        ProfileEntry entry;
        entry.key = item.k;
        entry.count = item.v;
        entry.has_kernel_stack = !STACK_ID_EFAULT(item.k.kern_stack_id);
        entry.has_user_stack = !STACK_ID_EFAULT(item.k.user_stack_id);
        
        // Collect stack traces
        entry.user_stack.resize(env.perf_max_stack_depth);
        entry.kernel_stack.resize(env.perf_max_stack_depth);
        
        if (entry.has_user_stack) {
            if (bpf_map_lookup_elem(stack_fd, &item.k.user_stack_id, entry.user_stack.data()) != 0) {
                entry.has_user_stack = false;
                entry.user_stack.clear();
            }
        }
        
        if (entry.has_kernel_stack) {
            if (bpf_map_lookup_elem(stack_fd, &item.k.kern_stack_id, entry.kernel_stack.data()) != 0) {
                entry.has_kernel_stack = false;
                entry.kernel_stack.clear();
            }
        }
        
        data.entries.push_back(std::move(entry));
    }
    
    return data;
}

std::string ProfileCollector::format_data(const ProfileData& data) {
    // For the return value, we just return a summary since the actual output
    // is printed directly to stdout by the show_stack_trace functions
    std::ostringstream oss;
    oss << "Collected " << data.entries.size() << " profile entries";
    return oss.str();
}

void ProfileCollector::print_data(const ProfileData& data) {
    if (!symbolizer) {
        return;
    }
    
    for (const auto& entry : data.entries) {
        if (!env.folded) {
            /* multi-line stack output */
            /* Show kernel stack first */
            if (!env.user_stacks_only && entry.has_kernel_stack) {
                if (entry.kernel_stack.empty()) {
                    fprintf(stderr, "    [Missed Kernel Stack]\n");
                } else {
                    show_stack_trace(symbolizer.get(), 
                        const_cast<__u64 *>(reinterpret_cast<const __u64 *>(entry.kernel_stack.data())), 
                        env.perf_max_stack_depth, 0);
                }
            }

            if (env.delimiter && !env.user_stacks_only && !env.kernel_stacks_only &&
                entry.has_user_stack && entry.has_kernel_stack) {
                printf("    --\n");
            }

            /* Then show user stack */
            if (!env.kernel_stacks_only && entry.has_user_stack) {
                if (entry.user_stack.empty()) {
                    fprintf(stderr, "    [Missed User Stack]\n");
                } else {
                    show_stack_trace(symbolizer.get(), 
                        const_cast<__u64 *>(reinterpret_cast<const __u64 *>(entry.user_stack.data())), 
                        env.perf_max_stack_depth, entry.key.pid);
                }
            }

            printf("    %-16s %s (%d)\n", "-", entry.key.name, entry.key.pid);
            printf("        %lld\n", entry.count);
        } else {
            /* folded stack output */
            printf("%s", entry.key.name);
            
            /* Print user stack first for folded format */
            if (entry.has_user_stack && !env.kernel_stacks_only) {
                if (entry.user_stack.empty()) {
                    printf(";[Missed User Stack]");
                } else {
                    printf(";");
                    show_stack_trace_folded(symbolizer.get(), 
                        const_cast<__u64 *>(reinterpret_cast<const __u64 *>(entry.user_stack.data())), 
                        env.perf_max_stack_depth, entry.key.pid, ';', true);
                }
            }
            
            /* Then print kernel stack if it exists */
            if (entry.has_kernel_stack && !env.user_stacks_only) {
                /* Add delimiter between user and kernel stacks if needed */
                if (entry.has_user_stack && env.delimiter && !env.kernel_stacks_only)
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
            
            printf(" %lld\n", entry.count);
        }
        
        /* Add a newline between stack traces for better readability in non-folded mode */
        if (!env.folded)
            printf("\n");
    }
}

CollectorData ProfileCollector::get_data() {
    if (!running || !obj) {
        return CollectorData("profile", "", false);
    }
    
    // Print headers first (if not in folded mode)
    if (!env.folded)
        print_headers();
    
    // Collect the data from BPF maps
    ProfileData data = collect_data();
    
    // Print the data directly to stdout
    print_data(data);
    
    // Also format as string for return value
    std::string formatted = format_data(data);
    
    return CollectorData("profile", formatted, true);
}

int ProfileCollector::open_and_attach_perf_event(struct bpf_program *prog) {
	struct perf_event_attr attr = {
		.type = PERF_TYPE_SOFTWARE,
		.config = PERF_COUNT_SW_CPU_CLOCK,
		.sample_freq = static_cast<__u64>(env.sample_freq),
		.freq = env.freq,
	};
	int i, fd;

    // Resize links vector to accommodate all CPUs
    links.resize(nr_cpus);

	for (i = 0; i < nr_cpus; i++) {
		if (env.cpu != -1 && env.cpu != i)
			continue;

		fd = syscall(__NR_perf_event_open, &attr, -1, i, -1, 0);
		if (fd < 0) {
			/* Ignore CPU that is offline */
			if (errno == ENODEV)
				continue;

			fprintf(stderr, "failed to init perf sampling: %s\n",
				strerror(errno));
			return -1;
		}

		links[i].reset(bpf_program__attach_perf_event(prog, fd));
		if (!links[i]) {
			fprintf(stderr, "failed to attach perf event on cpu: "
				"%d\n", i);
			close(fd);
			return -1;
		}
	}

	return 0;
} 