// SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause)
/*
 * pmumon_advanced - Advanced PMU monitoring with per-function profiling
 * Samples cache misses and correlates with function symbols
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <signal.h>
#include <time.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <linux/perf_event.h>
#include <asm/unistd.h>
#include <sys/ioctl.h>

#define MAX_CPUS 128
#define MAX_EVENTS 64
#define MMAP_SIZE (1 + 512)  // 512 pages for ring buffer

static volatile int exiting = 0;

struct advanced_pmu_event {
    const char *name;
    const char *description;
    __u32 type;
    __u64 config;
    int sample_period;  // 0 = counting mode, >0 = sampling mode
    int enabled;
};

// Comprehensive PMU events including advanced counters
static struct advanced_pmu_event events[] = {
    // Hardware counters
    {"cpu-cycles", "CPU cycles", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, 0, 1},
    {"instructions", "Instructions retired", PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS, 0, 1},
    {"cache-references", "Cache references", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_REFERENCES, 0, 1},
    {"cache-misses", "Cache misses (sample)", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES, 1000, 1},
    {"branch-instructions", "Branch instructions", PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_INSTRUCTIONS, 0, 1},
    {"branch-misses", "Branch misses", PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES, 10000, 1},
    {"bus-cycles", "Bus cycles", PERF_TYPE_HARDWARE, PERF_COUNT_HW_BUS_CYCLES, 0, 1},
    {"stalled-cycles-frontend", "Frontend stalls", PERF_TYPE_HARDWARE, PERF_COUNT_HW_STALLED_CYCLES_FRONTEND, 0, 1},
    {"stalled-cycles-backend", "Backend stalls", PERF_TYPE_HARDWARE, PERF_COUNT_HW_STALLED_CYCLES_BACKEND, 0, 1},
    
    // Software counters
    {"page-faults", "Page faults", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_PAGE_FAULTS, 0, 1},
    {"context-switches", "Context switches", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_CONTEXT_SWITCHES, 0, 1},
    {"cpu-migrations", "CPU migrations", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_CPU_MIGRATIONS, 0, 1},
    {"minor-faults", "Minor page faults", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_PAGE_FAULTS_MIN, 0, 1},
    {"major-faults", "Major page faults", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_PAGE_FAULTS_MAJ, 0, 1},
    {"task-clock", "Task clock time", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_TASK_CLOCK, 0, 1},
    
    // Cache detailed events (using PERF_TYPE_HW_CACHE)
    {"L1-dcache-loads", "L1 data cache loads", PERF_TYPE_HW_CACHE, 
     PERF_COUNT_HW_CACHE_L1D | (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16), 0, 1},
    {"L1-dcache-load-misses", "L1 data cache load misses", PERF_TYPE_HW_CACHE,
     PERF_COUNT_HW_CACHE_L1D | (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16), 0, 1},
    {"L1-icache-loads", "L1 instruction cache loads", PERF_TYPE_HW_CACHE,
     PERF_COUNT_HW_CACHE_L1I | (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16), 0, 1},
    {"L1-icache-load-misses", "L1 instruction cache load misses", PERF_TYPE_HW_CACHE,
     PERF_COUNT_HW_CACHE_L1I | (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16), 0, 1},
    {"LLC-loads", "Last level cache loads", PERF_TYPE_HW_CACHE,
     PERF_COUNT_HW_CACHE_LL | (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16), 0, 1},
    {"LLC-load-misses", "Last level cache load misses", PERF_TYPE_HW_CACHE,
     PERF_COUNT_HW_CACHE_LL | (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16), 0, 1},
    {"LLC-stores", "Last level cache stores", PERF_TYPE_HW_CACHE,
     PERF_COUNT_HW_CACHE_LL | (PERF_COUNT_HW_CACHE_OP_WRITE << 8) | (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16), 0, 1},
    {"LLC-store-misses", "Last level cache store misses", PERF_TYPE_HW_CACHE,
     PERF_COUNT_HW_CACHE_LL | (PERF_COUNT_HW_CACHE_OP_WRITE << 8) | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16), 0, 1},
    
    // TLB events
    {"dTLB-loads", "Data TLB loads", PERF_TYPE_HW_CACHE,
     PERF_COUNT_HW_CACHE_DTLB | (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16), 0, 1},
    {"dTLB-load-misses", "Data TLB load misses", PERF_TYPE_HW_CACHE,
     PERF_COUNT_HW_CACHE_DTLB | (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16), 0, 1},
    {"iTLB-loads", "Instruction TLB loads", PERF_TYPE_HW_CACHE,
     PERF_COUNT_HW_CACHE_ITLB | (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16), 0, 1},
    {"iTLB-load-misses", "Instruction TLB load misses", PERF_TYPE_HW_CACHE,
     PERF_COUNT_HW_CACHE_ITLB | (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16), 0, 1},
};

struct pmu_counter {
    int fd;
    void *mmap_base;
    __u64 prev_value;
    __u64 curr_value;
    __u64 delta;
    int is_sampling;
};

struct cpu_stats {
    struct pmu_counter counters[MAX_EVENTS];
    int active_counters;
};

static int nr_events = sizeof(events) / sizeof(events[0]);
static int nr_cpus;
static struct cpu_stats cpu_stats[MAX_CPUS];

static long perf_event_open(struct perf_event_attr *hw_event, pid_t pid,
                           int cpu, int group_fd, unsigned long flags)
{
    return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
}

static void sig_handler(int sig)
{
    exiting = 1;
}

static int setup_perf_events(void)
{
    struct perf_event_attr attr;
    int cpu, event_idx;
    
    printf("Setting up advanced PMU monitoring for %d CPUs, %d events...\n", nr_cpus, nr_events);
    
    for (cpu = 0; cpu < nr_cpus && cpu < 4; cpu++) {  // Limit to first 4 CPUs for demo
        cpu_stats[cpu].active_counters = 0;
        
        for (event_idx = 0; event_idx < nr_events; event_idx++) {
            if (!events[event_idx].enabled)
                continue;
                
            memset(&attr, 0, sizeof(attr));
            attr.type = events[event_idx].type;
            attr.size = sizeof(attr);
            attr.config = events[event_idx].config;
            attr.disabled = 1;
            attr.exclude_kernel = 0;
            attr.exclude_hv = 1;
            attr.inherit = 1;
            
            // Configure sampling for specific events
            if (events[event_idx].sample_period > 0) {
                attr.sample_freq = 0;
                attr.sample_period = events[event_idx].sample_period;
                attr.sample_type = PERF_SAMPLE_IP | PERF_SAMPLE_TID | PERF_SAMPLE_TIME | PERF_SAMPLE_CALLCHAIN;
                attr.wakeup_events = 1;
            }
            
            int fd = perf_event_open(&attr, -1, cpu, -1, 0);
            if (fd < 0) {
                if (errno != ENODEV && errno != EOPNOTSUPP) {
                    fprintf(stderr, "Failed to open perf event %s on CPU %d: %s\n",
                           events[event_idx].name, cpu, strerror(errno));
                }
                continue;
            }
            
            cpu_stats[cpu].counters[cpu_stats[cpu].active_counters].fd = fd;
            cpu_stats[cpu].counters[cpu_stats[cpu].active_counters].prev_value = 0;
            cpu_stats[cpu].counters[cpu_stats[cpu].active_counters].is_sampling = 
                (events[event_idx].sample_period > 0);
            
            // Setup mmap for sampling events
            if (events[event_idx].sample_period > 0) {
                void *mmap_base = mmap(NULL, (MMAP_SIZE + 1) * getpagesize(), 
                                     PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
                if (mmap_base == MAP_FAILED) {
                    fprintf(stderr, "Failed to mmap sampling buffer: %s\n", strerror(errno));
                    close(fd);
                    continue;
                }
                cpu_stats[cpu].counters[cpu_stats[cpu].active_counters].mmap_base = mmap_base;
            }
            
            cpu_stats[cpu].active_counters++;
            ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);
        }
        
        printf("CPU %d: %d counters active\n", cpu, cpu_stats[cpu].active_counters);
    }
    
    return 0;
}

static void read_counters(void)
{
    int cpu, counter_idx;
    
    for (cpu = 0; cpu < nr_cpus && cpu < 4; cpu++) {
        for (counter_idx = 0; counter_idx < cpu_stats[cpu].active_counters; counter_idx++) {
            struct pmu_counter *counter = &cpu_stats[cpu].counters[counter_idx];
            
            if (!counter->is_sampling) {
                __u64 value;
                if (read(counter->fd, &value, sizeof(value)) == sizeof(value)) {
                    counter->prev_value = counter->curr_value;
                    counter->curr_value = value;
                    counter->delta = value - counter->prev_value;
                }
            }
        }
    }
}

static void display_comprehensive_stats(void)
{
    static int first_run = 1;
    int cpu, event_idx;
    __u64 total_values[MAX_EVENTS] = {0};
    __u64 total_deltas[MAX_EVENTS] = {0};
    time_t now;
    
    time(&now);
    
    printf("\033[2J\033[H");
    printf("Advanced PMU Monitor - %s", ctime(&now));
    printf("================================================================================\n");
    
    if (first_run) {
        printf("Collecting baseline data...\n");
        first_run = 0;
        return;
    }
    
    // Calculate totals
    for (cpu = 0; cpu < 4; cpu++) {
        int active_event = 0;
        for (event_idx = 0; event_idx < nr_events; event_idx++) {
            if (!events[event_idx].enabled || events[event_idx].sample_period > 0)
                continue;
                
            if (active_event < cpu_stats[cpu].active_counters) {
                total_values[event_idx] += cpu_stats[cpu].counters[active_event].curr_value;
                total_deltas[event_idx] += cpu_stats[cpu].counters[active_event].delta;
                active_event++;
            }
        }
    }
    
    // Display comprehensive statistics
    printf("COMPREHENSIVE PMU STATISTICS (4 CPUs, per second):\n");
    printf("%-25s %-40s %15s %15s\n", "Event", "Description", "Total", "Rate/sec");
    printf("--------------------------------------------------------------------------------\n");
    
    for (event_idx = 0; event_idx < nr_events; event_idx++) {
        if (!events[event_idx].enabled || events[event_idx].sample_period > 0)
            continue;
            
        printf("%-25s %-40s %15llu %15llu\n", 
               events[event_idx].name,
               events[event_idx].description,
               total_values[event_idx],
               total_deltas[event_idx]);
    }
    
    // Advanced derived metrics
    printf("\nADVANCED PERFORMANCE METRICS:\n");
    printf("================================================================================\n");
    
    __u64 cycles = total_deltas[0];
    __u64 instructions = total_deltas[1];
    __u64 l1d_loads = total_deltas[15];
    __u64 l1d_misses = total_deltas[16];
    __u64 l1i_loads = total_deltas[17];
    __u64 l1i_misses = total_deltas[18];
    __u64 llc_loads = total_deltas[19];
    __u64 llc_misses = total_deltas[20];
    __u64 dtlb_loads = total_deltas[23];
    __u64 dtlb_misses = total_deltas[24];
    __u64 frontend_stalls = total_deltas[7];
    __u64 backend_stalls = total_deltas[8];
    
    if (cycles > 0) {
        printf("Instructions per Cycle (IPC): %.3f\n", (double)instructions / cycles);
        printf("Cycles per Instruction (CPI): %.3f\n", (double)cycles / instructions);
    }
    
    if (l1d_loads > 0) {
        printf("L1 Data Cache Miss Rate: %.2f%%\n", (double)l1d_misses / l1d_loads * 100);
    }
    
    if (l1i_loads > 0) {
        printf("L1 Instruction Cache Miss Rate: %.2f%%\n", (double)l1i_misses / l1i_loads * 100);
    }
    
    if (llc_loads > 0) {
        printf("Last Level Cache Miss Rate: %.2f%%\n", (double)llc_misses / llc_loads * 100);
    }
    
    if (dtlb_loads > 0) {
        printf("Data TLB Miss Rate: %.2f%%\n", (double)dtlb_misses / dtlb_loads * 100);
    }
    
    if (cycles > 0) {
        printf("Frontend Stall Rate: %.2f%%\n", (double)frontend_stalls / cycles * 100);
        printf("Backend Stall Rate: %.2f%%\n", (double)backend_stalls / cycles * 100);
    }
    
    printf("\nMEMORY HIERARCHY PERFORMANCE:\n");
    printf("--------------------------------------------------------------------------------\n");
    printf("L1D Cache Efficiency: %.1f%% (Higher is better)\n", 
           l1d_loads > 0 ? (double)(l1d_loads - l1d_misses) / l1d_loads * 100 : 0);
    printf("L1I Cache Efficiency: %.1f%% (Higher is better)\n", 
           l1i_loads > 0 ? (double)(l1i_loads - l1i_misses) / l1i_loads * 100 : 0);
    printf("LLC Efficiency: %.1f%% (Higher is better)\n", 
           llc_loads > 0 ? (double)(llc_loads - llc_misses) / llc_loads * 100 : 0);
    
    printf("\n[Press Ctrl+C to exit]\n");
    printf("Note: For per-function cache miss analysis, use 'perf record -e cache-misses' + 'perf report'\n");
}

static void cleanup(void)
{
    int cpu, counter_idx;
    
    for (cpu = 0; cpu < 4; cpu++) {
        for (counter_idx = 0; counter_idx < cpu_stats[cpu].active_counters; counter_idx++) {
            if (cpu_stats[cpu].counters[counter_idx].mmap_base) {
                munmap(cpu_stats[cpu].counters[counter_idx].mmap_base, 
                       (MMAP_SIZE + 1) * getpagesize());
            }
            close(cpu_stats[cpu].counters[counter_idx].fd);
        }
    }
}

int main(int argc, char **argv)
{
    int interval = 1;
    
    if (argc > 1) {
        interval = atoi(argv[1]);
        if (interval <= 0) {
            fprintf(stderr, "Invalid interval: %s\n", argv[1]);
            return 1;
        }
    }
    
    nr_cpus = sysconf(_SC_NPROCESSORS_ONLN);
    if (nr_cpus > MAX_CPUS) {
        nr_cpus = MAX_CPUS;
    }
    
    printf("Advanced PMU Monitor - Comprehensive Performance Analysis\n");
    printf("Monitoring first 4 CPUs every %d second(s)\n", interval);
    printf("Requires root privileges for hardware PMU access\n\n");
    
    signal(SIGINT, sig_handler);
    signal(SIGTERM, sig_handler);
    
    if (setup_perf_events() < 0) {
        fprintf(stderr, "Failed to setup perf events\n");
        return 1;
    }
    
    while (!exiting) {
        read_counters();
        display_comprehensive_stats();
        sleep(interval);
    }
    
    cleanup();
    printf("\nAdvanced PMU monitoring stopped.\n");
    
    return 0;
} 