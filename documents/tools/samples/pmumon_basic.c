// SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause)
/*
 * pmumon - PMU (Performance Monitoring Unit) monitoring tool
 * Reads various PMU counters and displays comprehensive performance statistics
 * every second.
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
#include <linux/perf_event.h>
#include <asm/unistd.h>
#include <sys/ioctl.h>

#define MAX_CPUS 128
#define MAX_EVENTS 32

static volatile int exiting = 0;

struct pmu_event {
    const char *name;
    __u32 type;
    __u64 config;
    int enabled;
};

struct pmu_counter {
    int fd;
    __u64 prev_value;
    __u64 curr_value;
    __u64 delta;
};

struct cpu_stats {
    struct pmu_counter counters[MAX_EVENTS];
    int active_counters;
};

// PMU events we want to monitor
static struct pmu_event events[] = {
    {"cpu-cycles", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, 1},
    {"instructions", PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS, 1},
    {"cache-references", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_REFERENCES, 1},
    {"cache-misses", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES, 1},
    {"branch-instructions", PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_INSTRUCTIONS, 1},
    {"branch-misses", PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES, 1},
    {"page-faults", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_PAGE_FAULTS, 1},
    {"context-switches", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_CONTEXT_SWITCHES, 1},
    {"cpu-migrations", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_CPU_MIGRATIONS, 1},
    {"minor-faults", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_PAGE_FAULTS_MIN, 1},
    {"major-faults", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_PAGE_FAULTS_MAJ, 1},
    {"task-clock", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_TASK_CLOCK, 1},
    {"cpu-clock", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_CPU_CLOCK, 1},
    {"alignment-faults", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_ALIGNMENT_FAULTS, 1},
    {"emulation-faults", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_EMULATION_FAULTS, 1},
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
    
    printf("Setting up PMU monitoring for %d CPUs, %d events...\n", nr_cpus, nr_events);
    
    for (cpu = 0; cpu < nr_cpus; cpu++) {
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
            
            // Monitor all processes on this CPU
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
            cpu_stats[cpu].active_counters++;
            
            // Enable the counter
            ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);
        }
        
        printf("CPU %d: %d counters active\n", cpu, cpu_stats[cpu].active_counters);
    }
    
    return 0;
}

static void read_counters(void)
{
    int cpu, counter_idx;
    
    for (cpu = 0; cpu < nr_cpus; cpu++) {
        for (counter_idx = 0; counter_idx < cpu_stats[cpu].active_counters; counter_idx++) {
            struct pmu_counter *counter = &cpu_stats[cpu].counters[counter_idx];
            __u64 value;
            
            if (read(counter->fd, &value, sizeof(value)) != sizeof(value)) {
                fprintf(stderr, "Failed to read counter on CPU %d\n", cpu);
                continue;
            }
            
            counter->prev_value = counter->curr_value;
            counter->curr_value = value;
            counter->delta = value - counter->prev_value;
        }
    }
}

static void display_stats(void)
{
    static int first_run = 1;
    int cpu, event_idx;
    __u64 total_values[MAX_EVENTS] = {0};
    __u64 total_deltas[MAX_EVENTS] = {0};
    time_t now;
    
    time(&now);
    
    // Clear screen and print header
    printf("\033[2J\033[H");
    printf("PMU Monitor - %s", ctime(&now));
    printf("================================================================================\n");
    
    if (first_run) {
        printf("Collecting baseline data...\n");
        first_run = 0;
        return;
    }
    
    // Calculate totals across all CPUs
    for (cpu = 0; cpu < nr_cpus; cpu++) {
        int active_event = 0;
        for (event_idx = 0; event_idx < nr_events; event_idx++) {
            if (!events[event_idx].enabled)
                continue;
                
            if (active_event < cpu_stats[cpu].active_counters) {
                total_values[event_idx] += cpu_stats[cpu].counters[active_event].curr_value;
                total_deltas[event_idx] += cpu_stats[cpu].counters[active_event].delta;
                active_event++;
            }
        }
    }
    
    // Display system-wide statistics
    printf("SYSTEM-WIDE STATISTICS (per second):\n");
    printf("%-20s %15s %15s\n", "Event", "Total", "Rate/sec");
    printf("--------------------------------------------------------------------------------\n");
    
    for (event_idx = 0; event_idx < nr_events; event_idx++) {
        if (!events[event_idx].enabled)
            continue;
            
        printf("%-20s %15llu %15llu\n", 
               events[event_idx].name,
               total_values[event_idx],
               total_deltas[event_idx]);
    }
    
    // Calculate and display derived metrics
    printf("\nDERIVED METRICS:\n");
    printf("--------------------------------------------------------------------------------\n");
    
    __u64 cycles = total_deltas[0];  // cpu-cycles
    __u64 instructions = total_deltas[1];  // instructions
    __u64 cache_refs = total_deltas[2];  // cache-references
    __u64 cache_misses = total_deltas[3];  // cache-misses
    __u64 branches = total_deltas[4];  // branch-instructions
    __u64 branch_misses = total_deltas[5];  // branch-misses
    
    if (cycles > 0) {
        double ipc = (double)instructions / cycles;
        printf("Instructions per Cycle (IPC): %.3f\n", ipc);
        printf("CPU Utilization: %.1f%% (approx)\n", 
               (double)cycles / (nr_cpus * 1000000000) * 100);
    }
    
    if (cache_refs > 0) {
        double cache_miss_rate = (double)cache_misses / cache_refs * 100;
        printf("Cache Miss Rate: %.2f%%\n", cache_miss_rate);
    }
    
    if (branches > 0) {
        double branch_miss_rate = (double)branch_misses / branches * 100;
        printf("Branch Miss Rate: %.2f%%\n", branch_miss_rate);
    }
    
    // Display per-CPU breakdown for key metrics
    printf("\nPER-CPU BREAKDOWN (rates per second):\n");
    printf("CPU   Cycles/sec    Instrs/sec   Cache-Miss   Branch-Miss    Page-Faults\n");
    printf("--------------------------------------------------------------------------------\n");
    
    for (cpu = 0; cpu < nr_cpus; cpu++) {
        if (cpu_stats[cpu].active_counters >= 6) {
            printf("%3d %12llu %12llu %12llu %12llu %12llu\n",
                   cpu,
                   cpu_stats[cpu].counters[0].delta,  // cycles
                   cpu_stats[cpu].counters[1].delta,  // instructions
                   cpu_stats[cpu].counters[3].delta,  // cache-misses
                   cpu_stats[cpu].counters[5].delta,  // branch-misses
                   cpu_stats[cpu].counters[6].delta); // page-faults
        }
    }
    
    printf("\n[Press Ctrl+C to exit]\n");
}

static void cleanup(void)
{
    int cpu, counter_idx;
    
    for (cpu = 0; cpu < nr_cpus; cpu++) {
        for (counter_idx = 0; counter_idx < cpu_stats[cpu].active_counters; counter_idx++) {
            close(cpu_stats[cpu].counters[counter_idx].fd);
        }
    }
}

int main(int argc, char **argv)
{
    int interval = 1;  // 1 second default
    
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
    
    printf("PMU Monitor starting - monitoring %d CPUs every %d second(s)\n", 
           nr_cpus, interval);
    printf("Note: Requires root privileges for some PMU counters\n\n");
    
    signal(SIGINT, sig_handler);
    signal(SIGTERM, sig_handler);
    
    if (setup_perf_events() < 0) {
        fprintf(stderr, "Failed to setup perf events\n");
        return 1;
    }
    
    while (!exiting) {
        read_counters();
        display_stats();
        sleep(interval);
    }
    
    cleanup();
    printf("\nPMU monitoring stopped.\n");
    
    return 0;
} 