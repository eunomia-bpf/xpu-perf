#define _GNU_SOURCE
#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <stdlib.h>

// CPU-intensive function
void cpu_work() {
    volatile long sum = 0;
    for (int i = 0; i < 100000000; i++) {
        sum += i * i;
    }
}

// Blocking function
void blocking_work() {
    usleep(100000); // Sleep for 100ms
}

int test_work() {
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        cpu_work();
        clock_gettime(CLOCK_MONOTONIC, &end);
        long cpu_work_time = (end.tv_sec - start.tv_sec) * 1000000000 + (end.tv_nsec - start.tv_nsec);
        // Do some blocking work  
        // printf("blocking... ");
        blocking_work(); 
        return cpu_work_time;
}

int main() {
    printf("Test program started (PID: %d)\n", getpid());
    printf("This program will alternate between CPU work and blocking operations\n");
    printf("Run the combined profiler script with this PID to see both on-CPU and off-CPU activity\n");
    
    while (1) {
        long cpu_work_time = test_work();
        printf("CPU work time: %ld\n", cpu_work_time / 1000 / 1000);
    }
    
    return 0;
} 