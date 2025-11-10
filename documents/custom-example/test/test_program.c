// Test program to demonstrate custom context propagation
// Compile: gcc -o test_program test_program.c -g
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>

// This function will be used as uprobe attachment point
__attribute__((noinline))
void report_trace(uint64_t context_id) {
    // This function triggers trace collection with the context_id
    // The eBPF uprobe will:
    // 1. Read context_id from RDI register (first parameter)
    // 2. Store it in custom_context_map
    // 3. Tail call to custom__generic for trace collection
    asm volatile("" ::: "memory"); // Prevent optimization
}

__attribute__((noinline))
void do_work(int iterations) {
    volatile int sum = 0;
    for (int i = 0; i < iterations; i++) {
        sum += i;
    }
}

int main() {
    printf("Test program for custom context propagation\n");
    printf("PID: %d\n", getpid());
    printf("Attach uprobe to report_trace function\n\n");

    for (int i = 0; i < 5; i++) {
        printf("Iteration %d:\n", i);

        // Set a custom context ID (e.g., request ID, transaction ID, etc.)
        uint64_t context_id = 0x1000 + i;
        printf("  Calling report_trace with context_id = 0x%lx (%lu)\n", context_id, context_id);

        // Do some work
        do_work(context_id);

        // Report trace with the context
        report_trace(context_id);

        sleep(2);
    }

    printf("\nTest complete\n");
    return 0;
}
