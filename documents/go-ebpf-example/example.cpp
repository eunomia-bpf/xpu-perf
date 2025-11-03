//go:build ignore

#include <stdio.h>
#include <unistd.h>

// Define STAP_SDT_V2 to enable semaphores for USDT probes
#define STAP_SDT_V2
#include <sys/sdt.h>

// Simple function to add two numbers
int add_numbers(int a, int b) {
    int result = a + b;

    // USDT probe with semaphore support
    DTRACE_PROBE2(myapp, add_operation, a, b);

    return result;
}

int main() {
    int counter = 0;

    printf("Starting example program with USDT probes...\n");
    printf("PID: %d\n", getpid());
    printf("Running in a loop, press Ctrl+C to exit\n\n");

    while (1) {
        counter++;

        // Call our function that has a USDT probe
        int result = add_numbers(counter, counter * 2);

        printf("Iteration %d: add_numbers(%d, %d) = %d\n",
               counter, counter, counter * 2, result);

        sleep(1);
    }

    return 0;
}
