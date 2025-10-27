#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main() {
    printf("Starting malloc test (PID: %d)\n", getpid());
    printf("Press Ctrl+C to stop\n");

    while (1) {
        // Allocate and free memory repeatedly
        void *ptr = malloc(1024);
        if (ptr) {
            free(ptr);
        }
        sleep(1);
    }

    return 0;
}
