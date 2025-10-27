#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main() {
    printf("Starting malloc/free test (PID: %d)\n", getpid());

    for (int i = 0; i < 5; i++) {
        void *ptr = malloc(1024);
        printf("Allocated: %p\n", ptr);
        free(ptr);
        printf("Freed: %p\n", ptr);
        sleep(1);
    }

    return 0;
}
