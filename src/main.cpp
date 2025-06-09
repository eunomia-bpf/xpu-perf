#include <iostream>
#include <memory>
#include <signal.h>
#include <unistd.h>
#include <string.h>
#include <cstdlib>

#include "collectors/collector_interface.hpp"
#include "collectors/oncpu/profile.hpp"
#include "collectors/offcpu/offcputime.hpp"

static volatile bool running = true;

static void sig_handler(int sig)
{
    running = false;
}

void print_usage(const char* program_name) {
    printf("Usage: %s <collector_type> [duration_seconds]\n", program_name);
    printf("\nCollector types:\n");
    printf("  profile     - CPU profiling by sampling stack traces\n");
    printf("  offcputime  - Off-CPU time analysis\n");
    printf("\nOptional arguments:\n");
    printf("  duration_seconds - How long to run the collector (default: run until interrupted)\n");
    printf("\nExample:\n");
    printf("  %s profile 10        # Profile for 10 seconds\n", program_name);
    printf("  %s offcputime        # Run until Ctrl+C\n", program_name);
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string collector_type = argv[1];
    
    // Check for help
    if (collector_type == "-h" || collector_type == "--help") {
        print_usage(argv[0]);
        return 0;
    }

    // Parse optional duration
    int duration = 99999999; // Default: very long time
    if (argc >= 3) {
        duration = atoi(argv[2]);
        if (duration <= 0) {
            fprintf(stderr, "Invalid duration: %s\n", argv[2]);
            return 1;
        }
    }

    // Set up signal handler BEFORE creating collectors
    signal(SIGINT, sig_handler);
    signal(SIGTERM, sig_handler);

    // Create the appropriate collector
    std::unique_ptr<ICollector> collector;
    
    if (collector_type == "profile") {
        auto profile_collector = std::make_unique<ProfileCollector>();
        profile_collector->get_config().duration = duration;
        collector = std::move(profile_collector);
    } else if (collector_type == "offcputime") {
        auto offcpu_collector = std::make_unique<OffCPUTimeCollector>();
        offcpu_collector->get_config().duration = duration;
        collector = std::move(offcpu_collector);
    } else {
        fprintf(stderr, "Unknown collector type: %s\n", collector_type.c_str());
        print_usage(argv[0]);
        return 1;
    }

    if (!collector) {
        fprintf(stderr, "Failed to create collector\n");
        return 1;
    }

    // Start the collector
    if (!collector->start()) {
        fprintf(stderr, "Failed to start %s collector\n", collector->get_name().c_str());
        return 1;
    }

    printf("Started %s collector", collector->get_name().c_str());
    if (duration < 99999999) {
        printf(" for %d seconds", duration);
    }
    printf("\n");

    // Sleep for the specified duration or until interrupted
    int remaining = duration;
    while (running && remaining > 0) {
        sleep(1);
        remaining--;
    }

    printf("\nStopping collector...\n");

    // Get the collected data - this will print the results directly
    CollectorData data = collector->get_data();
    if (!data.success) {
        fprintf(stderr, "Failed to collect data from %s\n", collector->get_name().c_str());
        return 1;
    }

    printf("\nCollector %s finished successfully\n", collector->get_name().c_str());
    return 0;
} 