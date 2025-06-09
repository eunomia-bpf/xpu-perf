#include <iostream>
#include <memory>
#include <signal.h>
#include <unistd.h>
#include <string.h>

#include "collector_interface.hpp"
#include "profile.hpp"
#include "offcputime.hpp"
#include "arg_parse.h"

static volatile bool running = true;

static void sig_handler(int sig)
{
    running = false;
}

void print_usage(const char* program_name) {
    printf("Usage: %s [OPTIONS] <collector_type>\n", program_name);
    printf("\nCollector types:\n");
    printf("  profile     - CPU profiling by sampling stack traces\n");
    printf("  offcputime  - Off-CPU time analysis\n");
    printf("\nOptions:\n");
    printf("  -h, --help  Show this help message\n");
    printf("\nFor collector-specific options, use:\n");
    printf("  %s <collector_type> --help\n", program_name);
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

    // Create new argv without the collector type for arg parsing
    char **new_argv = (char**)malloc(argc * sizeof(char*));
    new_argv[0] = argv[0];
    for (int i = 2; i < argc; i++) {
        new_argv[i-1] = argv[i];
    }
    int new_argc = argc - 1;

    // Parse arguments based on collector type
    int err;
    if (collector_type == "profile") {
        err = parse_common_args(new_argc, new_argv, TOOL_PROFILE);
    } else if (collector_type == "offcputime") {
        err = parse_common_args(new_argc, new_argv, TOOL_OFFCPUTIME);
    } else {
        fprintf(stderr, "Unknown collector type: %s\n", collector_type.c_str());
        print_usage(argv[0]);
        free(new_argv);
        return 1;
    }

    free(new_argv);

    if (err) {
        return err;
    }

    err = validate_common_args();
    if (err) {
        return err;
    }

    // Set up signal handler BEFORE creating collectors
    signal(SIGINT, sig_handler);
    signal(SIGTERM, sig_handler);

    // Create the appropriate collector
    std::unique_ptr<ICollector> collector;
    
    if (collector_type == "profile") {
        collector.reset(new ProfileCollector());
    } else if (collector_type == "offcputime") {
        collector.reset(new OffCPUTimeCollector());
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

    printf("Started %s collector\n", collector->get_name().c_str());

    // Sleep for the specified duration or until interrupted
    int remaining = env.duration;
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