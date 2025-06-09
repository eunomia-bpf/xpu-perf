#include <iostream>
#include <memory>
#include <signal.h>
#include <unistd.h>
#include <string.h>
#include <cstdlib>

#include "analyzers/analyzer.hpp"

static volatile bool running = true;

static void sig_handler(int sig)
{
    running = false;
}

void print_usage(const char* program_name) {
    printf("Usage: %s <analyzer_type> [duration_seconds]\n", program_name);
    printf("\nAnalyzer types:\n");
    printf("  profile     - CPU profiling by sampling stack traces\n");
    printf("  offcputime  - Off-CPU time analysis\n");
    printf("  wallclock   - Combined on-CPU and off-CPU analysis\n");
    printf("\nOptional arguments:\n");
    printf("  duration_seconds - How long to run the analyzer (default: run until interrupted)\n");
    printf("\nExample:\n");
    printf("  %s profile 10        # Profile for 10 seconds\n", program_name);
    printf("  %s wallclock 30      # Combined analysis for 30 seconds\n", program_name);
    printf("  %s offcputime        # Run until Ctrl+C\n", program_name);
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string analyzer_type = argv[1];
    
    // Check for help
    if (analyzer_type == "-h" || analyzer_type == "--help") {
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

    // Set up signal handler BEFORE creating analyzers
    signal(SIGINT, sig_handler);
    signal(SIGTERM, sig_handler);

    // Create the appropriate analyzer
    std::unique_ptr<IAnalyzer> analyzer;
    
    if (analyzer_type == "profile") {
        auto profile_analyzer = std::make_unique<ProfileAnalyzer>();
        profile_analyzer->get_config().duration = duration;
        analyzer = std::move(profile_analyzer);
    } else if (analyzer_type == "offcputime") {
        auto offcpu_analyzer = std::make_unique<OffCPUTimeAnalyzer>();
        offcpu_analyzer->get_config().duration = duration;
        analyzer = std::move(offcpu_analyzer);
    } else if (analyzer_type == "wallclock") {
        auto wallclock_analyzer = std::make_unique<WallClockAnalyzer>();
        wallclock_analyzer->configure(duration);  // Use default values for other parameters
        analyzer = std::move(wallclock_analyzer);
    } else {
        fprintf(stderr, "Unknown analyzer type: %s\n", analyzer_type.c_str());
        print_usage(argv[0]);
        return 1;
    }

    if (!analyzer) {
        fprintf(stderr, "Failed to create analyzer\n");
        return 1;
    }

    // Start the analyzer
    if (!analyzer->start()) {
        fprintf(stderr, "Failed to start %s analyzer\n", analyzer->get_name().c_str());
        return 1;
    }

    printf("Started %s analyzer", analyzer->get_name().c_str());
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

    printf("\nStopping analyzer...\n");

    // Get the flamegraph from the analyzer
    auto flamegraph = analyzer->get_flamegraph();
    if (!flamegraph || !flamegraph->success) {
        fprintf(stderr, "Failed to get flamegraph from %s\n", analyzer->get_name().c_str());
        return 1;
    }

    // Print summary
    printf("\nFlameGraph data:\n");
    printf("%s\n", flamegraph->to_summary().c_str());
    
    if (analyzer->get_name() == "profile_analyzer" || analyzer->get_name() == "offcputime_analyzer") {
        // For single-threaded analyzers, show folded format and top stacks
        printf("\nFolded format (for flamegraph.pl):\n");
        printf("%s\n", flamegraph->to_folded_format().c_str());
        
        printf("\nTop 10 stacks:\n");
        auto top_stacks = flamegraph->get_top_stacks(10);
        for (size_t i = 0; i < top_stacks.size(); i++) {
            const auto& entry = top_stacks[i];
            printf("%zu. [%.2f%%] %s (%llu %s)\n", i + 1, entry.percentage, 
                   entry.folded_stack.c_str(), entry.sample_count, flamegraph->time_unit.c_str());
        }
    } else if (analyzer->get_name() == "wallclock_analyzer") {
        auto* wallclock_analyzer = dynamic_cast<WallClockAnalyzer*>(analyzer.get());
        if (wallclock_analyzer) {
            // Get per-thread flamegraphs and print each thread separately
            auto per_thread_flamegraphs = wallclock_analyzer->get_per_thread_flamegraphs();
            
            printf("Per-thread flamegraphs:\n");
            for (const auto& [tid, thread_flamegraph] : per_thread_flamegraphs) {
                if (thread_flamegraph && thread_flamegraph->success) {
                    printf("\n--- Thread %d ---\n", tid);
                    printf("%s\n", thread_flamegraph->to_summary().c_str());
                    printf("Folded format:\n%s\n", thread_flamegraph->to_folded_format().c_str());
                }
            }
        }
    }

    printf("\nAnalyzer %s finished successfully\n", analyzer->get_name().c_str());
    return 0;
} 