#include <iostream>
#include <memory>
#include <signal.h>
#include <unistd.h>
#include <string.h>
#include <cstdlib>
#include <argparse/argparse.hpp>

#include "analyzers/analyzer.hpp"

static volatile bool running = true;

static void sig_handler(int sig)
{
    running = false;
}

int main(int argc, char **argv)
{
    argparse::ArgumentParser program("profiler", "1.0");
    
    // Configure program description and epilog
    program.add_description("BPF-based profiler for system performance analysis");
    program.add_epilog("Examples:\n"
                      "  profiler profile --duration 10     # Profile for 10 seconds\n"
                      "  profiler wallclock --duration 30   # Combined analysis for 30 seconds\n"
                      "  profiler offcputime                 # Run until Ctrl+C");
    
    // Add subcommands for different analyzer types
    program.add_argument("analyzer")
        .help("Type of analysis to perform")
        .choices("profile", "offcputime", "wallclock")
        .metavar("ANALYZER");
    
    program.add_argument("-d", "--duration")
        .help("Duration to run the analyzer in seconds (default: run until interrupted)")
        .scan<'i', int>()
        .default_value(99999999)
        .metavar("SECONDS");
    
    program.add_argument("-v", "--verbose")
        .help("Enable verbose output")
        .flag();
    
    program.add_argument("--version")
        .help("Show version information")
        .flag();

    try {
        program.parse_args(argc, argv);
    }
    catch (const std::exception& err) {
        std::cerr << "Error: " << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    // Handle version flag
    if (program.get<bool>("--version")) {
        std::cout << "BPF Profiler v1.0" << std::endl;
        std::cout << "Built with modern C++20 and eBPF technology" << std::endl;
        return 0;
    }

    // Get parsed arguments
    std::string analyzer_type = program.get<std::string>("analyzer");
    int duration = program.get<int>("--duration");
    bool verbose = program.get<bool>("--verbose");

    if (verbose) {
        std::cout << "Selected analyzer: " << analyzer_type << std::endl;
        std::cout << "Duration: " << (duration < 99999999 ? std::to_string(duration) + " seconds" : "until interrupted") << std::endl;
    }

    // Set up signal handler BEFORE creating analyzers
    signal(SIGINT, sig_handler);
    signal(SIGTERM, sig_handler);

    // Create the appropriate analyzer
    std::unique_ptr<IAnalyzer> analyzer;
    
    if (analyzer_type == "profile") {
        analyzer = std::make_unique<ProfileAnalyzer>();
        auto* profile_analyzer = dynamic_cast<ProfileAnalyzer*>(analyzer.get());
        if (profile_analyzer) {
            profile_analyzer->get_config().duration = duration;
        }
    } else if (analyzer_type == "offcputime") {
        analyzer = std::make_unique<OffCPUTimeAnalyzer>();
        auto* offcpu_analyzer = dynamic_cast<OffCPUTimeAnalyzer*>(analyzer.get());
        if (offcpu_analyzer) {
            offcpu_analyzer->get_config().duration = duration;
        }
    } else if (analyzer_type == "wallclock") {
        analyzer = std::make_unique<WallClockAnalyzer>();
        auto* wallclock_analyzer = dynamic_cast<WallClockAnalyzer*>(analyzer.get());
        if (wallclock_analyzer) {
            wallclock_analyzer->configure(duration);  // Use default values for other parameters
        }
    }

    if (!analyzer) {
        std::cerr << "Failed to create analyzer" << std::endl;
        return 1;
    }

    // Start the analyzer
    if (!analyzer->start()) {
        std::cerr << "Failed to start " << analyzer->get_name() << " analyzer" << std::endl;
        return 1;
    }

    std::cout << "Started " << analyzer->get_name() << " analyzer";
    if (duration < 99999999) {
        std::cout << " for " << duration << " seconds";
    }
    std::cout << " (Press Ctrl+C to stop)" << std::endl;
    
    if (verbose) {
        std::cout << "Collecting data..." << std::endl;
    }

    // Sleep for the specified duration or until interrupted
    int remaining = duration;
    while (running && remaining > 0) {
        sleep(1);
        remaining--;
    }

    std::cout << "\nStopping analyzer..." << std::endl;

    // Get the flamegraph from the analyzer
    auto flamegraph = analyzer->get_flamegraph();
    if (!flamegraph || !flamegraph->success) {
        std::cerr << "Failed to get flamegraph from " << analyzer->get_name() << std::endl;
        return 1;
    }

    // Print summary
    std::cout << "\nFlameGraph data:" << std::endl;
    std::cout << flamegraph->to_summary() << std::endl;
    
    if (analyzer->get_name() == "profile_analyzer" || analyzer->get_name() == "offcputime_analyzer") {
        // For single-threaded analyzers, show folded format and top stacks
        if (verbose) {
            std::cout << "\nFolded format (for flamegraph.pl):" << std::endl;
            std::cout << flamegraph->to_folded_format() << std::endl;
        }
        
        std::cout << "\nTop 10 stacks:" << std::endl;
        auto top_stacks = flamegraph->entries;
        for (size_t i = 0; i < top_stacks.size(); i++) {
            const auto& entry = top_stacks[i];
            // Convert vector to folded string format
            std::string folded_str;
            if (!entry.folded_stack.empty()) {
                folded_str = entry.folded_stack[0];
                for (size_t j = 1; j < entry.folded_stack.size(); ++j) {
                    folded_str += ";" + entry.folded_stack[j];
                }
            }
            std::cout << folded_str << " " << entry.sample_count << std::endl;
        }
        
        if (verbose) {
            std::cout << "\nReadable format:" << std::endl;
            std::cout << flamegraph->to_readable_format() << std::endl;
        }
    } else if (analyzer->get_name() == "wallclock_analyzer") {
        auto* wallclock_analyzer = dynamic_cast<WallClockAnalyzer*>(analyzer.get());
        if (wallclock_analyzer) {
            // Get per-thread flamegraphs and print each thread separately
            auto per_thread_flamegraphs = wallclock_analyzer->get_per_thread_flamegraphs();
            
            std::cout << "Per-thread flamegraphs:" << std::endl;
            for (const auto& [tid, thread_flamegraph] : per_thread_flamegraphs) {
                if (thread_flamegraph && thread_flamegraph->success) {
                    std::cout << "\n--- Thread " << tid << " ---" << std::endl;
                    std::cout << thread_flamegraph->to_summary() << std::endl;
                    if (verbose) {
                        std::cout << "Folded format:" << std::endl << thread_flamegraph->to_folded_format() << std::endl;
                    }
                }
            }
        }
    }
    
    return 0;
} 