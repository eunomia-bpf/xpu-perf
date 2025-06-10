#include <iostream>
#include <memory>
#include <signal.h>
#include <unistd.h>
#include <string.h>
#include <cstdlib>
#include <algorithm>

#include "analyzers/analyzer.hpp"
#include "args_parser.hpp"
#include "collectors/utils.hpp"
#include "collectors/oncpu/profile.hpp"
#include "collectors/offcpu/offcputime.hpp"

static volatile bool running = true;

static void sig_handler(int sig)
{
    running = false;
}

int main(int argc, char **argv)
{
    // Parse arguments using the new parser
    ProfilerArgs args = ArgsParser::parse(argc, argv);

    if (args.verbose) {
        std::cout << "Selected analyzer: " << args.analyzer_type << std::endl;
        std::cout << "Duration: " << (args.duration < 99999999 ? std::to_string(args.duration) + " seconds" : "until interrupted") << std::endl;
        
        if (!args.pids.empty()) {
            std::cout << "Target PIDs: ";
            for (size_t i = 0; i < args.pids.size(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << args.pids[i];
            }
            std::cout << std::endl;
        }
        
        if (!args.tids.empty()) {
            std::cout << "Target TIDs: ";
            for (size_t i = 0; i < args.tids.size(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << args.tids[i];
            }
            std::cout << std::endl;
        }
    }

    // Set up signal handler BEFORE creating analyzers
    signal(SIGINT, sig_handler);
    signal(SIGTERM, sig_handler);

    // Create the appropriate analyzer with simplified config
    std::unique_ptr<IAnalyzer> analyzer;
    
    if (args.analyzer_type == "profile") {
        auto config = ArgsParser::create_profile_config(args);
        analyzer = std::make_unique<ProfileAnalyzer>(std::move(config));
    } else if (args.analyzer_type == "offcputime") {
        auto config = ArgsParser::create_offcpu_config(args);
        analyzer = std::make_unique<OffCPUTimeAnalyzer>(std::move(config));
    } else if (args.analyzer_type == "wallclock") {
        auto config = ArgsParser::create_wallclock_config(args);
        analyzer = std::make_unique<WallClockAnalyzer>(std::move(config));
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
    if (args.duration < 99999999) {
        std::cout << " for " << args.duration << " seconds";
    }
    
    if (args.verbose) {
        std::cout << "Collecting data..." << std::endl;
    }

    // Sleep for the specified duration or until interrupted
    int remaining = args.duration;
    while (running && remaining > 0) {
        sleep(1);
        remaining--;
    }

    // Get the flamegraph from the analyzer
    auto flamegraph = analyzer->get_flamegraph();
    if (!flamegraph || !flamegraph->success) {
        std::cerr << "Failed to get flamegraph from " << analyzer->get_name() << std::endl;
        return 1;
    }

    // Check if we have any data before showing detailed output
    bool has_data = !flamegraph->entries.empty();
    
    if (!has_data) {
        std::cout << "\nNo samples collected. This could be due to:" << std::endl;
        std::cout << "- Process not active during profiling period" << std::endl;
        std::cout << "- Insufficient permissions (try running with sudo)" << std::endl;
        std::cout << "- Process ID not found or invalid" << std::endl;
        if (!args.pids.empty()) {
            std::cout << "- Verify PID " << args.pids[0] << " is still running" << std::endl;
        }
    }
    
    if (analyzer->get_name() == "profile_analyzer" || analyzer->get_name() == "offcputime_analyzer") {
        // Only show detailed output if we have data
        if (has_data) {
            if (args.verbose) {
                std::cout << "\nFolded format (for flamegraph.pl):" << std::endl;
                std::cout << flamegraph->to_folded_format() << std::endl;
            }
            
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
            
            if (args.verbose) {
                std::cout << "\nReadable format:" << std::endl;
                std::cout << flamegraph->to_readable_format() << std::endl;
            }
        }
    } else if (analyzer->get_name() == "wallclock_analyzer") {
        auto* wallclock_analyzer = dynamic_cast<WallClockAnalyzer*>(analyzer.get());
        if (wallclock_analyzer) {
            // Get per-thread flamegraphs and print each thread separately
            auto per_thread_flamegraphs = wallclock_analyzer->get_per_thread_flamegraphs();
            
            if (!per_thread_flamegraphs.empty()) {
                std::cout << "Per-thread flamegraphs:" << std::endl;
                for (const auto& [tid, thread_flamegraph] : per_thread_flamegraphs) {
                    if (thread_flamegraph && thread_flamegraph->success) {
                        std::cout << "\n--- Thread " << tid << " ---" << std::endl;
                        std::cout << thread_flamegraph->to_summary() << std::endl;
                        if (args.verbose && !thread_flamegraph->entries.empty()) {
                            std::cout << "Folded format:" << std::endl << thread_flamegraph->to_folded_format() << std::endl;
                        }
                    }
                }
            }
        }
    }
    
    return 0;
} 