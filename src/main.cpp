#include <iostream>
#include <memory>
#include <signal.h>
#include <unistd.h>
#include <string.h>
#include <cstdlib>
#include <algorithm>
#include <sstream>
#include <ctime>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <filesystem>

#include "analyzers/analyzer.hpp"
#include "args_parser.hpp"
#include "collectors/utils.hpp"
#include "collectors/oncpu/profile.hpp"
#include "collectors/offcpu/offcputime.hpp"
#include "flamegraph_generator.hpp"

static volatile bool running = true;

static void sig_handler(int sig)
{
    running = false;
}

std::string create_output_filename(const std::string& analyzer_type, const ProfilerArgs& args) {
    auto now = std::time(nullptr);
    std::stringstream ss;
    
    ss << analyzer_type << "_profile";
    if (!args.pids.empty()) {
        ss << "_pid" << args.pids[0];
    }
    ss << "_" << now;
    
    return ss.str();
}

void generate_flamegraph_for_analyzer(const std::string& analyzer_name, 
                                    std::unique_ptr<FlameGraphView>& flamegraph,
                                    const ProfilerArgs& args,
                                    double actual_runtime_seconds) {
    if (!flamegraph || !flamegraph->success || flamegraph->entries.empty()) {
        std::cout << "No data available for flamegraph generation (analyzer: " << analyzer_name << ")" << std::endl;
        return;
    }
    
    // Create output directory
    std::string output_dir = create_output_filename(analyzer_name, args);
    std::filesystem::create_directories(output_dir);
    
    // Create flamegraph generator
    FlamegraphGenerator fg_gen(output_dir, args.frequency, args.duration);
    fg_gen.set_actual_wall_clock_time(actual_runtime_seconds);
    
    // Convert FlameGraphView entries to FlamegraphEntry format
    std::vector<FlamegraphEntry> entries;
    for (const auto& entry : flamegraph->entries) {
        FlamegraphEntry fg_entry;
        
        // Convert folded stack to stack trace
        std::string stack_trace;
        if (!entry.folded_stack.empty()) {
            stack_trace = entry.folded_stack[0];
            for (size_t i = 1; i < entry.folded_stack.size(); ++i) {
                stack_trace += ";" + entry.folded_stack[i];
            }
        }
        
        // Use command as first part if stack trace is empty
        if (stack_trace.empty()) {
            stack_trace = entry.command;
        }
        
        fg_entry.stack_trace = stack_trace;
        fg_entry.value = entry.sample_count;
        fg_entry.is_oncpu = entry.is_oncpu;
        
        entries.push_back(fg_entry);
    }
    
    if (entries.empty()) {
        std::cout << "No flamegraph entries to process" << std::endl;
        return;
    }
    
    // Generate files
    std::string folded_file = fg_gen.generate_folded_file(entries, analyzer_name + "_profile");
    if (!folded_file.empty()) {
        std::string title = analyzer_name;
        if (analyzer_name == "profile") {
            title = "On-CPU Profile";
        } else if (analyzer_name == "offcputime") {
            title = "Off-CPU Profile";
        }
        
        std::string svg_file = fg_gen.generate_svg_from_folded(folded_file, title);
        fg_gen.generate_analysis_file(analyzer_name + "_profile", entries, "Single-Thread");
        
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "FLAMEGRAPH FILES GENERATED" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        std::cout << "📊 Folded data: " << folded_file << std::endl;
        if (!svg_file.empty()) {
            std::cout << "🔥 Flamegraph:  " << svg_file << std::endl;
            std::cout << "   Open " << svg_file << " in a web browser to view the interactive flamegraph" << std::endl;
        }
        
        if (analyzer_name == "profile") {
            std::cout << "\n📝 Interpretation guide:" << std::endl;
            std::cout << "   • Red frames show CPU-intensive code paths" << std::endl;
            std::cout << "   • Wider sections represent more time spent in those functions" << std::endl;
        } else if (analyzer_name == "offcputime") {
            std::cout << "\n📝 Interpretation guide:" << std::endl;
            std::cout << "   • Blue frames show blocking/waiting operations" << std::endl;
            std::cout << "   • Wider sections represent longer blocking times" << std::endl;
        }
    } else {
        std::cout << "Failed to generate flamegraph files" << std::endl;
    }
}

int main(int argc, char **argv)
{
    // Record start time for runtime calculation
    auto start_time = std::chrono::steady_clock::now();
    
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
    std::cout << " (Press Ctrl+C to stop)" << std::endl;
    
    if (args.verbose) {
        std::cout << "Collecting data..." << std::endl;
    }

    // Sleep for the specified duration or until interrupted
    int remaining = args.duration;
    while (running && remaining > 0) {
        sleep(1);
        remaining--;
    }

    std::cout << "\nStopping analyzer..." << std::endl;

    // Calculate actual runtime
    auto end_time = std::chrono::steady_clock::now();
    auto runtime_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    double actual_runtime_seconds = runtime_duration.count() / 1000.0;

    // Get the flamegraph from the analyzer
    auto flamegraph = analyzer->get_flamegraph();
    if (!flamegraph || !flamegraph->success) {
        std::cerr << "Failed to get flamegraph from " << analyzer->get_name() << std::endl;
        
        // Still show runtime information
        std::cout << "\nRuntime Summary:" << std::endl;
        std::cout << "Actual program runtime: " << std::fixed << std::setprecision(3) << actual_runtime_seconds << "s" << std::endl;
        std::cout << "Requested duration: " << args.duration << "s" << std::endl;
        return 1;
    }

    // Print summary with runtime information
    std::cout << "\nFlameGraph data:" << std::endl;
    std::cout << flamegraph->to_summary() << std::endl;
    
    // Add runtime information to summary
    std::cout << "\nRuntime Summary:" << std::endl;
    std::cout << "Actual program runtime: " << std::fixed << std::setprecision(3) << actual_runtime_seconds << "s" << std::endl;
    std::cout << "Requested duration: " << args.duration << "s" << std::endl;
    std::cout << "Data collection efficiency: " << std::fixed << std::setprecision(1) 
              << (actual_runtime_seconds / args.duration * 100.0) << "%" << std::endl;
    
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
        std::cout << "- Sampling frequency too low (" << args.frequency << " Hz)" << std::endl;
    } else {
        // Generate flamegraph files for single analyzers
        if (analyzer->get_name() == "profile_analyzer") {
            generate_flamegraph_for_analyzer("profile", flamegraph, args, actual_runtime_seconds);
        } else if (analyzer->get_name() == "offcputime_analyzer") {
            generate_flamegraph_for_analyzer("offcputime", flamegraph, args, actual_runtime_seconds);
        } else if (analyzer->get_name() == "wallclock_analyzer") {
            // Wallclock analyzer handles its own file generation
            std::cout << "\nWallclock analyzer completed - files generated in output directory" << std::endl;
        }
        
        if (args.verbose) {
            std::cout << "\nAll stacks:" << std::endl;
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
        }
    }
    
    return 0;
} 