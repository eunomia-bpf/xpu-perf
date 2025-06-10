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
#include <map>

#include "analyzers/analyzer.hpp"
#include "analyzers/wallclock_analyzer.hpp"
#include "args_parser.hpp"
#include "collectors/utils.hpp"
#include "collectors/oncpu/profile.hpp"
#include "collectors/offcpu/offcputime.hpp"
#include "flamegraph_generator.hpp"
#include "server/profile_server.hpp"

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

int main(int argc, char **argv)
{
    // Record start time for runtime calculation
    auto start_time = std::chrono::steady_clock::now();
    
    // Parse arguments using the new parser
    ProfilerArgs args = ArgsParser::parse(argc, argv);

    // Handle server subcommand differently
    if (args.analyzer_type == "server") {
        std::cout << "Starting server mode..." << std::endl;
        
        ProfileServer server("0.0.0.0", 8080);
        
        // Set up signal handler for graceful shutdown
        signal(SIGINT, [](int) {
            std::cout << "\nShutting down server..." << std::endl;
            exit(0);
        });
        signal(SIGTERM, [](int) {
            std::cout << "\nShutting down server..." << std::endl;
            exit(0);
        });
        
        if (!server.start()) {
            std::cerr << "Failed to start server" << std::endl;
            return 1;
        }
        
        return 0;
    }

    if (args.verbose) {
        std::cout << "Analyzer: " << args.analyzer_type << std::endl;
        
        if (!args.pids.empty()) {
            std::cout << "PIDs: ";
            for (size_t i = 0; i < args.pids.size(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << args.pids[i];
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

    std::cout << "Started " << analyzer->get_name();
    if (args.duration < 99999999) {
        std::cout << " for " << args.duration << "s";
    }
    std::cout << " (Press Ctrl+C to stop)" << std::endl;
    
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

    // Get the flamegraph using the generator approach for all analyzers
    std::unique_ptr<FlameGraphView> flamegraph;
    std::string output_dir = create_output_filename(args.analyzer_type, args);
    FlamegraphGenerator fg_gen(output_dir, args.frequency, actual_runtime_seconds);
    
    if (analyzer->get_name() == "wallclock_analyzer") {
        WallClockAnalyzer* wallclock_analyzer = dynamic_cast<WallClockAnalyzer*>(analyzer.get());
        if (!wallclock_analyzer) {
            std::cerr << "Failed to get flamegraph from " << analyzer->get_name() << std::endl;
            return 1;
        }
        flamegraph = fg_gen.get_flamegraph_for_wallclock(*wallclock_analyzer);
        if (!flamegraph) {
            std::cerr << "Failed to get flamegraph from " << analyzer->get_name() << std::endl;
            return 1;
        }
        fg_gen.generate_flamegraph_files_for_wallclock(*wallclock_analyzer);
        std::cout << "Flamegraphs generated in " << output_dir << std::endl;
    } else if (analyzer->get_name() == "profile_analyzer" || analyzer->get_name() == "offcputime_analyzer") {
        ProfileAnalyzer* profile_analyzer = dynamic_cast<ProfileAnalyzer*>(analyzer.get());
        if (!profile_analyzer) {
            std::cerr << "Failed to get flamegraph from " << analyzer->get_name() << std::endl;
            return 1;
        }
        flamegraph = profile_analyzer->get_flamegraph();
        if (!flamegraph) {
            std::cerr << "Failed to get flamegraph from " << analyzer->get_name() << std::endl;
            return 1;
        }
        
        // Create a map structure for the generate_single_flamegraph method
        std::map<pid_t, std::unique_ptr<FlameGraphView>> per_thread_data;
        pid_t tid = args.pids.empty() ? 0 : args.pids[0];
        per_thread_data[tid] = std::move(flamegraph);
        
        fg_gen.generate_single_flamegraph(per_thread_data, args.pids);
    }
    
    // Add runtime information to summary
    std::cout << "\nRuntime Summary:" << std::endl;
    std::cout << "Actual program runtime: " << std::fixed << std::setprecision(3) << actual_runtime_seconds << "s" << std::endl;

    
    return 0;
} 