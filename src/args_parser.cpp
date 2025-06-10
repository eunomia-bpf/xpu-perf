#include "args_parser.hpp"
#include <argparse/argparse.hpp>
#include <iostream>
#include <sstream>
#include <algorithm>

ProfilerArgs ArgsParser::parse(int argc, char** argv) {
    argparse::ArgumentParser program("profiler", "1.0");
    
    setup_parser(program);
    
    try {
        program.parse_args(argc, argv);
    }
    catch (const std::exception& err) {
        std::cerr << "Error: " << err.what() << std::endl;
        std::cerr << program;
        exit(1);
    }
    
    return extract_args(program);
}

void ArgsParser::setup_parser(argparse::ArgumentParser& program) {
    // Configure program description and epilog
    program.add_description("BPF-based profiler for system performance analysis");
    program.add_epilog("Examples:\n"
                      "  profiler profile --duration 10                    # Profile for 10 seconds\n"
                      "  profiler profile --pid 1234                       # Profile specific process\n"
                      "  profiler profile --pid 1234,5678 --tid 9012       # Profile specific processes and thread\n"
                      "  profiler wallclock --duration 30 --pid 1234       # Combined analysis for 30 seconds\n"
                      "  profiler offcputime --tid 5678                     # Run until Ctrl+C for specific thread\n"
                      "  profiler server                                    # Start HTTP server mode");
    
    // Add subcommands for different analyzer types
    program.add_argument("analyzer")
        .help("Type of analysis to perform")
        .choices("profile", "offcputime", "wallclock", "server")
        .metavar("ANALYZER");
    
    // Duration argument
    program.add_argument("-d", "--duration")
        .help("Duration to run the analyzer in seconds (default: run until interrupted)")
        .scan<'i', int>()
        .default_value(99999999)
        .metavar("SECONDS");
    
    // PID arguments
    program.add_argument("-p", "--pid")
        .help("Process ID(s) to profile (comma-separated list, e.g., 1234,5678)")
        .default_value("")
        .metavar("PID[,PID...]");
    
    // Thread ID arguments
    program.add_argument("-t", "--tid")
        .help("Thread ID(s) to profile (comma-separated list, e.g., 1234,5678)")
        .default_value("")
        .metavar("TID[,TID...]");
    
    // Sampling frequency
    program.add_argument("-f", "--frequency")
        .help("Sampling frequency in Hz (default: 49)")
        .scan<'i', int>()
        .default_value(49)
        .metavar("HZ");
    
    // Min block time for off-CPU analysis
    program.add_argument("-m", "--min-block")
        .help("Minimum blocking time in microseconds for off-CPU analysis (default: 1000)")
        .scan<'u', __u64>()
        .default_value(1000ULL)
        .metavar("MICROSECONDS");
    
    // CPU filter
    program.add_argument("-c", "--cpu")
        .help("CPU to profile (default: all CPUs)")
        .scan<'i', int>()
        .default_value(-1)
        .metavar("CPU");
    
    // Stack filtering options
    program.add_argument("-U", "--user-stacks-only")
        .help("Show only user-space stacks")
        .flag();
    
    program.add_argument("-K", "--kernel-stacks-only")
        .help("Show only kernel-space stacks")
        .flag();
    
    // Thread filtering options
    program.add_argument("--user-threads-only")
        .help("Profile only user threads")
        .flag();
    
    program.add_argument("--kernel-threads-only")
        .help("Profile only kernel threads")
        .flag();
    
    // Verbose output
    program.add_argument("-v", "--verbose")
        .help("Enable verbose output")
        .flag();
    
    // Version information
    program.add_argument("--version")
        .help("Show version information")
        .flag();
}

ProfilerArgs ArgsParser::extract_args(const argparse::ArgumentParser& program) {
    ProfilerArgs args;
    
    // Handle version flag early
    if (program.get<bool>("--version")) {
        std::cout << "BPF Profiler v1.0" << std::endl;
        std::cout << "Built with modern C++20 and eBPF technology" << std::endl;
        exit(0);
    }
    
    // Extract basic arguments
    args.analyzer_type = program.get<std::string>("analyzer");
    args.duration = program.get<int>("--duration");
    args.verbose = program.get<bool>("--verbose");
    args.frequency = program.get<int>("--frequency");
    args.min_block_us = program.get<__u64>("--min-block");
    args.cpu = program.get<int>("--cpu");
    
    // Extract boolean flags
    args.user_stacks_only = program.get<bool>("--user-stacks-only");
    args.kernel_stacks_only = program.get<bool>("--kernel-stacks-only");
    args.user_threads_only = program.get<bool>("--user-threads-only");
    args.kernel_threads_only = program.get<bool>("--kernel-threads-only");
    
    // Parse PID list
    std::string pid_str = program.get<std::string>("--pid");
    if (!pid_str.empty()) {
        args.pids = parse_pid_list(pid_str);
    }
    
    // Parse TID list
    std::string tid_str = program.get<std::string>("--tid");
    if (!tid_str.empty()) {
        args.tids = parse_pid_list(tid_str);
    }
    
    // Validation
    if (args.user_stacks_only && args.kernel_stacks_only) {
        std::cerr << "Error: Cannot specify both --user-stacks-only and --kernel-stacks-only" << std::endl;
        exit(1);
    }
    
    if (args.user_threads_only && args.kernel_threads_only) {
        std::cerr << "Error: Cannot specify both --user-threads-only and --kernel-threads-only" << std::endl;
        exit(1);
    }
    
    if (args.frequency <= 0) {
        std::cerr << "Error: Frequency must be positive" << std::endl;
        exit(1);
    }
    
    if (args.duration <= 0) {
        std::cerr << "Error: Duration must be positive" << std::endl;
        exit(1);
    }
    
    return args;
}

std::vector<pid_t> ArgsParser::parse_pid_list(const std::string& pid_str) {
    std::vector<pid_t> pids;
    std::stringstream ss(pid_str);
    std::string pid_token;
    
    while (std::getline(ss, pid_token, ',')) {
        // Trim whitespace
        pid_token.erase(0, pid_token.find_first_not_of(" \t"));
        pid_token.erase(pid_token.find_last_not_of(" \t") + 1);
        
        if (!pid_token.empty()) {
            try {
                pid_t pid = static_cast<pid_t>(std::stoi(pid_token));
                if (pid > 0) {
                    pids.push_back(pid);
                } else {
                    std::cerr << "Error: Invalid PID/TID: " << pid_token << std::endl;
                    exit(1);
                }
            } catch (const std::exception& e) {
                std::cerr << "Error: Invalid PID/TID format: " << pid_token << std::endl;
                exit(1);
            }
        }
    }
    
    return pids;
}

template<typename ConfigType>
void ArgsParser::apply_common_config(ConfigType& config, const ProfilerArgs& args) {
    config.duration = args.duration;
    config.pids = args.pids;
    config.tids = args.tids;
}

std::unique_ptr<ProfileAnalyzerConfig> ArgsParser::create_profile_config(const ProfilerArgs& args) {
    auto config = std::make_unique<ProfileAnalyzerConfig>();
    apply_common_config(*config, args);
    
    // Profile-specific settings
    config->frequency = args.frequency;
    
    return config;
}

std::unique_ptr<OffCPUAnalyzerConfig> ArgsParser::create_offcpu_config(const ProfilerArgs& args) {
    auto config = std::make_unique<OffCPUAnalyzerConfig>();
    apply_common_config(*config, args);
    
    // OffCPU-specific settings
    config->min_block_us = args.min_block_us;
    
    return config;
}

std::unique_ptr<WallClockAnalyzerConfig> ArgsParser::create_wallclock_config(const ProfilerArgs& args) {
    auto config = std::make_unique<WallClockAnalyzerConfig>();
    apply_common_config(*config, args);
    
    // WallClock-specific settings
    config->frequency = args.frequency;
    config->min_block_us = args.min_block_us;
    
    return config;
} 