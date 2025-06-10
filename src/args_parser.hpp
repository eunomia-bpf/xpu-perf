#ifndef __ARGS_PARSER_HPP
#define __ARGS_PARSER_HPP

#include <string>
#include <vector>
#include <sys/types.h>
#include <stdint.h>
#include <memory>
#include "analyzers/analyzer_config.hpp"

// Forward declare Linux types to avoid pulling in kernel headers
typedef unsigned long long __u64;

// Forward declarations to avoid including argparse in header
namespace argparse {
    class ArgumentParser;
}

struct ProfilerArgs {
    std::string analyzer_type;
    int duration;
    bool verbose;
    bool version;
    std::vector<pid_t> pids;
    std::vector<pid_t> tids;
    int frequency;
    __u64 min_block_us;
    int cpu;
    bool user_stacks_only;
    bool kernel_stacks_only;
    bool user_threads_only;
    bool kernel_threads_only;
    
    ProfilerArgs() : 
        duration(99999999),
        verbose(false),
        version(false),
        frequency(49),
        min_block_us(1000),
        cpu(-1),
        user_stacks_only(false),
        kernel_stacks_only(false),
        user_threads_only(false),
        kernel_threads_only(false) {}
};

class ArgsParser {
public:
    static ProfilerArgs parse(int argc, char** argv);
    
    // Create analyzer-specific configurations from parsed arguments
    static std::unique_ptr<ProfileAnalyzerConfig> create_profile_config(const ProfilerArgs& args);
    static std::unique_ptr<OffCPUAnalyzerConfig> create_offcpu_config(const ProfilerArgs& args);
    static std::unique_ptr<WallClockAnalyzerConfig> create_wallclock_config(const ProfilerArgs& args);
    
private:
    static std::vector<pid_t> parse_pid_list(const std::string& pid_str);
    static void setup_parser(argparse::ArgumentParser& program);
    static ProfilerArgs extract_args(const argparse::ArgumentParser& program);
    
    // Helper to apply common configuration
    template<typename ConfigType>
    static void apply_common_config(ConfigType& config, const ProfilerArgs& args);
};

#endif /* __ARGS_PARSER_HPP */ 