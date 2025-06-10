#ifndef __FLAMEGRAPH_GENERATOR_HPP
#define __FLAMEGRAPH_GENERATOR_HPP

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <fstream>
#include <filesystem>

// Forward declaration
class FlameGraphView;

struct FlamegraphEntry {
    std::string stack_trace;
    uint64_t value;
    bool is_oncpu;  // true for on-CPU, false for off-CPU
};

struct ThreadInfo {
    pid_t tid;
    std::string command;
    std::string role;
};

class FlamegraphGenerator {
private:
    std::string output_dir_;
    int sampling_freq_;
    double actual_wall_clock_time_;  // Actual runtime in seconds
    
public:
    FlamegraphGenerator(const std::string& output_dir, int freq, double actual_wall_clock_time);
    
    // Setup flamegraph tools (download if needed)
    bool setup_flamegraph_tools();
    
    // Generate folded format file
    std::string generate_folded_file(const std::vector<FlamegraphEntry>& entries, 
                                   const std::string& filename_prefix);
    
    // Generate SVG from folded file
    std::string generate_svg_from_folded(const std::string& folded_file, 
                                       const std::string& title = "");
    
    // Add annotations to stack traces
    std::string add_annotation(const std::string& stack, bool is_oncpu);
    
    // Generate analysis file
    void generate_analysis_file(const std::string& filename,
                              const std::vector<FlamegraphEntry>& entries,
                              const std::string& analysis_type = "single");
    
    // Create output directory
    bool create_output_directory();
    
    // New methods for wallclock analyzer
    void generate_single_flamegraph(const std::map<pid_t, std::unique_ptr<FlameGraphView>>& per_thread_data,
                                   const std::vector<pid_t>& pids);
    
    void generate_multithread_flamegraphs(const std::map<pid_t, std::unique_ptr<FlameGraphView>>& per_thread_data,
                                        const std::vector<ThreadInfo>& detected_threads);

private:
    std::string get_flamegraph_script_path();
    bool download_flamegraph_tools();
    void create_custom_flamegraph_script();
    
    // Helper method for converting FlameGraphView to FlamegraphEntry
    std::vector<FlamegraphEntry> convert_flamegraph_to_entries(const FlameGraphView& flamegraph);
    
    // Helper method for getting thread role
    std::string get_thread_role(pid_t tid, const std::string& cmd, const std::vector<pid_t>& pids);
};

#endif /* __FLAMEGRAPH_GENERATOR_HPP */ 