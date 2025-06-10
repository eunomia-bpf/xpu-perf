#ifndef __FLAMEGRAPH_GENERATOR_HPP
#define __FLAMEGRAPH_GENERATOR_HPP

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <fstream>
#include <filesystem>

struct FlamegraphEntry {
    std::string stack_trace;
    uint64_t value;
    bool is_oncpu;  // true for on-CPU, false for off-CPU
};

class FlamegraphGenerator {
private:
    std::string output_dir_;
    int sampling_freq_;
    int duration_;
    double actual_wall_clock_time_;  // Actual runtime in seconds
    
public:
    FlamegraphGenerator(const std::string& output_dir, int freq, int duration);
    
    // Set the actual wall clock time that was measured
    void set_actual_wall_clock_time(double actual_time_seconds);
    
    // Setup flamegraph tools (download if needed)
    bool setup_flamegraph_tools();
    
    // Generate folded format file
    std::string generate_folded_file(const std::vector<FlamegraphEntry>& entries, 
                                   const std::string& filename_prefix);
    
    // Generate SVG from folded file
    std::string generate_svg_from_folded(const std::string& folded_file, 
                                       const std::string& title = "");
    
    // Normalize off-CPU time to equivalent samples
    uint64_t normalize_offcpu_time(uint64_t microseconds);
    
    // Add annotations to stack traces
    std::string add_annotation(const std::string& stack, bool is_oncpu);
    
    // Generate analysis file
    void generate_analysis_file(const std::string& filename,
                              const std::vector<FlamegraphEntry>& entries,
                              const std::string& analysis_type = "single");
    
    // Create output directory
    bool create_output_directory();

private:
    std::string get_flamegraph_script_path();
    bool download_flamegraph_tools();
    void create_custom_flamegraph_script();
};

#endif /* __FLAMEGRAPH_GENERATOR_HPP */ 