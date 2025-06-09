#ifndef __FLAMEGRAPH_VIEW_HPP
#define __FLAMEGRAPH_VIEW_HPP

#include "collectors/bpf_event.h"
#include "collectors/collector_interface.hpp"
#include <vector>
#include <string>
#include <map>
#include <cstdint>
#include <linux/types.h>
#include <sys/types.h>
#include <memory>

// Forward declarations
class SamplingData;
class SymbolResolver;

// Aggregated flamegraph entry representing a unique stack trace
struct FlameGraphEntry {
    std::string folded_stack;        // Complete stack trace in folded format (func1;func2;func3)
    std::string command;             // Process/thread command name
    pid_t pid;                       // Process/thread ID
    __u64 sample_count;              // Aggregated sample count for this stack
    double percentage;               // Percentage of total samples
    int stack_depth;                 // Depth of this stack trace
    bool is_oncpu;                   // True if this is on-CPU data (vs off-CPU)
};

// FlameGraph view with aggregated and hierarchical stack trace data
class FlameGraphView : public CollectorData {
public:
    ~FlameGraphView() override = default;
    
    std::vector<FlameGraphEntry> entries;
    __u64 total_samples;
    double total_time_seconds;
    std::string analyzer_name;
    std::string time_unit;  // "samples", "microseconds", "seconds", etc.
    
    FlameGraphView(const std::string& analyzer_name, bool success = true);
    
    // Core flamegraph functionality
    void add_stack_trace(const std::vector<std::string>& user_stack,
                        const std::vector<std::string>& kernel_stack,
                        const std::string& command,
                        pid_t pid,
                        __u64 value,
                        bool is_oncpu = true,
                        bool include_delimiter = true);
    
    // Add stack trace with raw addresses (will resolve symbols internally)
    void add_stack_trace_raw(__u64* user_stack, int user_stack_size,
                            __u64* kernel_stack, int kernel_stack_size,
                            const std::string& command,
                            pid_t pid,
                            __u64 value,
                            bool is_oncpu = true);
    
    void finalize();  // Call after adding all stacks to calculate percentages
    
    // Output formats
    std::string to_folded_format() const;
    std::string to_summary() const;
    
    // Analysis methods
    std::vector<FlameGraphEntry> get_top_stacks(int limit = 10) const;
    std::map<std::string, __u64> get_function_totals() const;
    double get_oncpu_percentage() const;
    double get_offcpu_percentage() const;
    
    // Thread-specific analysis
    std::map<pid_t, std::vector<FlameGraphEntry>> group_by_thread() const;
    
    // Access to symbolizer for direct use if needed
    SymbolResolver* get_symbolizer() { return symbolizer_.get(); }
    const SymbolResolver* get_symbolizer() const { return symbolizer_.get(); }
    
private:
    std::unique_ptr<SymbolResolver> symbolizer_;
    std::map<std::string, FlameGraphEntry> stack_aggregation_;  // For deduplication during building
    
    std::string build_folded_stack(const std::vector<std::string>& user_stack,
                                  const std::vector<std::string>& kernel_stack,
                                  const std::string& command,
                                  bool include_delimiter = true) const;
    
    int calculate_stack_depth(const std::string& folded_stack) const;
};

// Utility function to convert sampling data to flamegraph
std::unique_ptr<FlameGraphView> sampling_data_to_flamegraph(
    const SamplingData& data, 
    const std::string& analyzer_name,
    bool is_oncpu = true);

#endif /* __FLAMEGRAPH_VIEW_HPP */ 