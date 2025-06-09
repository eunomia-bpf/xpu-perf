#ifndef __FLAMEGRAPH_VIEW_HPP
#define __FLAMEGRAPH_VIEW_HPP

#include "../collectors/bpf_event.h"
#include "../collectors/collector_interface.hpp"
#include <vector>
#include <string>
#include <map>
#include <cstdint>
#include <linux/types.h>
#include <algorithm>
#include <sstream>
#include <iostream>

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
    
    FlameGraphView(const std::string& analyzer_name, bool success = true) 
        : CollectorData(analyzer_name, success, Type::SAMPLING), 
          total_samples(0), total_time_seconds(0.0), analyzer_name(analyzer_name), time_unit("samples") {}
    
    // Core flamegraph functionality
    void add_stack_trace(const std::vector<std::string>& user_stack,
                        const std::vector<std::string>& kernel_stack,
                        const std::string& command,
                        pid_t pid,
                        __u64 value,
                        bool is_oncpu = true,
                        bool include_delimiter = true);
    
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
    
public:
    std::map<std::string, FlameGraphEntry> stack_aggregation_;  // For deduplication during building
    
    std::string build_folded_stack(const std::vector<std::string>& user_stack,
                                  const std::vector<std::string>& kernel_stack,
                                  const std::string& command,
                                  bool include_delimiter = true) const;
    
    int calculate_stack_depth(const std::string& folded_stack) const;
};

// Implementation
inline void FlameGraphView::add_stack_trace(const std::vector<std::string>& user_stack,
                                           const std::vector<std::string>& kernel_stack,
                                           const std::string& command,
                                           pid_t pid,
                                           __u64 value,
                                           bool is_oncpu,
                                           bool include_delimiter) {
    
    std::string folded_stack = build_folded_stack(user_stack, kernel_stack, command, include_delimiter);
    
    // Aggregate identical stacks
    auto& entry = stack_aggregation_[folded_stack];
    if (entry.folded_stack.empty()) {
        // New entry
        entry.folded_stack = folded_stack;
        entry.command = command;
        entry.pid = pid;
        entry.sample_count = value;
        entry.is_oncpu = is_oncpu;
        entry.stack_depth = calculate_stack_depth(folded_stack);
    } else {
        // Aggregate existing entry
        entry.sample_count += value;
    }
    
    total_samples += value;
}

inline void FlameGraphView::finalize() {
    entries.clear();
    entries.reserve(stack_aggregation_.size());
    
    // Convert map to vector and calculate percentages
    for (const auto& [stack, entry] : stack_aggregation_) {
        FlameGraphEntry final_entry = entry;
        final_entry.percentage = total_samples > 0 ? (double(entry.sample_count) / total_samples) * 100.0 : 0.0;
        entries.push_back(final_entry);
    }
    
    // Sort by sample count (descending)
    std::sort(entries.begin(), entries.end(), 
              [](const FlameGraphEntry& a, const FlameGraphEntry& b) {
                  return a.sample_count > b.sample_count;
              });
}

inline std::string FlameGraphView::build_folded_stack(const std::vector<std::string>& user_stack,
                                                     const std::vector<std::string>& kernel_stack,
                                                     const std::string& command,
                                                     bool include_delimiter) const {
    std::ostringstream oss;
    oss << command;
    
    // Add user stack (bottom to top for flamegraph)
    if (!user_stack.empty()) {
        for (auto it = user_stack.rbegin(); it != user_stack.rend(); ++it) {
            oss << ";" << *it;
        }
    }
    
    // Add delimiter between user and kernel if both exist
    if (include_delimiter && !user_stack.empty() && !kernel_stack.empty()) {
        oss << ";--";
    }
    
    // Add kernel stack (bottom to top for flamegraph)  
    if (!kernel_stack.empty()) {
        for (auto it = kernel_stack.rbegin(); it != kernel_stack.rend(); ++it) {
            oss << ";" << *it;
        }
    }
    
    return oss.str();
}

inline int FlameGraphView::calculate_stack_depth(const std::string& folded_stack) const {
    return std::count(folded_stack.begin(), folded_stack.end(), ';');
}

inline std::string FlameGraphView::to_folded_format() const {
    std::ostringstream oss;
    for (const auto& entry : entries) {
        oss << entry.folded_stack << " " << entry.sample_count << "\n";
    }
    return oss.str();
}

inline std::string FlameGraphView::to_summary() const {
    std::ostringstream oss;
    oss << "FlameGraph Summary: " << analyzer_name << "\n";
    oss << "Total samples: " << total_samples << " " << time_unit << "\n";
    oss << "Unique stacks: " << entries.size() << "\n";
    oss << "On-CPU percentage: " << get_oncpu_percentage() << "%\n";
    oss << "Off-CPU percentage: " << get_offcpu_percentage() << "%\n";
    return oss.str();
}

inline std::vector<FlameGraphEntry> FlameGraphView::get_top_stacks(int limit) const {
    std::vector<FlameGraphEntry> top_stacks;
    int count = std::min(limit, static_cast<int>(entries.size()));
    top_stacks.reserve(count);
    
    for (int i = 0; i < count; i++) {
        top_stacks.push_back(entries[i]);
    }
    
    return top_stacks;
}

inline std::map<std::string, __u64> FlameGraphView::get_function_totals() const {
    std::map<std::string, __u64> function_totals;
    
    for (const auto& entry : entries) {
        // Split the folded stack and count each function
        std::istringstream iss(entry.folded_stack);
        std::string function;
        
        while (std::getline(iss, function, ';')) {
            if (!function.empty() && function != "--") {  // Skip delimiter
                function_totals[function] += entry.sample_count;
            }
        }
    }
    
    return function_totals;
}

inline double FlameGraphView::get_oncpu_percentage() const {
    __u64 oncpu_samples = 0;
    for (const auto& entry : entries) {
        if (entry.is_oncpu) {
            oncpu_samples += entry.sample_count;
        }
    }
    return total_samples > 0 ? (double(oncpu_samples) / total_samples) * 100.0 : 0.0;
}

inline double FlameGraphView::get_offcpu_percentage() const {
    return 100.0 - get_oncpu_percentage();
}

inline std::map<pid_t, std::vector<FlameGraphEntry>> FlameGraphView::group_by_thread() const {
    std::map<pid_t, std::vector<FlameGraphEntry>> thread_groups;
    
    for (const auto& entry : entries) {
        thread_groups[entry.pid].push_back(entry);
    }
    
    return thread_groups;
}

#endif /* __FLAMEGRAPH_VIEW_HPP */ 