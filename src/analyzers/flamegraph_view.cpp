#include "flamegraph_view.hpp"
#include "collectors/sampling_data.hpp"
#include "symbol_resolver.hpp"
#include <algorithm>
#include <sstream>
#include <iostream>
#include <iomanip>

FlameGraphView::FlameGraphView(const std::string& analyzer_name, bool success) 
    : CollectorData(analyzer_name, success, CollectorData::Type::SAMPLING), 
      total_samples(0), total_time_seconds(0.0), analyzer_name(analyzer_name), time_unit("samples"),
      symbolizer_(std::make_unique<SymbolResolver>()) {
}

void FlameGraphView::add_stack_trace(const std::vector<std::string>& user_stack,
                                   const std::vector<std::string>& kernel_stack,
                                   const std::string& command,
                                   pid_t pid,
                                   __u64 value,
                                   bool is_oncpu,
                                   bool include_delimiter) {
    
    std::vector<std::string> folded_stack = build_folded_stack(user_stack, kernel_stack, command, include_delimiter);
    
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

void FlameGraphView::add_stack_trace_raw(__u64* user_stack, int user_stack_size,
                                        __u64* kernel_stack, int kernel_stack_size,
                                        const std::string& command,
                                        pid_t pid,
                                        __u64 value,
                                        bool is_oncpu) {
    if (!symbolizer_) {
        return;
    }
    
    std::vector<std::string> user_stack_symbols;
    std::vector<std::string> kernel_stack_symbols;
    
    // Resolve user stack symbols
    if (user_stack && user_stack_size > 0) {
        user_stack_symbols = symbolizer_->get_stack_trace_symbols(
            user_stack, user_stack_size, pid);
    }
    
    // Resolve kernel stack symbols
    if (kernel_stack && kernel_stack_size > 0) {
        kernel_stack_symbols = symbolizer_->get_stack_trace_symbols(
            kernel_stack, kernel_stack_size, 0);  // kernel symbols use pid 0
    }
    
    // Add to flamegraph
    add_stack_trace(user_stack_symbols, kernel_stack_symbols, command, pid, value, is_oncpu);
}

void FlameGraphView::finalize() {
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

std::vector<std::string> FlameGraphView::build_folded_stack(const std::vector<std::string>& user_stack,
                                                          const std::vector<std::string>& kernel_stack,
                                                          const std::string& command,
                                                          bool include_delimiter) const {
    std::vector<std::string> folded_stack;
    
    // Start with the command name as the root
    folded_stack.push_back(command);
    
    // Add user stack (bottom to top for flamegraph)
    if (!user_stack.empty()) {
        for (auto it = user_stack.rbegin(); it != user_stack.rend(); ++it) {
            folded_stack.push_back(*it);
        }
    }
    
    // Add delimiter between user and kernel if both exist
//    if (include_delimiter && !user_stack.empty() && !kernel_stack.empty()) {
//        folded_stack.push_back("--");
//    }
    
    // Add kernel stack (bottom to top for flamegraph)  
    if (!kernel_stack.empty()) {
        for (auto it = kernel_stack.rbegin(); it != kernel_stack.rend(); ++it) {
            folded_stack.push_back(*it);
        }
    }
    
    return folded_stack;
}

int FlameGraphView::calculate_stack_depth(const std::vector<std::string>& folded_stack) const {
    return static_cast<int>(folded_stack.size() - 1);  // Subtract 1 to exclude the command name
}

std::string FlameGraphView::to_folded_format() const {
    std::ostringstream oss;
    for (const auto& entry : entries) {
        // Convert vector back to semicolon-separated string for output
        if (!entry.folded_stack.empty()) {
            oss << entry.folded_stack[0];  // First element (command)
            for (size_t i = 1; i < entry.folded_stack.size(); ++i) {
                oss << ";" << entry.folded_stack[i];
            }
        }
        oss << " " << entry.sample_count << "\n";
    }
    return oss.str();
}

std::string FlameGraphView::to_readable_format() const {
    std::ostringstream oss;
    oss << "FlameGraph Readable Format: " << analyzer_name << "\n";
    oss << "Total samples: " << total_samples << " " << time_unit << "\n";
    oss << "Unique stacks: " << entries.size() << "\n\n";
    
    for (size_t i = 0; i < entries.size() && i < 20; ++i) {  // Show top 20 stacks
        const auto& entry = entries[i];
        oss << "Stack #" << (i + 1) << " - " << entry.sample_count << " samples (" 
            << std::fixed << std::setprecision(2) << entry.percentage << "%)\n";
        oss << "Process: " << entry.command << " (PID: " << entry.pid << ")\n";
        oss << "Type: " << (entry.is_oncpu ? "On-CPU" : "Off-CPU") << "\n";
        oss << "Call Stack (depth " << entry.stack_depth << "):\n";
        
        // Print stack trace with indentation
        for (size_t j = 0; j < entry.folded_stack.size(); ++j) {
            std::string indent(j * 2, ' ');  // 2 spaces per level
            oss << indent << "-> " << entry.folded_stack[j] << "\n";
        }
        oss << "\n";
    }
    
    return oss.str();
}

std::string FlameGraphView::to_summary() const {
    std::ostringstream oss;
    oss << "FlameGraph Summary: " << analyzer_name << "\n";
    oss << "Total samples: " << total_samples << " " << time_unit << "\n";
    oss << "Unique stacks: " << entries.size() << "\n";
    oss << "On-CPU percentage: " << get_oncpu_percentage() << "%\n";
    oss << "Off-CPU percentage: " << get_offcpu_percentage() << "%\n";
    return oss.str();
}

std::map<std::string, __u64> FlameGraphView::get_function_totals() const {
    std::map<std::string, __u64> function_totals;
    
    for (const auto& entry : entries) {
        // Iterate through each function in the vector
        for (const auto& function : entry.folded_stack) {
            if (!function.empty() && function != "--") {  // Skip delimiter
                function_totals[function] += entry.sample_count;
            }
        }
    }
    
    return function_totals;
}

double FlameGraphView::get_oncpu_percentage() const {
    __u64 oncpu_samples = 0;
    for (const auto& entry : entries) {
        if (entry.is_oncpu) {
            oncpu_samples += entry.sample_count;
        }
    }
    return total_samples > 0 ? (double(oncpu_samples) / total_samples) * 100.0 : 0.0;
}

double FlameGraphView::get_offcpu_percentage() const {
    return 100.0 - get_oncpu_percentage();
}

std::map<pid_t, std::vector<FlameGraphEntry>> FlameGraphView::group_by_thread() const {
    std::map<pid_t, std::vector<FlameGraphEntry>> thread_groups;
    
    for (const auto& entry : entries) {
        thread_groups[entry.pid].push_back(entry);
    }
    
    return thread_groups;
}

// Utility function implementation - simplified since FlameGraphView now manages symbolizer
std::unique_ptr<FlameGraphView> FlameGraphView::sampling_data_to_flamegraph(
    const SamplingData& data, 
    const std::string& analyzer_name,
    bool is_oncpu) {
    
    auto flamegraph = std::make_unique<FlameGraphView>(analyzer_name, true);
    
    if (!flamegraph->symbolizer_ || !flamegraph->symbolizer_->is_valid()) {
        flamegraph->success = false;
        return flamegraph;
    }
    
    // Set appropriate time unit
    flamegraph->time_unit = is_oncpu ? "samples" : "microseconds";
    
    for (const auto& entry : data.entries) {
        __u64* user_stack = nullptr;
        int user_stack_size = 0;
        __u64* kernel_stack = nullptr;
        int kernel_stack_size = 0;
        
        // Prepare user stack
        if (entry.has_user_stack && !entry.user_stack.empty()) {
            user_stack = const_cast<__u64*>(reinterpret_cast<const __u64*>(entry.user_stack.data()));
            user_stack_size = static_cast<int>(entry.user_stack.size());
        }
        
        // Prepare kernel stack
        if (entry.has_kernel_stack && !entry.kernel_stack.empty()) {
            kernel_stack = const_cast<__u64*>(reinterpret_cast<const __u64*>(entry.kernel_stack.data()));
            kernel_stack_size = static_cast<int>(entry.kernel_stack.size());
        }
        
        flamegraph->add_stack_trace_raw(
            user_stack, user_stack_size,
            kernel_stack, kernel_stack_size,
            std::string(entry.key.comm),
            entry.key.pid,
            entry.value,
            is_oncpu
        );
    }
    
    flamegraph->finalize();
    return flamegraph;
} 