#include "wallclock_analyzer.hpp"
#include "../sampling_printer.hpp"
#include "../collectors/sampling_data.hpp"
#include <algorithm>
#include <set>

WallClockAnalyzer::WallClockAnalyzer() 
    : BaseAnalyzer("wallclock_analyzer"),
      profile_collector_(std::make_unique<ProfileCollector>()),
      offcpu_collector_(std::make_unique<OffCPUTimeCollector>()),
      sampling_frequency_(49) {
}

void WallClockAnalyzer::configure(int duration, int pid, int frequency, __u64 min_block_us) {
    sampling_frequency_ = frequency;
    
    // Configure profile collector
    auto& profile_config = profile_collector_->get_config();
    profile_config.duration = duration;
    profile_config.sample_freq = frequency;
    if (pid > 0) {
        profile_config.pids[0] = pid;
    }
    
    // Configure offcpu collector  
    auto& offcpu_config = offcpu_collector_->get_config();
    offcpu_config.duration = duration;
    offcpu_config.min_block_time = min_block_us;
    if (pid > 0) {
        offcpu_config.pids[0] = pid;
    }
}

bool WallClockAnalyzer::start() {
    if (!profile_collector_ || !offcpu_collector_) {
        return false;
    }
    
    // Start both collectors sequentially
    if (!profile_collector_->start()) {
        printf("Failed to start profile collector\n");
        return false;
    }
    
    if (!offcpu_collector_->start()) {
        printf("Failed to start offcpu collector\n");
        return false;
    }
    
    return true;
}

std::unique_ptr<FlameGraphView> WallClockAnalyzer::get_flamegraph() {
    auto per_thread_data = get_per_thread_flamegraphs();
    
    // Return combined flamegraph from all threads
    auto combined = std::make_unique<FlameGraphView>(get_name(), true);
    combined->time_unit = "normalized_samples";
    
    for (auto& [tid, flamegraph] : per_thread_data) {
        if (flamegraph && flamegraph->success) {
            for (const auto& entry : flamegraph->entries) {
                combined->add_stack_trace(
                    {}, {}, // Empty stacks since we already have folded format
                    entry.command,
                    entry.pid,
                    entry.sample_count,
                    entry.is_oncpu
                );
                // Manually add the pre-built folded stack
                if (!combined->stack_aggregation_.empty()) {
                    auto& last_entry = combined->stack_aggregation_.rbegin()->second;
                    last_entry.folded_stack = entry.folded_stack;
                }
            }
        }
    }
    
    combined->finalize();
    return combined;
}

std::map<pid_t, std::unique_ptr<FlameGraphView>> WallClockAnalyzer::get_per_thread_flamegraphs() {
    return combine_and_resolve_data();
}

std::map<pid_t, std::unique_ptr<FlameGraphView>> WallClockAnalyzer::combine_and_resolve_data() {
    std::map<pid_t, std::unique_ptr<FlameGraphView>> thread_data;
    
    if (!profile_collector_ || !offcpu_collector_) {
        return thread_data;
    }
    
    // Get data from both collectors
    auto profile_data = profile_collector_->get_data();
    auto offcpu_data = offcpu_collector_->get_data();
    
    if (!profile_data || !profile_data->success) {
        printf("Profile collector failed to provide data\n");
        return thread_data;
    }
    
    if (!offcpu_data || !offcpu_data->success) {
        printf("OffCPU collector failed to provide data\n");
        return thread_data;
    }
    
    auto* profile_sampling = dynamic_cast<SamplingData*>(profile_data.get());
    auto* offcpu_sampling = dynamic_cast<SamplingData*>(offcpu_data.get());
    
    if (!profile_sampling || !offcpu_sampling) {
        return thread_data;
    }
    
    // Collect all unique thread IDs
    std::set<pid_t> all_tids;
    for (const auto& entry : profile_sampling->entries) {
        all_tids.insert(entry.key.pid);
    }
    for (const auto& entry : offcpu_sampling->entries) {
        all_tids.insert(entry.key.pid);
    }
    
    // Create flamegraph for each thread
    for (pid_t tid : all_tids) {
        auto flamegraph = std::make_unique<FlameGraphView>(get_name() + "_thread_" + std::to_string(tid), true);
        flamegraph->time_unit = "normalized_samples";
        
        // Process on-CPU data for this thread
        for (const auto& entry : profile_sampling->entries) {
            if (entry.key.pid == tid) {
                std::vector<std::string> user_stack_symbols;
                std::vector<std::string> kernel_stack_symbols;
                
                // Resolve symbols
                if (entry.has_user_stack && !entry.user_stack.empty()) {
                    user_stack_symbols = symbolizer_->get_stack_trace_symbols(
                        const_cast<__u64*>(reinterpret_cast<const __u64*>(entry.user_stack.data())),
                        static_cast<int>(entry.user_stack.size()),
                        entry.key.pid
                    );
                    // Add [c] annotation for CPU-intensive stacks
                    for (auto& symbol : user_stack_symbols) {
                        symbol += "_[c]";
                    }
                }
                
                if (entry.has_kernel_stack && !entry.kernel_stack.empty()) {
                    kernel_stack_symbols = symbolizer_->get_stack_trace_symbols(
                        const_cast<__u64*>(reinterpret_cast<const __u64*>(entry.kernel_stack.data())),
                        static_cast<int>(entry.kernel_stack.size()),
                        0
                    );
                    // Add [c] annotation for CPU-intensive stacks
                    for (auto& symbol : kernel_stack_symbols) {
                        symbol += "_[c]";
                    }
                }
                
                flamegraph->add_stack_trace(
                    user_stack_symbols,
                    kernel_stack_symbols,
                    std::string(entry.key.comm),
                    entry.key.pid,
                    entry.value,
                    true  // is_oncpu
                );
            }
        }
        
        // Process off-CPU data with normalization for this thread
        double normalization_factor = (1.0 / sampling_frequency_) * 1000000.0;  // microseconds per sample
        
        for (const auto& entry : offcpu_sampling->entries) {
            if (entry.key.pid == tid) {
                std::vector<std::string> user_stack_symbols;
                std::vector<std::string> kernel_stack_symbols;
                
                // Resolve symbols
                if (entry.has_user_stack && !entry.user_stack.empty()) {
                    user_stack_symbols = symbolizer_->get_stack_trace_symbols(
                        const_cast<__u64*>(reinterpret_cast<const __u64*>(entry.user_stack.data())),
                        static_cast<int>(entry.user_stack.size()),
                        entry.key.pid
                    );
                    // Add [o] annotation for off-CPU stacks
                    for (auto& symbol : user_stack_symbols) {
                        symbol += "_[o]";
                    }
                }
                
                if (entry.has_kernel_stack && !entry.kernel_stack.empty()) {
                    kernel_stack_symbols = symbolizer_->get_stack_trace_symbols(
                        const_cast<__u64*>(reinterpret_cast<const __u64*>(entry.kernel_stack.data())),
                        static_cast<int>(entry.kernel_stack.size()),
                        0
                    );
                    // Add [o] annotation for off-CPU stacks
                    for (auto& symbol : kernel_stack_symbols) {
                        symbol += "_[o]";
                    }
                }
                
                // Normalize microseconds to equivalent samples for visualization
                __u64 normalized_value = static_cast<__u64>(std::max(1.0, entry.value / normalization_factor));
                
                flamegraph->add_stack_trace(
                    user_stack_symbols,
                    kernel_stack_symbols,
                    std::string(entry.key.comm),
                    entry.key.pid,
                    normalized_value,
                    false  // is_oncpu = false
                );
            }
        }
        
        flamegraph->finalize();
        thread_data[tid] = std::move(flamegraph);
    }
    
    return thread_data;
} 