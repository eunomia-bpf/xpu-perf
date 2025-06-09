#include "wallclock_analyzer.hpp"
#include "collectors/sampling_data.hpp"
#include <algorithm>
#include <set>
#include <sys/types.h>

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
                // Use the add_stack_trace method with already resolved symbols
                std::vector<std::string> user_stack;
                std::vector<std::string> kernel_stack;
                
                // Parse the folded stack back into components if needed
                // For now, we'll add with empty stacks and the entry will be created with the command name
                combined->add_stack_trace(
                    user_stack, kernel_stack,
                    entry.command,
                    entry.pid,
                    entry.sample_count,
                    entry.is_oncpu
                );
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
    std::set<__u32> all_tids;
    for (const auto& entry : profile_sampling->entries) {
        all_tids.insert(entry.key.pid);
    }
    for (const auto& entry : offcpu_sampling->entries) {
        all_tids.insert(entry.key.pid);
    }
    
    // Create flamegraph for each thread
    for (__u32 tid : all_tids) {
        auto flamegraph = std::make_unique<FlameGraphView>(get_name() + "_thread_" + std::to_string(tid), true);
        flamegraph->time_unit = "normalized_samples";
        
        // Process on-CPU data for this thread
        for (const auto& entry : profile_sampling->entries) {
            if (entry.key.pid == tid) {
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
                    true  // is_oncpu
                );
            }
        }
        
        // Process off-CPU data with normalization for this thread
        double normalization_factor = (1.0 / sampling_frequency_) * 1000000.0;  // microseconds per sample
        
        for (const auto& entry : offcpu_sampling->entries) {
            if (entry.key.pid == tid) {
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
                
                // Normalize microseconds to equivalent samples for visualization
                __u64 normalized_value = static_cast<__u64>(std::max(1.0, entry.value / normalization_factor));
                
                flamegraph->add_stack_trace_raw(
                    user_stack, user_stack_size,
                    kernel_stack, kernel_stack_size,
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