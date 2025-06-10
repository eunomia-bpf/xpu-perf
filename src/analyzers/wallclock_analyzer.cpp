#include "wallclock_analyzer.hpp"
#include "collectors/sampling_data.hpp"
#include "collectors/utils.hpp"
#include <algorithm>
#include <set>
#include <sys/types.h>
#include <cstring>
#include <sstream>
#include <string>
#include <cstdlib>
#include <fstream>
#include <ctime>
#include <iostream>
#include <vector>
#include <memory>
#include <chrono>

WallClockAnalyzer::WallClockAnalyzer(std::unique_ptr<WallClockAnalyzerConfig> config) 
    : BaseAnalyzer("wallclock_analyzer"),
      profile_collector_(std::make_unique<ProfileCollector>()),
      offcpu_collector_(std::make_unique<OffCPUTimeCollector>()),
      config_(std::move(config)),
      is_multithreaded_(false),
      start_time_(std::chrono::steady_clock::now()) {
    
    configure_collectors();
}

void WallClockAnalyzer::configure_collectors() {
    if (!profile_collector_ || !offcpu_collector_ || !config_) {
        return;
    }
    
    // Configure profile collector
    auto& profile_config = profile_collector_->get_config();
    profile_config.duration = config_->duration;
    profile_config.attr.sample_freq = config_->frequency;
    profile_config.pids = config_->pids;
    profile_config.tids = config_->tids;
    
    // Configure offcpu collector  
    auto& offcpu_config = offcpu_collector_->get_config();
    offcpu_config.duration = config_->duration;
    offcpu_config.min_block_time = config_->min_block_us;
    offcpu_config.pids = config_->pids;
    offcpu_config.tids = config_->tids;
}

bool WallClockAnalyzer::discover_threads() {
    if (config_->pids.empty()) {
        is_multithreaded_ = false;
        return false;
    }
    
    pid_t main_pid = config_->pids[0];
    std::stringstream cmd;
    cmd << "ps -T -p " << main_pid << " 2>/dev/null";
    
    FILE* pipe = popen(cmd.str().c_str(), "r");
    if (!pipe) {
        is_multithreaded_ = false;
        return false;
    }
    
    char buffer[1024];
    std::vector<std::string> lines;
    
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        lines.push_back(std::string(buffer));
    }
    pclose(pipe);
    
    detected_threads_.clear();
    
    // Skip header line
    for (size_t i = 1; i < lines.size(); i++) {
        std::istringstream iss(lines[i]);
        std::string pid_str, tid_str, tty, time_str;
        
        if (iss >> pid_str >> tid_str >> tty >> time_str) {
            try {
                pid_t tid = std::stoi(tid_str);
                
                // Read remaining as command
                std::string cmd_part;
                std::string full_cmd;
                while (iss >> cmd_part) {
                    if (!full_cmd.empty()) full_cmd += " ";
                    full_cmd += cmd_part;
                }
                
                ThreadInfo thread_info;
                thread_info.tid = tid;
                thread_info.command = full_cmd;
                thread_info.role = get_thread_role(tid, full_cmd);
                
                detected_threads_.push_back(thread_info);
                
            } catch (const std::exception&) {
                continue;
            }
        }
    }
    
    is_multithreaded_ = detected_threads_.size() > 1;
    
    if (is_multithreaded_) {
        std::cout << "Detected " << detected_threads_.size() << " threads" << std::endl;
    }
    
    return is_multithreaded_;
}

std::string WallClockAnalyzer::get_thread_role(pid_t tid, const std::string& cmd) {
    if (!config_->pids.empty() && tid == config_->pids[0]) {
        return "main";
    } else {
        std::stringstream ss;
        ss << "thread_" << tid;
        return ss.str();
    }
}

bool WallClockAnalyzer::start() {
    if (!profile_collector_ || !offcpu_collector_) {
        return false;
    }
    
    // Discover threads first
    discover_threads();
    
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

double WallClockAnalyzer::get_actual_runtime_seconds() const {
    auto now = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time_);
    return duration.count() / 1000.0;
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
        return thread_data;
    }
    
    if (!offcpu_data || !offcpu_data->success) {
        return thread_data;
    }
    
    auto* profile_sampling = dynamic_cast<SamplingData*>(profile_data.get());
    auto* offcpu_sampling = dynamic_cast<SamplingData*>(offcpu_data.get());
    
    if (!profile_sampling || !offcpu_sampling) {
        std::cerr << "Invalid sampling data format" << std::endl;
        return thread_data;
    }
    
    // Collect all unique thread IDs from both datasets
    std::set<__u32> all_tids;
    for (const auto& entry : profile_sampling->entries) {
        all_tids.insert(entry.key.pid);
    }
    for (const auto& entry : offcpu_sampling->entries) {
        all_tids.insert(entry.key.pid);
    }
    
    // Normalization factor for on-CPU data (convert samples to microseconds)
    double sample_to_us_factor = (1.0 / config_->frequency) * 1000000.0;
    
    // Create combined flamegraph for each thread
    for (__u32 tid : all_tids) {
        auto flamegraph = std::make_unique<FlameGraphView>(get_name() + "_thread_" + std::to_string(tid), true);
        flamegraph->time_unit = "microseconds";
        
        int oncpu_count = 0, offcpu_count = 0;
        
        // Add ALL on-CPU data for this thread (convert samples to microseconds)
        for (const auto& entry : profile_sampling->entries) {
            if (entry.key.pid == tid) {
                oncpu_count++;
                __u64* user_stack = nullptr;
                int user_stack_size = 0;
                __u64* kernel_stack = nullptr;
                int kernel_stack_size = 0;
                
                if (entry.has_user_stack && !entry.user_stack.empty()) {
                    user_stack = const_cast<__u64*>(reinterpret_cast<const __u64*>(entry.user_stack.data()));
                    user_stack_size = static_cast<int>(entry.user_stack.size());
                }
                
                if (entry.has_kernel_stack && !entry.kernel_stack.empty()) {
                    kernel_stack = const_cast<__u64*>(reinterpret_cast<const __u64*>(entry.kernel_stack.data()));
                    kernel_stack_size = static_cast<int>(entry.kernel_stack.size());
                }
                
                // Convert samples to microseconds
                __u64 microseconds_value = static_cast<__u64>(std::max(1.0, entry.value * sample_to_us_factor));
                
                flamegraph->add_stack_trace_raw(
                    user_stack, user_stack_size,
                    kernel_stack, kernel_stack_size,
                    std::string(entry.key.comm),
                    entry.key.pid,
                    microseconds_value,
                    true  // is_oncpu
                );
            }
        }
        
        // Add ALL off-CPU data for this thread (already in microseconds)
        for (const auto& entry : offcpu_sampling->entries) {
            if (entry.key.pid == tid) {
                offcpu_count++;
                __u64* user_stack = nullptr;
                int user_stack_size = 0;
                __u64* kernel_stack = nullptr;
                int kernel_stack_size = 0;
                
                if (entry.has_user_stack && !entry.user_stack.empty()) {
                    user_stack = const_cast<__u64*>(reinterpret_cast<const __u64*>(entry.user_stack.data()));
                    user_stack_size = static_cast<int>(entry.user_stack.size());
                }
                
                if (entry.has_kernel_stack && !entry.kernel_stack.empty()) {
                    kernel_stack = const_cast<__u64*>(reinterpret_cast<const __u64*>(entry.kernel_stack.data()));
                    kernel_stack_size = static_cast<int>(entry.kernel_stack.size());
                }
                
                // Off-CPU data is already in microseconds, use as-is
                __u64 microseconds_value = std::max(static_cast<__u64>(1), entry.value);
                
                flamegraph->add_stack_trace_raw(
                    user_stack, user_stack_size,
                    kernel_stack, kernel_stack_size,
                    std::string(entry.key.comm),
                    entry.key.pid,
                    microseconds_value,
                    false  // is_oncpu = false
                );
            }
        }
        
        flamegraph->finalize();
        
        // Only add threads that have some data
        if (oncpu_count > 0 || offcpu_count > 0) {
            thread_data[tid] = std::move(flamegraph);
        }
    }
    
    return thread_data;
} 