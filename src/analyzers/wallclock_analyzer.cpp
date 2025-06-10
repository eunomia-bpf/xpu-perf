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

WallClockAnalyzer::WallClockAnalyzer(std::unique_ptr<WallClockAnalyzerConfig> config) 
    : BaseAnalyzer("wallclock_analyzer"),
      profile_collector_(std::make_unique<ProfileCollector>()),
      offcpu_collector_(std::make_unique<OffCPUTimeCollector>()),
      config_(std::move(config)),
      is_multithreaded_(false) {
    
    configure_collectors();
    
    // Initialize flamegraph generator
    std::string output_dir = create_output_directory();
    flamegraph_gen_ = std::make_unique<FlamegraphGenerator>(output_dir, config_->frequency, config_->duration);
}

std::string WallClockAnalyzer::create_output_directory() {
    auto now = std::time(nullptr);
    std::stringstream ss;
    
    if (!config_->pids.empty()) {
        ss << "wallclock_profile_pid" << config_->pids[0] << "_" << now;
    } else {
        ss << "wallclock_profile_" << now;
    }
    
    return ss.str();
}

void WallClockAnalyzer::configure_collectors() {
    if (!profile_collector_ || !offcpu_collector_ || !config_) {
        return;
    }
    
    // Configure profile collector
    auto& profile_config = profile_collector_->get_config();
    profile_config.duration = config_->duration;
    profile_config.sample_freq = config_->frequency;
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
        std::cout << "Multi-threaded application detected (" << detected_threads_.size() << " threads)" << std::endl;
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

void WallClockAnalyzer::generate_flamegraph_files() {
    // Get data from both collectors
    auto profile_data = profile_collector_->get_data();
    auto offcpu_data = offcpu_collector_->get_data();
    
    if (!profile_data || !profile_data->success || !offcpu_data || !offcpu_data->success) {
        std::cerr << "Failed to get data from collectors" << std::endl;
        return;
    }
    
    auto* profile_sampling = dynamic_cast<SamplingData*>(profile_data.get());
    auto* offcpu_sampling = dynamic_cast<SamplingData*>(offcpu_data.get());
    
    if (!profile_sampling || !offcpu_sampling) {
        std::cerr << "Invalid sampling data format" << std::endl;
        return;
    }
    
    // Generate flamegraphs using the new approach
    auto per_thread_data = get_per_thread_flamegraphs();
    
    // Let flamegraph generator handle all file generation and output
    if (is_multithreaded_) {
        flamegraph_gen_->generate_multithread_flamegraphs(per_thread_data, detected_threads_);
    } else {
        flamegraph_gen_->generate_single_flamegraph(per_thread_data, config_->pids);
    }
}

std::unique_ptr<FlameGraphView> WallClockAnalyzer::get_flamegraph() {
    // Generate flamegraph files as side effect
    generate_flamegraph_files();
    
    // Return combined flamegraph from all threads for compatibility
    auto per_thread_data = get_per_thread_flamegraphs();
    
    auto combined = std::make_unique<FlameGraphView>(get_name(), true);
    combined->time_unit = "normalized_samples";
    
    for (auto& [tid, flamegraph] : per_thread_data) {
        if (flamegraph && flamegraph->success) {
            for (const auto& entry : flamegraph->entries) {
                std::vector<std::string> user_stack;
                std::vector<std::string> kernel_stack;
                
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
    
    // Create flamegraph for each thread with proper normalization
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
                
                if (entry.has_user_stack && !entry.user_stack.empty()) {
                    user_stack = const_cast<__u64*>(reinterpret_cast<const __u64*>(entry.user_stack.data()));
                    user_stack_size = static_cast<int>(entry.user_stack.size());
                }
                
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
        // Normalize using the same method as the Python script
        double normalization_factor = (1.0 / config_->frequency) * 1000000.0;  // microseconds per sample
        
        for (const auto& entry : offcpu_sampling->entries) {
            if (entry.key.pid == tid) {
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