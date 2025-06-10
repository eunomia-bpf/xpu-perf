#ifndef __WALLCLOCK_ANALYZER_HPP
#define __WALLCLOCK_ANALYZER_HPP

#include "base_analyzer.hpp"
#include "analyzer_config.hpp"
#include "collectors/oncpu/profile.hpp"
#include "collectors/offcpu/offcputime.hpp"
#include "collectors/utils.hpp"
#include "../flamegraph_generator.hpp"
#include <memory>
#include <map>
#include <vector>
#include <string>
#include <cstring>
#include <algorithm>
#include <chrono>

// Forward declarations
class ProfileCollector;
class OffCPUTimeCollector;

class WallClockAnalyzer : public BaseAnalyzer {
private:
    std::unique_ptr<ProfileCollector> profile_collector_;
    std::unique_ptr<OffCPUTimeCollector> offcpu_collector_;
    std::unique_ptr<WallClockAnalyzerConfig> config_;
    
    // Thread analysis data
    std::vector<ThreadInfo> detected_threads_;
    bool is_multithreaded_;
    
    // Runtime tracking
    std::chrono::steady_clock::time_point start_time_;
    
    // Helper methods
    std::map<pid_t, std::unique_ptr<FlameGraphView>> combine_and_resolve_data();

public:
    explicit WallClockAnalyzer(std::unique_ptr<WallClockAnalyzerConfig> config);
    virtual ~WallClockAnalyzer() = default;
    
    // IAnalyzer interface
    bool start() override;
    std::map<pid_t, std::unique_ptr<FlameGraphView>> get_per_thread_flamegraphs() override;
    
    // New functionality matching Python script
    bool discover_threads();
    
    // Config access
    const WallClockAnalyzerConfig& get_config() const { return *config_; }
    
    // Access to individual collectors for fine-grained control
    ProfileCollector* get_profile_collector() { return profile_collector_.get(); }
    OffCPUTimeCollector* get_offcpu_collector() { return offcpu_collector_.get(); }
    
    // Access to thread information
    const std::vector<ThreadInfo>& get_detected_threads() const { return detected_threads_; }
    bool is_multithreaded() const { return is_multithreaded_; }
    
    // Runtime information
    double get_actual_runtime_seconds() const;

private:
    void configure_collectors();
    std::string get_thread_role(pid_t tid, const std::string& cmd);
    std::string create_output_directory();
};

#endif /* __WALLCLOCK_ANALYZER_HPP */ 