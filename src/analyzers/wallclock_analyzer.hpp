#ifndef __WALLCLOCK_ANALYZER_HPP
#define __WALLCLOCK_ANALYZER_HPP

#include "base_analyzer.hpp"
#include "collectors/oncpu/profile.hpp"
#include "collectors/offcpu/offcputime.hpp"
#include <memory>
#include <map>

// Forward declarations
class ProfileCollector;
class OffCPUTimeCollector;

class WallClockAnalyzer : public BaseAnalyzer {
private:
    std::unique_ptr<ProfileCollector> profile_collector_;
    std::unique_ptr<OffCPUTimeCollector> offcpu_collector_;
    
    // Normalization configuration
    int sampling_frequency_;
    
    // Helper methods
    std::map<pid_t, std::unique_ptr<FlameGraphView>> combine_and_resolve_data();

public:
    WallClockAnalyzer();
    virtual ~WallClockAnalyzer() = default;
    
    // IAnalyzer interface
    bool start() override;
    std::unique_ptr<FlameGraphView> get_flamegraph() override;
    std::map<pid_t, std::unique_ptr<FlameGraphView>> get_per_thread_flamegraphs() override;
    
    // Config access - combines both collectors' configs
    void configure(int duration, int pid = 0, int frequency = 49, __u64 min_block_us = 1000);
    void set_sampling_frequency(int freq) { sampling_frequency_ = freq; }
    int get_sampling_frequency() const { return sampling_frequency_; }
    
    // Access to individual collectors for fine-grained control
    ProfileCollector* get_profile_collector() { return profile_collector_.get(); }
    OffCPUTimeCollector* get_offcpu_collector() { return offcpu_collector_.get(); }
};

#endif /* __WALLCLOCK_ANALYZER_HPP */ 