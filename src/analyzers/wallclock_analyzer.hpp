#ifndef __WALLCLOCK_ANALYZER_HPP
#define __WALLCLOCK_ANALYZER_HPP

#include "base_analyzer.hpp"
#include "analyzer_config.hpp"
#include "collectors/oncpu/profile.hpp"
#include "collectors/offcpu/offcputime.hpp"
#include "collectors/utils.hpp"
#include <memory>
#include <map>
#include <cstring>
#include <algorithm>

// Forward declarations
class ProfileCollector;
class OffCPUTimeCollector;

class WallClockAnalyzer : public BaseAnalyzer {
private:
    std::unique_ptr<ProfileCollector> profile_collector_;
    std::unique_ptr<OffCPUTimeCollector> offcpu_collector_;
    std::unique_ptr<WallClockAnalyzerConfig> config_;
    
    // Helper methods
    std::map<pid_t, std::unique_ptr<FlameGraphView>> combine_and_resolve_data();

public:
    explicit WallClockAnalyzer(std::unique_ptr<WallClockAnalyzerConfig> config);
    virtual ~WallClockAnalyzer() = default;
    
    // IAnalyzer interface
    bool start() override;
    std::unique_ptr<FlameGraphView> get_flamegraph() override;
    std::map<pid_t, std::unique_ptr<FlameGraphView>> get_per_thread_flamegraphs() override;
    
    // Config access
    const WallClockAnalyzerConfig& get_config() const { return *config_; }
    
    // Access to individual collectors for fine-grained control (if needed)
    ProfileCollector* get_profile_collector() { return profile_collector_.get(); }
    OffCPUTimeCollector* get_offcpu_collector() { return offcpu_collector_.get(); }

private:
    void configure_collectors();
};

#endif /* __WALLCLOCK_ANALYZER_HPP */ 