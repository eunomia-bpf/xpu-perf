#ifndef __PROFILE_ANALYZER_HPP
#define __PROFILE_ANALYZER_HPP

#include "flamegraph_view.hpp"
#include "base_analyzer.hpp"
#include "analyzer_config.hpp"
#include "collectors/oncpu/profile.hpp"
#include <memory>
#include <cstring>
#include <algorithm>

// Forward declaration
class ProfileCollector;

class ProfileAnalyzer : public BaseAnalyzer {
private:
    std::unique_ptr<ProfileCollector> collector_;
    std::unique_ptr<ProfileAnalyzerConfig> config_;

public:
    explicit ProfileAnalyzer(std::unique_ptr<ProfileAnalyzerConfig> config);
    virtual ~ProfileAnalyzer() = default;
    
    // IAnalyzer interface
    bool start() override;
    std::unique_ptr<FlameGraphView> get_flamegraph();
    
    // Config access
    const ProfileAnalyzerConfig& get_config() const { return *config_; }

private:
};

// Implementation
inline ProfileAnalyzer::ProfileAnalyzer(std::unique_ptr<ProfileAnalyzerConfig> config) 
    : BaseAnalyzer("profile_analyzer"), 
      collector_(std::make_unique<ProfileCollector>()),
      config_(std::move(config)) {
    
    // Configure collector directly in constructor
    if (collector_ && config_) {
        auto& collector_config = collector_->get_config();
        
        // Apply essential settings
        collector_config.duration = config_->duration;
        collector_config.attr.sample_freq = config_->frequency;
        
        // Copy PIDs and TIDs directly to vectors
        collector_config.pids = config_->pids;
        collector_config.tids = config_->tids;
    }
}

inline bool ProfileAnalyzer::start() {
    if (!collector_) {
        return false;
    }
    return collector_->start();
}

inline std::unique_ptr<FlameGraphView> ProfileAnalyzer::get_flamegraph() {
    if (!collector_) {
        return std::make_unique<FlameGraphView>(get_name(), false);
    }
    
    auto data = collector_->get_data();
    if (!data || !data->success) {
        return std::make_unique<FlameGraphView>(get_name(), false);
    }
    
    auto* sampling_data = dynamic_cast<SamplingData*>(data.get());
    if (!sampling_data) {
        return std::make_unique<FlameGraphView>(get_name(), false);
    }
    
    auto flamegraph = FlameGraphView::sampling_data_to_flamegraph(*sampling_data, get_name(), true);
    return flamegraph;
}

#endif /* __PROFILE_ANALYZER_HPP */ 