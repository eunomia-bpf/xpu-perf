#ifndef __PROFILE_ANALYZER_HPP
#define __PROFILE_ANALYZER_HPP

#include "base_analyzer.hpp"
#include "collectors/oncpu/profile.hpp"
#include <memory>

// Forward declaration
class ProfileCollector;

class ProfileAnalyzer : public BaseAnalyzer {
private:
    std::unique_ptr<ProfileCollector> collector_;

public:
    ProfileAnalyzer();
    virtual ~ProfileAnalyzer() = default;
    
    // IAnalyzer interface
    bool start() override;
    std::unique_ptr<FlameGraphView> get_flamegraph() override;
    
    // Config access
    ProfileConfig& get_config() { return collector_->get_config(); }
    const ProfileConfig& get_config() const { return collector_->get_config(); }
};

// Implementation
inline ProfileAnalyzer::ProfileAnalyzer() 
    : BaseAnalyzer("profile_analyzer"), collector_(std::make_unique<ProfileCollector>()) {
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
    
    return sampling_data_to_flamegraph(*sampling_data, get_name(), true);
}

#endif /* __PROFILE_ANALYZER_HPP */ 