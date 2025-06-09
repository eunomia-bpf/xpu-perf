#ifndef __OFFCPUTIME_ANALYZER_HPP
#define __OFFCPUTIME_ANALYZER_HPP

#include "analyzers/flamegraph_view.hpp"
#include "base_analyzer.hpp"
#include "collectors/offcpu/offcputime.hpp"
#include <memory>

// Forward declaration
class OffCPUTimeCollector;

class OffCPUTimeAnalyzer : public BaseAnalyzer {
private:
    std::unique_ptr<OffCPUTimeCollector> collector_;

public:
    OffCPUTimeAnalyzer();
    virtual ~OffCPUTimeAnalyzer() = default;
    
    // IAnalyzer interface
    bool start() override;
    std::unique_ptr<FlameGraphView> get_flamegraph() override;
    
    // Config access
    OffCPUTimeConfig& get_config() { return collector_->get_config(); }
    const OffCPUTimeConfig& get_config() const { return collector_->get_config(); }
};

// Implementation
inline OffCPUTimeAnalyzer::OffCPUTimeAnalyzer() 
    : BaseAnalyzer("offcputime_analyzer"), collector_(std::make_unique<OffCPUTimeCollector>()) {
}

inline bool OffCPUTimeAnalyzer::start() {
    if (!collector_) {
        return false;
    }
    return collector_->start();
}

inline std::unique_ptr<FlameGraphView> OffCPUTimeAnalyzer::get_flamegraph() {
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
    
    return FlameGraphView::sampling_data_to_flamegraph(*sampling_data, get_name(), false);
}

#endif /* __OFFCPUTIME_ANALYZER_HPP */ 