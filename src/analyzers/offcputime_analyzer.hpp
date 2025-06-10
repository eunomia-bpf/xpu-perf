#ifndef __OFFCPUTIME_ANALYZER_HPP
#define __OFFCPUTIME_ANALYZER_HPP

#include "flamegraph_view.hpp"
#include "base_analyzer.hpp"
#include "analyzer_config.hpp"
#include "collectors/offcpu/offcputime.hpp"
#include "collectors/utils.hpp"
#include <memory>
#include <cstring>
#include <algorithm>

// Forward declaration
class OffCPUTimeCollector;

class OffCPUTimeAnalyzer : public BaseAnalyzer {
private:
    std::unique_ptr<OffCPUTimeCollector> collector_;
    std::unique_ptr<OffCPUAnalyzerConfig> config_;

public:
    explicit OffCPUTimeAnalyzer(std::unique_ptr<OffCPUAnalyzerConfig> config);
    virtual ~OffCPUTimeAnalyzer() = default;
    
    // IAnalyzer interface
    bool start() override;
    std::unique_ptr<FlameGraphView> get_flamegraph() override;
    
    // Config access
    const OffCPUAnalyzerConfig& get_config() const { return *config_; }

private:
};

// Implementation
inline OffCPUTimeAnalyzer::OffCPUTimeAnalyzer(std::unique_ptr<OffCPUAnalyzerConfig> config) 
    : BaseAnalyzer("offcputime_analyzer"), 
      collector_(std::make_unique<OffCPUTimeCollector>()),
      config_(std::move(config)) {
    
    // Configure collector directly in constructor
    if (collector_ && config_) {
        auto& collector_config = collector_->get_config();
        
        // Apply essential settings
        collector_config.duration = config_->duration;
        collector_config.min_block_time = config_->min_block_us;
        
        // Copy PIDs and TIDs directly to vectors
        collector_config.pids = config_->pids;
        collector_config.tids = config_->tids;
    }
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