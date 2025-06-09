#include "base_analyzer.hpp"
#include "../sampling_printer.hpp"
#include "../collectors/sampling_data.hpp"

BaseAnalyzer::BaseAnalyzer(const std::string& name) 
    : name_(name), symbolizer_(std::make_unique<SamplingPrinter>()) {
}

BaseAnalyzer::~BaseAnalyzer() = default;

std::unique_ptr<FlameGraphView> BaseAnalyzer::sampling_data_to_flamegraph(
    const SamplingData& data, 
    const std::string& analyzer_name,
    bool is_oncpu) {
    
    return ::sampling_data_to_flamegraph(data, analyzer_name, symbolizer_.get(), is_oncpu);
} 