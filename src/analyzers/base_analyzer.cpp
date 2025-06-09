#include "base_analyzer.hpp"
#include "../collectors/sampling_data.hpp"

BaseAnalyzer::BaseAnalyzer(const std::string& name) 
    : name_(name) {
}

BaseAnalyzer::~BaseAnalyzer() = default;

std::unique_ptr<FlameGraphView> BaseAnalyzer::sampling_data_to_flamegraph(
    const SamplingData& data, 
    const std::string& analyzer_name,
    bool is_oncpu) {
    
    auto flamegraph = std::make_unique<FlameGraphView>(analyzer_name, true);
    
    // Set appropriate time unit
    flamegraph->time_unit = is_oncpu ? "samples" : "microseconds";
    
    for (const auto& entry : data.entries) {
        std::vector<std::string> user_stack_symbols;
        std::vector<std::string> kernel_stack_symbols;
        
        // Placeholder symbol resolution (addresses as strings for now)
        if (entry.has_user_stack && !entry.user_stack.empty()) {
            for (size_t i = 0; i < entry.user_stack.size(); i++) {
                user_stack_symbols.push_back("func_" + std::to_string(entry.user_stack[i]));
            }
        }
        
        if (entry.has_kernel_stack && !entry.kernel_stack.empty()) {
            for (size_t i = 0; i < entry.kernel_stack.size(); i++) {
                kernel_stack_symbols.push_back("kernel_" + std::to_string(entry.kernel_stack[i]));
            }
        }
        
        // Add to flamegraph with appropriate annotations
        if (is_oncpu) {
            // Add [c] annotation for CPU-intensive stacks
            for (auto& symbol : user_stack_symbols) {
                symbol += "_[c]";
            }
            for (auto& symbol : kernel_stack_symbols) {
                symbol += "_[c]";
            }
        } else {
            // Add [o] annotation for off-CPU stacks
            for (auto& symbol : user_stack_symbols) {
                symbol += "_[o]";
            }
            for (auto& symbol : kernel_stack_symbols) {
                symbol += "_[o]";
            }
        }
        
        flamegraph->add_stack_trace(
            user_stack_symbols,
            kernel_stack_symbols,
            std::string(entry.key.comm),
            entry.key.pid,
            entry.value,
            is_oncpu
        );
    }
    
    flamegraph->finalize();
    return flamegraph;
} 