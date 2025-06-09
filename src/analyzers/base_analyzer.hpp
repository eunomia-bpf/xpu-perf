#ifndef __BASE_ANALYZER_HPP
#define __BASE_ANALYZER_HPP

#include "flamegraph_view.hpp"
#include "collectors/collector_interface.hpp"
#include "symbol_resolver.hpp"
#include <memory>
#include <vector>
#include <string>
#include <map>

// Forward declarations
class SamplingData;

// Base analyzer interface
class IAnalyzer {
public:
    virtual ~IAnalyzer() = default;
    
    // Start the analysis (may involve multiple collectors)
    virtual bool start() = 0;
    
    // Get analyzed flamegraph data
    virtual std::unique_ptr<FlameGraphView> get_flamegraph() = 0;
    
    // Get per-thread flamegraph data (for multi-threaded analysis)
    virtual std::map<pid_t, std::unique_ptr<FlameGraphView>> get_per_thread_flamegraphs() {
        // Default implementation: return single flamegraph for main thread
        std::map<pid_t, std::unique_ptr<FlameGraphView>> result;
        result[0] = get_flamegraph();
        return result;
    }
    
    // Get analyzer name
    virtual std::string get_name() const = 0;
};

// Base analyzer implementation with common functionality
class BaseAnalyzer : public IAnalyzer {
protected:
    std::string name_;

public:
    BaseAnalyzer(const std::string& name) : name_(name) {}
    virtual ~BaseAnalyzer() = default;
    
    std::string get_name() const override { return name_; }
};

#endif /* __BASE_ANALYZER_HPP */ 