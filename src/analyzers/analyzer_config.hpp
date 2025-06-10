#ifndef __ANALYZER_CONFIG_HPP
#define __ANALYZER_CONFIG_HPP

#include <vector>
#include <sys/types.h>

// Forward declare Linux types to avoid pulling in kernel headers
typedef unsigned long long __u64;

// Base configuration for all analyzers - keep it simple
class BaseAnalyzerConfig {
public:
    int duration;
    std::vector<pid_t> pids;
    std::vector<pid_t> tids;
    
    BaseAnalyzerConfig() : duration(99999999) {}
    virtual ~BaseAnalyzerConfig() = default;
};

// Profile analyzer specific configuration
class ProfileAnalyzerConfig : public BaseAnalyzerConfig {
public:
    int frequency;
    
    ProfileAnalyzerConfig() : BaseAnalyzerConfig(), frequency(49) {}
};

// OffCPU analyzer specific configuration  
class OffCPUAnalyzerConfig : public BaseAnalyzerConfig {
public:
    __u64 min_block_us;
    
    OffCPUAnalyzerConfig() : BaseAnalyzerConfig(), min_block_us(1000) {}
};

// WallClock analyzer specific configuration
class WallClockAnalyzerConfig : public BaseAnalyzerConfig {
public:
    int frequency;
    __u64 min_block_us;
    
    WallClockAnalyzerConfig() : BaseAnalyzerConfig(), frequency(49), min_block_us(1000) {}
};

#endif /* __ANALYZER_CONFIG_HPP */ 