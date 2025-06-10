#ifndef __OFFCPUTIME_HPP
#define __OFFCPUTIME_HPP

#include "collectors/collector_interface.hpp"
#include "collectors/bpf_event.h"
#include "collectors/utils.hpp"
#include "collectors/sampling_data.hpp"
#include "collectors/config.hpp"
#include <memory>
#include <vector>
#include <stdexcept>

#ifdef __cplusplus
extern "C" {
#endif

#include <bpf/libbpf.h>
#include <bpf/bpf.h>

#ifdef __cplusplus
}
#endif

// Forward declarations
struct offcputime_bpf;
struct blazesym;

// OffCPU Time specific configuration
class OffCPUTimeConfig : public Config {
public:
    __u64 min_block_time;
    __u64 max_block_time;
    long state;

    OffCPUTimeConfig() : Config() {
        // OffCPU specific defaults
        min_block_time = 1;
        max_block_time = (__u64)(-1);
        state = -1;
        
        // Validate configuration
        validate();
    }


protected:
    void validate() override {
        Config::validate(); // Call base validation
        
        if (user_threads_only && kernel_threads_only) {
            throw std::invalid_argument("user_threads_only and kernel_threads_only cannot be used together");
        }
        if (min_block_time >= max_block_time) {
            throw std::invalid_argument("min_block_time should be smaller than max_block_time");
        }
        if (state < -1 || state > 2) {
            throw std::invalid_argument("state must be between -1 and 2");
        }
    }
};

// Use unified data structures from sampling_common.hpp
using OffCPUData = SamplingData;
using OffCPUEntry = SamplingEntry;

// Custom deleters for RAII
struct OffCPUBPFDeleter {
    void operator()(struct offcputime_bpf* obj) const;
};

class OffCPUTimeCollector : public ICollector {
private:
    std::unique_ptr<struct offcputime_bpf, OffCPUBPFDeleter> obj;
    bool running;
    OffCPUTimeConfig config;
    std::string libbpf_output_buffer_;  // Buffer to capture libbpf debug output
    
public:
    OffCPUTimeCollector();
    ~OffCPUTimeCollector();  // Custom destructor to clean up registry
    
    std::string get_name() const override;
    bool start() override;
    std::unique_ptr<CollectorData> get_data() override;
    
    // Config management
    OffCPUTimeConfig& get_config() { return config; }
    const OffCPUTimeConfig& get_config() const { return config; }
    
    // Get captured libbpf output
    const std::string& get_libbpf_output() const { return libbpf_output_buffer_; }
    
    // Internal method to append libbpf output (used by print callback)
    void append_libbpf_output(const std::string& output) { libbpf_output_buffer_ += output; }
    
private:
    bool probe_tp_btf(const char *name);
    OffCPUData collect_data();
};

#endif /* __OFFCPUTIME_HPP */ 