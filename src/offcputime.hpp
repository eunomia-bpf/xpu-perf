#ifndef __OFFCPUTIME_HPP
#define __OFFCPUTIME_HPP

#include "collector_interface.hpp"
#include "bpf_event.h"
#include "utils.hpp"
#include "sampling_common.hpp"
#include "config.hpp"
#include <memory>
#include <vector>

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
    std::unique_ptr<struct blazesym, BlazesymDeleter> symbolizer;
    OffCPUTimeConfig config;
    
public:
    OffCPUTimeCollector();
    ~OffCPUTimeCollector() = default;
    
    std::string get_name() const override;
    bool start() override;
    CollectorData get_data() override;
    
    // Config management
    OffCPUTimeConfig& get_config() { return config; }
    const OffCPUTimeConfig& get_config() const { return config; }
    
private:
    bool probe_tp_btf(const char *name);
    OffCPUData collect_data();
    std::string format_data(const OffCPUData& data);
    void print_data(const OffCPUData& data);
};

#endif /* __OFFCPUTIME_HPP */ 