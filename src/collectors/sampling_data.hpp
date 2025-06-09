#ifndef __SAMPLING_COMMON_HPP
#define __SAMPLING_COMMON_HPP

#include "bpf_event.h"
#include "collector_interface.hpp"
#include <vector>
#include <cstdint>
#include <linux/types.h>

// Forward declaration
class SamplingPrinter;

// Unified data structure for collected sampling data (works for both profile and offcpu)
struct SamplingEntry {
    struct sample_key_t key;
    __u64 value;  // Can be count (for profile) or delta time (for offcpu)
    std::vector<unsigned long> user_stack;
    std::vector<unsigned long> kernel_stack;
    bool has_user_stack;
    bool has_kernel_stack;
};

class SamplingData : public CollectorData {
public:
    std::vector<SamplingEntry> entries;
    
    SamplingData(const std::string& collector_name, bool success = true) 
        : name(collector_name), success(success) {}
    
    // Implement CollectorData interface
    std::string get_name() const override { return name; }
    bool is_success() const override { return success; }
    std::string get_type() const override { return "sampling"; }
    
private:
    std::string name;
    bool success;
};

#endif /* __SAMPLING_COMMON_HPP */ 