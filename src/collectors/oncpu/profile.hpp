#ifndef __PROFILE_HPP
#define __PROFILE_HPP

#include "collectors/collector_interface.hpp"
#include "profile.skel.h"
#include "collectors/config.hpp"
#include "collectors/utils.hpp"
#include "collectors/sampling_data.hpp"
#include <memory>
#include <vector>
#include <stdexcept>
#include "collectors/bpf_event.h"

#ifdef __cplusplus
extern "C" {
#endif

#include <bpf/libbpf.h>
#include <bpf/bpf.h>
#include <linux/perf_event.h>
#include <asm/unistd.h>
#include <limits.h>

#ifdef __cplusplus
}
#endif

// Forward declarations
struct profile_bpf;
struct bpf_link;
struct blazesym;

// Profile specific configuration
class ProfileConfig : public Config {
public:
    struct perf_event_attr attr;
    bool include_idle;

    ProfileConfig() : Config() {
        // Profile specific defaults
        duration = INT_MAX;
        include_idle = false;
        
        // Initialize perf_event_attr with default values
        attr = {
            .type = PERF_TYPE_SOFTWARE,
            .config = PERF_COUNT_SW_CPU_CLOCK,
            .sample_freq = 50,
            .freq = 1,
        };
        
        // Validate configuration
        validate();
    }

protected:
    void validate() override {
        Config::validate(); // Call base validation
        
        if (user_stacks_only && kernel_stacks_only) {
            throw std::invalid_argument("user_stacks_only and kernel_stacks_only cannot be used together");
        }
        if (attr.sample_freq <= 0) {
            throw std::invalid_argument("sample_freq must be positive");
        }
    }
};

// Use unified data structures from sampling_common.hpp
using ProfileData = SamplingData;
using ProfileEntry = SamplingEntry;

// Custom deleters for RAII
struct ProfileBPFDeleter {
    void operator()(struct profile_bpf* obj) const;
};

struct BPFLinkDeleter {
    void operator()(struct bpf_link* link) const;
};

class ProfileCollector : public ICollector {
private:
    std::vector<std::unique_ptr<struct bpf_link, BPFLinkDeleter>> links;
    std::unique_ptr<struct profile_bpf, ProfileBPFDeleter> obj;
    bool running;
    int nr_cpus;
    ProfileConfig config;
    std::string libbpf_output_buffer_;  // Buffer to capture libbpf debug output
    
public:
    ProfileCollector();
    ~ProfileCollector();  // Custom destructor to clean up registry
    
    std::string get_name() const override;
    bool start() override;
    std::unique_ptr<CollectorData> get_data() override;
    
    // Config management
    ProfileConfig& get_config() { return config; }
    const ProfileConfig& get_config() const { return config; }
    
    // Get captured libbpf output
    const std::string& get_libbpf_output() const { return libbpf_output_buffer_; }
    
    // Internal method to append libbpf output (used by print callback)
    void append_libbpf_output(const std::string& output) { libbpf_output_buffer_ += output; }
    
private:
    int open_and_attach_perf_event(struct bpf_program *prog);
    ProfileData collect_data();
};

#endif /* __PROFILE_HPP */ 