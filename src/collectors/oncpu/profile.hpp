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
    bool freq;
    int sample_freq;
    bool include_idle;

    ProfileConfig() : Config() {
        // Profile specific defaults
        duration = INT_MAX;
        freq = true;
        sample_freq = 50;
        include_idle = false;
        
        // Validate configuration
        validate();
    }

protected:
    void validate() override {
        Config::validate(); // Call base validation
        
        if (user_stacks_only && kernel_stacks_only) {
            throw std::invalid_argument("user_stacks_only and kernel_stacks_only cannot be used together");
        }
        if (sample_freq <= 0) {
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
    
public:
    ProfileCollector();
    ~ProfileCollector() = default;
    
    std::string get_name() const override;
    bool start() override;
    std::unique_ptr<CollectorData> get_data() override;
    
    // Config management
    ProfileConfig& get_config() { return config; }
    const ProfileConfig& get_config() const { return config; }
    
private:
    int open_and_attach_perf_event(struct bpf_program *prog);
    ProfileData collect_data();
};

#endif /* __PROFILE_HPP */ 