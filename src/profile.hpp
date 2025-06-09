#ifndef __PROFILE_HPP
#define __PROFILE_HPP

#include "collector_interface.hpp"
#include "profile.h"
#include "arg_parse.h"
#include <memory>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

#include <bpf/libbpf.h>
#include <bpf/bpf.h>
#include <linux/perf_event.h>
#include <asm/unistd.h>

#ifdef __cplusplus
}
#endif

// Forward declarations
struct profile_bpf;
struct bpf_link;
struct blazesym;

// Data structure for collected profile data
struct ProfileEntry {
    struct profile_key_t key;
    __u64 count;
    std::vector<unsigned long> user_stack;
    std::vector<unsigned long> kernel_stack;
    bool has_user_stack;
    bool has_kernel_stack;
};

struct ProfileData {
    std::vector<ProfileEntry> entries;
};

// Custom deleters for RAII
struct ProfileBPFDeleter {
    void operator()(struct profile_bpf* obj) const;
};

struct BlazesymDeleter {
    void operator()(struct blazesym* sym) const;
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
    std::unique_ptr<struct blazesym, BlazesymDeleter> symbolizer;
    
public:
    ProfileCollector();
    ~ProfileCollector() = default;
    
    std::string get_name() const override;
    bool start() override;
    CollectorData get_data() override;
    
private:
    int open_and_attach_perf_event(struct bpf_program *prog);
    ProfileData collect_data();
    std::string format_data(const ProfileData& data);
    void print_data(const ProfileData& data);
};

#endif /* __PROFILE_HPP */ 