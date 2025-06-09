#ifndef __PROFILE_HPP
#define __PROFILE_HPP

#include "collector_interface.hpp"
#include "profile.h"
#include "arg_parse.h"

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

class ProfileCollector : public ICollector {
private:
    struct bpf_link *links[MAX_CPU_NR];
    struct profile_bpf *obj;
    bool running;
    int nr_cpus;
    struct blazesym *symbolizer;
    
public:
    ProfileCollector();
    ~ProfileCollector();
    
    std::string get_name() const override;
    bool start() override;
    CollectorData get_data() override;
    
private:
    int open_and_attach_perf_event(struct bpf_program *prog, struct bpf_link *links[]);
};

#endif /* __PROFILE_HPP */ 