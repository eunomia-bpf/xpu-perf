#ifndef __OFFCPUTIME_HPP
#define __OFFCPUTIME_HPP

#include "collector_interface.hpp"
#include "offcputime.h"
#include "arg_parse.h"

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

class OffCPUTimeCollector : public ICollector {
private:
    struct offcputime_bpf *obj;
    bool running;
    struct blazesym *symbolizer;
    
public:
    OffCPUTimeCollector();
    ~OffCPUTimeCollector();
    
    std::string get_name() const override;
    bool start() override;
    CollectorData get_data() override;
    
private:
    bool probe_tp_btf(const char *name);
};

#endif /* __OFFCPUTIME_HPP */ 