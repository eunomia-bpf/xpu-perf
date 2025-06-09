#ifndef __OFFCPUTIME_HPP
#define __OFFCPUTIME_HPP

#include "collector_interface.hpp"
#include "offcputime.h"
#include "arg_parse.h"
#include "utils.hpp"
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

// Data structure for collected off-CPU data
struct OffCPUEntry {
    struct offcpu_key_t key;
    struct offcpu_val_t val;
    std::vector<unsigned long> user_stack;
    std::vector<unsigned long> kernel_stack;
    bool has_user_stack;
    bool has_kernel_stack;
};

struct OffCPUData {
    std::vector<OffCPUEntry> entries;
};

// Custom deleters for RAII
struct OffCPUBPFDeleter {
    void operator()(struct offcputime_bpf* obj) const;
};

class OffCPUTimeCollector : public ICollector {
private:
    std::unique_ptr<struct offcputime_bpf, OffCPUBPFDeleter> obj;
    bool running;
    std::unique_ptr<struct blazesym, BlazesymDeleter> symbolizer;
    
public:
    OffCPUTimeCollector();
    ~OffCPUTimeCollector() = default;
    
    std::string get_name() const override;
    bool start() override;
    CollectorData get_data() override;
    
private:
    bool probe_tp_btf(const char *name);
    OffCPUData collect_data();
    std::string format_data(const OffCPUData& data);
    void print_data(const OffCPUData& data);
};

#endif /* __OFFCPUTIME_HPP */ 