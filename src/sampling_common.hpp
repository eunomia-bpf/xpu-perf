#ifndef __SAMPLING_COMMON_HPP
#define __SAMPLING_COMMON_HPP

#include "utils.hpp"
#include "config.hpp"
#include "bpf_event.h"
#include <memory>
#include <vector>
#include <sstream>

#ifdef __cplusplus
extern "C" {
#endif

#include <bpf/libbpf.h>
#include <bpf/bpf.h>

#ifdef __cplusplus
}
#endif

// Forward declarations
struct blazesym;

// Unified data structure for collected sampling data (works for both profile and offcpu)
struct SamplingEntry {
    struct sample_key_t key;
    __u64 value;  // Can be count (for profile) or delta time (for offcpu)
    std::vector<unsigned long> user_stack;
    std::vector<unsigned long> kernel_stack;
    bool has_user_stack;
    bool has_kernel_stack;
};

struct SamplingData {
    std::vector<SamplingEntry> entries;
};

// Common print functions for sampling data
class SamplingPrinter {
public:
    static void print_data(const SamplingData& data, struct blazesym* symbolizer, const Config& config, const std::string& value_label = "");
    static std::string format_data(const SamplingData& data, const std::string& tool_name);
    
private:
    static void print_entry_multiline(const SamplingEntry& entry, struct blazesym* symbolizer, const Config& config, const std::string& value_label);
    static void print_entry_folded(const SamplingEntry& entry, struct blazesym* symbolizer, const Config& config);
};

// Implementation
inline void SamplingPrinter::print_data(const SamplingData& data, struct blazesym* symbolizer, const Config& config, const std::string& value_label) {
    if (!symbolizer) {
        return;
    }
    
    for (size_t i = 0; i < data.entries.size(); i++) {
        const auto& entry = data.entries[i];
        
        if (config.folded) {
            print_entry_folded(entry, symbolizer, config);
        } else {
            print_entry_multiline(entry, symbolizer, config, value_label);
            
            // Add a newline between stack traces for better readability
            if (i < data.entries.size() - 1) {
                printf("\n");
            }
        }
    }
}

inline std::string SamplingPrinter::format_data(const SamplingData& data, const std::string& tool_name) {
    std::ostringstream oss;
    oss << "Collected " << data.entries.size() << " " << tool_name << " entries";
    return oss.str();
}

inline void SamplingPrinter::print_entry_multiline(const SamplingEntry& entry, struct blazesym* symbolizer, const Config& config, const std::string& value_label) {
    /* multi-line stack output */
    /* Show kernel stack first */
    if (!config.user_stacks_only && entry.has_kernel_stack) {
        if (entry.kernel_stack.empty()) {
            fprintf(stderr, "    [Missed Kernel Stack]\n");
        } else {
            show_stack_trace(symbolizer, 
                const_cast<__u64 *>(reinterpret_cast<const __u64 *>(entry.kernel_stack.data())), 
                config.perf_max_stack_depth, 0);
        }
    }

    if (config.delimiter && !config.user_stacks_only && !config.kernel_stacks_only &&
        entry.has_user_stack && entry.has_kernel_stack) {
        printf("    --\n");
    }

    /* Then show user stack */
    if (!config.kernel_stacks_only && entry.has_user_stack) {
        if (entry.user_stack.empty()) {
            fprintf(stderr, "    [Missed User Stack]\n");
        } else {
            show_stack_trace(symbolizer, 
                const_cast<__u64 *>(reinterpret_cast<const __u64 *>(entry.user_stack.data())), 
                config.perf_max_stack_depth, entry.key.pid);
        }
    }

    printf("    %-16s %s (%d)\n", "-", entry.key.comm, entry.key.pid);
    if (!value_label.empty()) {
        printf("        %lld %s\n", entry.value, value_label.c_str());
    } else {
        printf("        %lld\n", entry.value);
    }
}

inline void SamplingPrinter::print_entry_folded(const SamplingEntry& entry, struct blazesym* symbolizer, const Config& config) {
    /* folded stack output */
    printf("%s", entry.key.comm);
    
    /* Print user stack first for folded format */
    if (entry.has_user_stack && !config.kernel_stacks_only) {
        if (entry.user_stack.empty()) {
            printf(";[Missed User Stack]");
        } else {
            printf(";");
            show_stack_trace_folded(symbolizer, 
                const_cast<__u64 *>(reinterpret_cast<const __u64 *>(entry.user_stack.data())), 
                config.perf_max_stack_depth, entry.key.pid, ';', true);
        }
    }
    
    /* Then print kernel stack if it exists */
    if (entry.has_kernel_stack && !config.user_stacks_only) {
        /* Add delimiter between user and kernel stacks if needed */
        if (entry.has_user_stack && config.delimiter && !config.kernel_stacks_only)
            printf("-");
            
        if (entry.kernel_stack.empty()) {
            printf(";[Missed Kernel Stack]");
        } else {
            printf(";");
            show_stack_trace_folded(symbolizer, 
                const_cast<__u64 *>(reinterpret_cast<const __u64 *>(entry.kernel_stack.data())), 
                config.perf_max_stack_depth, 0, ';', true);
        }
    }
    
    printf(" %lld\n", entry.value);
}

#endif /* __SAMPLING_COMMON_HPP */ 