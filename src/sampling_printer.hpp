#ifndef __SAMPLING_PRINTER_HPP
#define __SAMPLING_PRINTER_HPP

#include "collectors/utils.hpp"
#include "collectors/config.hpp"
#include "collectors/bpf_event.h"
#include <memory>
#include <vector>
#include <sstream>
#include "collectors/sampling_data.hpp"

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
// Common print functions for sampling data
class SamplingPrinter {
public:
    static inline void print_data(const SamplingData& data, struct blazesym* symbolizer, const Config& config, const std::string& value_label = "");
    static inline std::string format_data(const SamplingData& data, const std::string& tool_name);
    
private:
    static inline void print_entry_multiline(const SamplingEntry& entry, struct blazesym* symbolizer, const Config& config, const std::string& value_label);
    static inline void print_entry_folded(const SamplingEntry& entry, struct blazesym* symbolizer, const Config& config);
};


/**
 * show_stack_trace - Display a stack trace with symbol resolution
 * @symbolizer: Blazesym symbolizer instance
 * @stack: Array of stack addresses
 * @stack_sz: Size of the stack array
 * @pid: Process ID (0 for kernel)
 */
inline void show_stack_trace(struct blazesym *symbolizer, __u64 *stack, int stack_sz, pid_t pid)
{
    const struct blazesym_result *result;
    const struct blazesym_csym *sym;
    struct sym_src_cfg src;
    int i, j;

    memset(&src, 0, sizeof(src));
    if (pid) {
        src.src_type = SRC_T_PROCESS;
        src.params.process.pid = pid;
    } else {
        src.src_type = SRC_T_KERNEL;
        src.params.kernel.kallsyms = NULL;
        src.params.kernel.kernel_image = NULL;
    }

    result = blazesym_symbolize(symbolizer, &src, 1, (const uint64_t *)stack, stack_sz);

    for (i = 0; i < stack_sz; i++) {
        if (!stack[i])
            continue;

        if (!result || result->size <= i || !result->entries[i].size) {
            printf("    [unknown]\n");
            continue;
        }

        if (result->entries[i].size == 1) {
            sym = &result->entries[i].syms[0];
            printf("    %s\n", sym->symbol);
            continue;
        }

        for (j = 0; j < result->entries[i].size; j++) {
            sym = &result->entries[i].syms[j];
            printf("    %s\n", sym->symbol);
        }
    }

    blazesym_result_free(result);
}

/**
 * show_stack_trace_folded - Display a stack trace in folded format for flamegraphs
 * @symbolizer: Blazesym symbolizer instance
 * @stack: Array of stack addresses
 * @stack_sz: Size of the stack array
 * @pid: Process ID (0 for kernel)
 * @separator: Character to use as separator between frames (typically ';')
 * @reverse: Whether to print the stack in reverse order (true for flamegraphs)
 */
inline void show_stack_trace_folded(struct blazesym *symbolizer, __u64 *stack, int stack_sz, 
                            pid_t pid, char separator, bool reverse)
{
    const struct blazesym_result *result;
    const struct blazesym_csym *sym;
    struct sym_src_cfg src;
    int i;
    bool first = true;

    memset(&src, 0, sizeof(src));
    if (pid) {
        src.src_type = SRC_T_PROCESS;
        src.params.process.pid = pid;
    } else {
        src.src_type = SRC_T_KERNEL;
        src.params.kernel.kallsyms = NULL;
        src.params.kernel.kernel_image = NULL;
    }

    result = blazesym_symbolize(symbolizer, &src, 1, (const uint64_t *)stack, stack_sz);

    /* For flamegraphs, we need to print the stack in reverse order */
    if (reverse) {
        for (i = stack_sz - 1; i >= 0; i--) {
            if (!stack[i])
                continue;

            if (!result || result->size <= i || !result->entries[i].size) {
                if (!first) {
                    printf("%c", separator);
                }
                printf("[unknown]");
                first = false;
                continue;
            }

            if (result->entries[i].size >= 1) {
                sym = &result->entries[i].syms[0];
                if (!first) {
                    printf("%c", separator);
                }
                printf("%s", sym->symbol);
                first = false;
            }
        }
    } else {
        /* Print stack in normal order */
        for (i = 0; i < stack_sz; i++) {
            if (!stack[i])
                continue;

            if (!result || result->size <= i || !result->entries[i].size) {
                if (!first) {
                    printf("%c", separator);
                }
                printf("[unknown]");
                first = false;
                continue;
            }

            if (result->entries[i].size >= 1) {
                sym = &result->entries[i].syms[0];
                if (!first) {
                    printf("%c", separator);
                }
                printf("%s", sym->symbol);
                first = false;
            }
        }
    }

    blazesym_result_free(result);
}

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

#endif /* __SAMPLING_PRINTER_HPP */ 