#include "symbol_resolver.hpp"
#include <cstring>
#include <algorithm>

#ifdef __cplusplus
extern "C" {
#endif

#include "blazesym.h"

#ifdef __cplusplus
}
#endif

struct BlazesymDeleter {
    void operator()(struct blazesym* sym) {
        if (sym) {
            blazesym_free(sym);
        }
    }
};

SymbolResolver::SymbolResolver() 
    : symbolizer_(blazesym_new()) {
    if (!symbolizer_) {
        // Log error but don't throw - let is_valid() handle this
    }
}

std::vector<std::string> SymbolResolver::get_stack_trace_symbols(__u64 *stack, int stack_sz, pid_t pid) {
    std::vector<std::string> symbols;
    
    if (!symbolizer_ || !stack || stack_sz <= 0) {
        return symbols;
    }
    
    const struct blazesym_result *result;
    const struct blazesym_csym *sym;
    struct sym_src_cfg src;
    
    memset(&src, 0, sizeof(src));
    if (pid) {
        src.src_type = SRC_T_PROCESS;
        src.params.process.pid = pid;
    } else {
        src.src_type = SRC_T_KERNEL;
        src.params.kernel.kallsyms = NULL;
        src.params.kernel.kernel_image = NULL;
    }
    
    result = blazesym_symbolize(symbolizer_.get(), &src, 1, (const uint64_t *)stack, stack_sz);
    
    for (int i = 0; i < stack_sz; i++) {
        if (!stack[i])
            continue;
            
        if (!result || result->size <= i || !result->entries[i].size) {
            symbols.push_back("[unknown]");
            continue;
        }
        
        if (result->entries[i].size == 1) {
            sym = &result->entries[i].syms[0];
            symbols.push_back(std::string(sym->symbol));
            continue;
        }
        
        for (int j = 0; j < result->entries[i].size; j++) {
            sym = &result->entries[i].syms[j];
            symbols.push_back(std::string(sym->symbol));
        }
    }
    
    if (result) {
        blazesym_result_free(result);
    }
    
    return symbols;
}

std::string SymbolResolver::stack_trace_to_string(const std::vector<std::string>& symbols) {
    std::ostringstream oss;
    for (const auto& symbol : symbols) {
        oss << "    " << symbol << "\n";
    }
    return oss.str();
}

std::string SymbolResolver::stack_trace_to_folded_string(const std::vector<std::string>& symbols, 
                                                       char separator, bool reverse) {
    if (symbols.empty()) {
        return "";
    }
    
    std::ostringstream oss;
    bool first = true;
    
    if (reverse) {
        // For flamegraphs, print stack in reverse order (bottom-up)
        for (auto it = symbols.rbegin(); it != symbols.rend(); ++it) {
            if (!first) {
                oss << separator;
            }
            oss << *it;
            first = false;
        }
    } else {
        // Print stack in normal order (top-down)
        for (const auto& symbol : symbols) {
            if (!first) {
                oss << separator;
            }
            oss << symbol;
            first = false;
        }
    }
    
    return oss.str();
}

std::string SymbolResolver::print_data(const SamplingData& data, const Config& config, const std::string& value_label) {
    std::ostringstream oss;
    
    if (!symbolizer_) {
        oss << "Error: Symbolizer not initialized\n";
        return oss.str();
    }
    
    for (size_t i = 0; i < data.entries.size(); i++) {
        const auto& entry = data.entries[i];
        
        if (config.folded) {
            oss << print_entry_folded(entry, config);
        } else {
            oss << print_entry_multiline(entry, config, value_label);
            
            // Add a newline between stack traces for better readability
            if (i < data.entries.size() - 1) {
                oss << "\n";
            }
        }
    }
    
    return oss.str();
}

std::string SymbolResolver::print_entry_multiline(const SamplingEntry& entry, const Config& config, const std::string& value_label) {
    std::ostringstream oss;
    
    // Show kernel stack first
    if (!config.user_stacks_only && entry.has_kernel_stack) {
        if (entry.kernel_stack.empty()) {
            oss << "    [Missed Kernel Stack]\n";
        } else {
            auto symbols = get_stack_trace_symbols(
                const_cast<__u64 *>(reinterpret_cast<const __u64 *>(entry.kernel_stack.data())), 
                std::min(static_cast<int>(entry.kernel_stack.size()), config.perf_max_stack_depth), 
                0);
            oss << this->stack_trace_to_string(symbols);
        }
    }
    
    // Delimiter between kernel and user stacks
    if (config.delimiter && !config.user_stacks_only && !config.kernel_stacks_only &&
        entry.has_user_stack && entry.has_kernel_stack) {
        oss << "    --\n";
    }
    
    // Show user stack
    if (!config.kernel_stacks_only && entry.has_user_stack) {
        if (entry.user_stack.empty()) {
            oss << "    [Missed User Stack]\n";
        } else {
            auto symbols = get_stack_trace_symbols(
                const_cast<__u64 *>(reinterpret_cast<const __u64 *>(entry.user_stack.data())), 
                std::min(static_cast<int>(entry.user_stack.size()), config.perf_max_stack_depth), 
                entry.key.pid);
            oss << this->stack_trace_to_string(symbols);
        }
    }
    
    oss << "    " << entry.key.comm << " (" << entry.key.pid << ")\n";
    if (!value_label.empty()) {
        oss << "        " << entry.value << " " << value_label << "\n";
    } else {
        oss << "        " << entry.value << "\n";
    }
    
    return oss.str();
}

std::string SymbolResolver::print_entry_folded(const SamplingEntry& entry, const Config& config) {
    std::ostringstream oss;
    
    // Start with the command name
    oss << entry.key.comm;
    
    // Print user stack first for folded format (this is the standard for flamegraphs)
    if (entry.has_user_stack && !config.kernel_stacks_only) {
        if (entry.user_stack.empty()) {
            oss << ";[Missed User Stack]";
        } else {
            auto symbols = get_stack_trace_symbols(
                const_cast<__u64 *>(reinterpret_cast<const __u64 *>(entry.user_stack.data())), 
                std::min(static_cast<int>(entry.user_stack.size()), config.perf_max_stack_depth), 
                entry.key.pid);
            if (!symbols.empty()) {
                oss << ";" << stack_trace_to_folded_string(symbols, ';', true);
            }
        }
    }
    
    // Add kernel stack if it exists
    if (entry.has_kernel_stack && !config.user_stacks_only) {
        // Add delimiter between user and kernel stacks if both exist
        if (entry.has_user_stack && config.delimiter && !config.kernel_stacks_only) {
            oss << ";-";
        }
        
        if (entry.kernel_stack.empty()) {
            oss << ";[Missed Kernel Stack]";
        } else {
            auto symbols = get_stack_trace_symbols(
                const_cast<__u64 *>(reinterpret_cast<const __u64 *>(entry.kernel_stack.data())), 
                std::min(static_cast<int>(entry.kernel_stack.size()), config.perf_max_stack_depth), 
                0);
            if (!symbols.empty()) {
                oss << ";" << stack_trace_to_folded_string(symbols, ';', true);
            }
        }
    }
    
    // Add the count at the end
    oss << " " << entry.value << "\n";
    
    return oss.str();
} 