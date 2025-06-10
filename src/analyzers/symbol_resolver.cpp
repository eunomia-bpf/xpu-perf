#include "symbol_resolver.hpp"
#include "collectors/sampling_data.hpp"
#include <sstream>
#include <cstring>
#include <algorithm>

#ifdef __cplusplus
extern "C" {
#endif

#include "blazesym.h"

#ifdef __cplusplus
}
#endif

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
