#ifndef __SYMBOL_RESOLVER_HPP
#define __SYMBOL_RESOLVER_HPP

#include "collectors/config.hpp"
#include "collectors/bpf_event.h"
#include <memory>
#include <vector>
#include <sstream>
#include <string>
#include "collectors/sampling_data.hpp"

// Forward declare blazesym functions until blazesym.h is available
extern "C" {
    struct blazesym;
    struct blazesym* blazesym_new(void);
    void blazesym_free(struct blazesym* symbolizer);
}

// Custom deleter for blazesym
struct BlazesymDeleter {
    void operator()(struct blazesym* sym) const {
        if (sym) {
            blazesym_free(sym);
        }
    }
};

/**
 * SymbolResolver - A class to resolve symbols using blazesym
 * 
 * This class encapsulates the blazesym symbolizer and provides methods
 * to resolve stack trace addresses to function symbols.
 */
class SymbolResolver {
private:
    std::unique_ptr<struct blazesym, BlazesymDeleter> symbolizer_;
    
    // Private helper methods
    std::string stack_trace_to_string(const std::vector<std::string>& symbols);
    std::string stack_trace_to_folded_string(const std::vector<std::string>& symbols, 
                                           char separator = ';', bool reverse = false);

public:
    /**
     * Constructor - Creates a new SymbolResolver with its own symbolizer
     */
    SymbolResolver();
    
    /**
     * Destructor - Cleanup is handled by unique_ptr
     */
    ~SymbolResolver() = default;
    
    // Non-copyable but movable
    SymbolResolver(const SymbolResolver&) = delete;
    SymbolResolver& operator=(const SymbolResolver&) = delete;
    SymbolResolver(SymbolResolver&&) = default;
    SymbolResolver& operator=(SymbolResolver&&) = default;
    
    /**
     * get_stack_trace_symbols - Convert stack trace to function name vector
     * @stack: Array of stack addresses
     * @stack_sz: Size of the stack array
     * @pid: Process ID (0 for kernel)
     * @return: Vector of function names
     */
    std::vector<std::string> get_stack_trace_symbols(__u64 *stack, int stack_sz, pid_t pid);
    
    /**
     * is_valid - Check if the symbolizer is properly initialized
     * @return: true if symbolizer is valid, false otherwise
     */
    bool is_valid() const { return symbolizer_ != nullptr; }
};

#endif /* __SYMBOL_RESOLVER_HPP */ 