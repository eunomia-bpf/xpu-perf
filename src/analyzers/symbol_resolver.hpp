#ifndef __SYMBOL_RESOLVER_HPP
#define __SYMBOL_RESOLVER_HPP

#include "collectors/utils.hpp"
#include "collectors/config.hpp"
#include "collectors/bpf_event.h"
#include <memory>
#include <vector>
#include <sstream>
#include <string>
#include "collectors/sampling_data.hpp"

struct blazesym;
// Custom deleter for blazesym
struct BlazesymDeleter;

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
    std::string print_entry_multiline(const SamplingEntry& entry, const Config& config, const std::string& value_label = "");
    std::string print_entry_folded(const SamplingEntry& entry, const Config& config);
    std::string stack_trace_to_string(const std::vector<std::string>& symbols);
    std::string stack_trace_to_folded_string(const std::vector<std::string>& symbols, 
                                           char separator = ';', bool reverse = true);

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
     * print_data - Format and print sampling data
     * @data: The sampling data to print
     * @config: Configuration for output formatting
     * @value_label: Optional label for the value column
     * @return: Formatted string representation
     */
    std::string print_data(const SamplingData& data, const Config& config, const std::string& value_label = "");
    
    /**
     * is_valid - Check if the symbolizer is properly initialized
     * @return: true if symbolizer is valid, false otherwise
     */
    bool is_valid() const { return symbolizer_ != nullptr; }
};

#endif /* __SYMBOL_RESOLVER_HPP */ 