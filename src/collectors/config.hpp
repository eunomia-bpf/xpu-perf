#ifndef __CONFIG_HPP
#define __CONFIG_HPP

#include "utils.hpp"

#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <limits.h>

#ifdef __cplusplus
}
#endif

#include <stdexcept>

// Base configuration class
class Config {
public:
    pid_t pids[MAX_PID_NR];
    pid_t tids[MAX_TID_NR];
    bool user_threads_only;
    bool kernel_threads_only;
    bool user_stacks_only;
    bool kernel_stacks_only;
    int stack_storage_size;
    int perf_max_stack_depth;
    int duration;
    bool verbose;
    bool folded;
    bool delimiter;
    int cpu;

    Config() {
        // Initialize arrays to zero
        memset(pids, 0, sizeof(pids));
        memset(tids, 0, sizeof(tids));
        
        // Set default values
        user_threads_only = false;
        kernel_threads_only = false;
        user_stacks_only = false;
        kernel_stacks_only = false;
        stack_storage_size = 1024;
        perf_max_stack_depth = 127;
        duration = 99999999;
        verbose = false;
        folded = true;
        delimiter = false;
        cpu = -1;
    }

protected:
    // Base validation - can be overridden by derived classes
    virtual void validate() {
        if (stack_storage_size <= 0) {
            throw std::invalid_argument("stack_storage_size must be positive");
        }
        if (perf_max_stack_depth <= 0) {
            throw std::invalid_argument("perf_max_stack_depth must be positive");
        }
        if (duration <= 0) {
            throw std::invalid_argument("duration must be positive");
        }
    }
};

#endif /* __CONFIG_HPP */ 