#ifndef __CONFIG_HPP
#define __CONFIG_HPP

#include "utils.hpp"

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
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
        folded = false;
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

// OffCPU Time specific configuration
class OffCPUTimeConfig : public Config {
public:
    __u64 min_block_time;
    __u64 max_block_time;
    long state;

    OffCPUTimeConfig() : Config() {
        // OffCPU specific defaults
        min_block_time = 1;
        max_block_time = (__u64)(-1);
        state = -1;
        
        // Validate configuration
        validate();
    }

    // Set configuration values with validation
    void set_min_block_time(__u64 time) {
        min_block_time = time;
        validate();
    }

    void set_max_block_time(__u64 time) {
        max_block_time = time;
        validate();
    }

    void set_state(long s) {
        state = s;
        validate();
    }

    void set_user_threads_only(bool value) {
        user_threads_only = value;
        validate();
    }

    void set_kernel_threads_only(bool value) {
        kernel_threads_only = value;
        validate();
    }

protected:
    void validate() override {
        Config::validate(); // Call base validation
        
        if (user_threads_only && kernel_threads_only) {
            throw std::invalid_argument("user_threads_only and kernel_threads_only cannot be used together");
        }
        if (min_block_time >= max_block_time) {
            throw std::invalid_argument("min_block_time should be smaller than max_block_time");
        }
        if (state < -1 || state > 2) {
            throw std::invalid_argument("state must be between -1 and 2");
        }
    }
};

// Profile specific configuration
class ProfileConfig : public Config {
public:
    bool freq;
    int sample_freq;
    bool include_idle;

    ProfileConfig() : Config() {
        // Profile specific defaults
        duration = INT_MAX;
        freq = true;
        sample_freq = 49;
        include_idle = false;
        
        // Validate configuration
        validate();
    }

    // Set configuration values with validation
    void set_sample_freq(int freq) {
        sample_freq = freq;
        validate();
    }

    void set_include_idle(bool value) {
        include_idle = value;
        validate();
    }

    void set_user_stacks_only(bool value) {
        user_stacks_only = value;
        validate();
    }

    void set_kernel_stacks_only(bool value) {
        kernel_stacks_only = value;
        validate();
    }

    void set_cpu(int c) {
        cpu = c;
        validate();
    }

protected:
    void validate() override {
        Config::validate(); // Call base validation
        
        if (user_stacks_only && kernel_stacks_only) {
            throw std::invalid_argument("user_stacks_only and kernel_stacks_only cannot be used together");
        }
        if (sample_freq <= 0) {
            throw std::invalid_argument("sample_freq must be positive");
        }
    }
};

#endif /* __CONFIG_HPP */ 