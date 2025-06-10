# BPF Profiler Tests

This directory contains comprehensive **functional tests** for the BPF profiler components using Catch2 framework.

## Test Philosophy

All tests are **functional** - they actually start the BPF collectors and test real profiling scenarios rather than meaningless unit tests. Tests gracefully handle environments where BPF is not available.

---

## 📊 **ProfileCollector Tests** (`test_profile_collector.cpp`)

### **Functional Tests** `[profile][functional]`
- **Collector Lifecycle**: Start/stop BPF profile collection with real perf events
- **Data Collection**: Generate CPU workload and verify actual sampling data is collected

### **Hardware Events** `[profile][hardware]` 
- **CPU Cycles Profiling**: Uses `PERF_COUNT_HW_CPU_CYCLES` with compute-intensive workload
- **Instructions Profiling**: Uses `PERF_COUNT_HW_INSTRUCTIONS` with instruction-heavy loops  
- **Cache Miss Profiling**: Uses `PERF_COUNT_HW_CACHE_MISSES` with random memory access patterns

### **Software Events** `[profile][software]`
- **Page Fault Profiling**: Uses `PERF_COUNT_SW_PAGE_FAULTS` with memory allocation workload
- **Context Switch Profiling**: Uses `PERF_COUNT_SW_CONTEXT_SWITCHES` with competing threads

### **Configuration Options** `[profile][config]`
- **High Frequency Sampling**: Tests with 5000 Hz sampling rate
- **Idle Process Inclusion**: Tests `include_idle` flag functionality

### **Error Handling** `[profile][error]`
- **Invalid Configurations**: Tests graceful failure with invalid perf event types
- **Resource Management**: Tests multiple start calls and error states

---

## 🔄 **OffCPUTimeCollector Tests** (`test_offcputime_collector.cpp`)

### **Functional Tests** `[offcpu][functional]`
- **Collector Lifecycle**: Start/stop BPF off-CPU time tracking
- **Baseline Data Collection**: Verify initial off-CPU data collection works

### **Blocking Workloads** `[offcpu][workload]`
- **Thread Synchronization**: Uses `std::condition_variable` to create measured blocking
- **Mutex Contention**: Tests detection of thread blocking on synchronization primitives

### **Sleep Workloads** `[offcpu][sleep]`
- **Sleep Detection**: Uses `std::this_thread::sleep_for()` to generate TASK_INTERRUPTIBLE events
- **Multi-threaded Sleep**: Tests detection across multiple sleeping threads

### **Configuration Tests** `[offcpu][config]`
- **Block Time Thresholds**: Tests `min_block_time`/`max_block_time` filtering (5ms-1s range)
- **Task State Filtering**: Tests `TASK_INTERRUPTIBLE` vs `TASK_UNINTERRUPTIBLE` state detection

### **Error Handling** `[offcpu][error]`
- **Unstarted Collection**: Tests data collection without starting the collector
- **Data Structure Validation**: Verifies stack trace consistency and blocking time values

---

## 🚀 **Running Tests**

### **All Tests**
```bash
make test
# or with sudo for BPF access
sudo ./build/profiler_tests
```

### **Specific Test Categories**
```bash
# Test hardware perf events
sudo ./build/profiler_tests "[hardware]" -s

# Test off-CPU blocking detection  
sudo ./build/profiler_tests "[workload]" -s

# Test functional collector lifecycle
sudo ./build/profiler_tests "[functional]" -s
```

### **Environment Requirements**

| Test Category | Requirements | Fallback Behavior |
|---------------|-------------|-------------------|
| **Functional Tests** | BPF support, privileges | Graceful skip with info |
| **Hardware Events** | Hardware perf counters | Graceful failure detection |
| **Software Events** | Basic perf_event support | Should work in most environments |
| **Blocking Tests** | BPF + thread scheduler events | Skip if BPF unavailable |

---

## 📈 **What These Tests Verify**

✅ **BPF Program Loading**: Collectors can load and attach BPF programs  
✅ **Perf Event Integration**: Hardware/software perf events work correctly  
✅ **Data Collection**: Real profiling data is captured from workloads  
✅ **Stack Trace Capture**: User and kernel stack traces are collected  
✅ **Thread Blocking Detection**: Off-CPU time measurement works  
✅ **Configuration Options**: Different sampling rates and filters work  
✅ **Error Handling**: Graceful failure when BPF/privileges unavailable  

---

## 📊 **Current Test Coverage**

- **47 assertions** across **12 test cases**
- **ProfileCollector**: 6 test cases covering hardware events, software events, configuration, and error handling
- **OffCPUTimeCollector**: 5 test cases covering blocking detection, sleep profiling, configuration, and error handling  
- **FlameGraph**: 1 test case for data structure validation

**All tests are functional and meaningful** - no configuration-only or trivial tests.