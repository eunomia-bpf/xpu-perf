# BPF Profiler Tests

This directory contains unit tests for the BPF profiler components using Catch2 framework.

## Test Organization

We use a **single test executable** approach for better build performance and easier CI/CD integration.

### Current Test Coverage

#### Core Components
- `test_flamegraph_view.cpp` - FlameGraph data structure and visualization tests
- `test_profile_collector.cpp` - ProfileCollector (on-CPU profiling) tests  
- `test_offcputime_collector.cpp` - OffCPUTimeCollector (off-CPU profiling) tests

#### Test Categories
- **Configuration Tests** - Validate config classes, defaults, and parameter handling
- **Collector Tests** - Test collector lifecycle, functionality, and API
- **Data Structure Tests** - Verify data types, aliases, and data handling
- **Edge Case Tests** - Test boundary conditions and error scenarios

## Running Tests

### Build and Run All Tests
```bash
make test
```

### Run Specific Test Categories
```bash
# Run all profile collector tests
./build/profiler_tests "[profile]"

# Run all off-CPU tests  
./build/profiler_tests "[offcpu]"

# Run all config-related tests
./build/profiler_tests "[config]"

# Run collector functionality tests
./build/profiler_tests "[collector]"

# Run edge case tests
./build/profiler_tests "[edge]"
```

### Run Individual Test Cases
```bash
# Run specific test case
./build/profiler_tests "ProfileConfig validation and defaults"

# Run with verbose output
./build/profiler_tests "[profile]" -v

# List all available tests
./build/profiler_tests --list-tests
```

## Test Structure

### ProfileCollector Tests (`test_profile_collector.cpp`)

Tests for on-CPU profiling functionality:

- ✅ **Configuration Validation**: Default values, perf_event_attr setup
- ✅ **Collector Lifecycle**: Constructor, destructor, name retrieval
- ✅ **Config Management**: Mutable/const access, PID/TID filtering
- ✅ **Data Structures**: ProfileData/ProfileEntry type aliases  
- ✅ **Edge Cases**: Multiple collectors, libbpf output isolation
- ✅ **Performance Events**: perf_event_attr configuration options

### OffCPUTimeCollector Tests (`test_offcputime_collector.cpp`)

Tests for off-CPU profiling functionality:

- ✅ **Configuration Validation**: Block time ranges, state filtering
- ✅ **Collector Lifecycle**: Constructor, destructor, name retrieval
- ✅ **Blocking Analysis**: Min/max block times, state values (TASK_*)
- ✅ **Data Structures**: OffCPUData/OffCPUEntry type aliases
- ✅ **Edge Cases**: Extreme values, collector isolation
- ✅ **State Semantics**: TASK_INTERRUPTIBLE, TASK_UNINTERRUPTIBLE

### FlameGraph Tests (`test_flamegraph_view.cpp`)

Tests for flamegraph data visualization:

- ✅ **Data Structures**: FlameGraphEntry, stack trace handling
- ✅ **Vector-based Stacks**: Folded stack functionality

## Test Design Principles

### Why Single Executable?
1. **Faster Build Times** - Shared dependencies, single link step
2. **Easier CI/CD** - One command to run all tests
3. **Better Resource Management** - Shared fixtures and utilities
4. **Catch2 Integration** - Built-in test discovery and filtering

### Testing Protected Methods
Since `validate()` methods are protected, we test validation **indirectly**:
- Through constructor behavior
- Through configuration setters/getters
- Through actual collector operations

### Test Categories (Tags)
- `[config]` - Configuration classes and validation
- `[collector]` - Collector lifecycle and functionality  
- `[data]` - Data structures and type aliases
- `[edge]` - Edge cases and error conditions
- `[profile]` - ProfileCollector specific tests
- `[offcpu]` - OffCPUTimeCollector specific tests

## Adding New Tests

### For New Collectors
1. Create `test_<collector_name>.cpp`
2. Add to `CMakeLists.txt` in `profiler_tests` target
3. Follow the existing patterns for config/collector/data tests
4. Use appropriate tags for categorization

### For New Features
1. Add test sections to existing files if related
2. Use descriptive section names
3. Include both positive and negative test cases
4. Test edge cases and boundary conditions

## Integration with Build System

Tests are automatically built and run via:
```cmake
# In CMakeLists.txt
add_executable(profiler_tests
    tests/test_main.cpp
    tests/test_flamegraph_view.cpp
    tests/test_profile_collector.cpp
    tests/test_offcputime_collector.cpp
)
```

The tests use the same `profiler_lib` as the main application, ensuring consistency

## Test Structure

- `test_main.cpp`