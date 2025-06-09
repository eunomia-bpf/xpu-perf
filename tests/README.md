# Profiler Test Framework

This directory contains the test suite for the BPF Profiler project using the Catch2 testing framework.

## Test Structure

- `test_main.cpp` - Contains the main test runner with Catch2 configuration
- `test_flamegraph_view.cpp` - Comprehensive tests for the FlameGraphView class
- **Catch2 is automatically downloaded via CMake FetchContent** (v2.13.10)

## Dependencies

The test framework automatically downloads and builds:
- **Catch2 v2.13.10** - Modern C++ testing framework via CMake FetchContent

No manual setup required! Just build the project and the dependencies will be fetched automatically.

## Running Tests

### Build and Run Tests
```bash
# From the project root directory
make -C build              # Build everything including tests (downloads Catch2 automatically)
./build/profiler_tests     # Run tests directly
```

### Using CTest
```bash
# From the build directory
cd build
ctest --verbose           # Run all tests with verbose output
ctest                     # Run tests quietly
```

### Test specific functionality
```bash
# Run only flamegraph tests
./build/profiler_tests "[flamegraph]"

# List all available tests
./build/profiler_tests --list-tests
```

## CMake Integration

The project uses modern CMake practices:

### FetchContent for Dependencies
```cmake
include(FetchContent)

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v2.13.10
    GIT_SHALLOW    TRUE
)

FetchContent_MakeAvailable(Catch2)
```

### Force Unix Makefiles Generator
The project is configured to use Unix Makefiles to avoid issues with Ninja and the custom BPF build commands. This is enforced through:
- `CMakePresets.json` - Defines build presets with Unix Makefiles
- `.vscode/settings.json` - VS Code configuration for proper generator selection

## What's Tested

### FlameGraphView Vector-based Functionality
- ✅ **FlameGraphEntry structure** - Basic data structure operations
- ✅ **FlameGraphView basic functionality** - Constructor and initialization
- ✅ **Add stack trace and build folded stack** - Core stack building logic
- ✅ **Stack depth calculation** - Proper depth counting for vector-based stacks
- ✅ **Folded format output** - Conversion back to semicolon-separated format
- ✅ **Readable format output** - Human-readable stack trace formatting
- ✅ **Function totals aggregation** - Cross-stack function counting
- ✅ **Top stacks retrieval** - Sorting and limiting functionality

## Key Changes from String to Vector

The tests validate the transition from string-based to vector-based folded stacks:

### Before (String-based)
```cpp
std::string folded_stack = "myapp;main;process_data;parse_input";
```

### After (Vector-based)
```cpp
std::vector<std::string> folded_stack = {"myapp", "main", "process_data", "parse_input"};
```

### Benefits Tested
1. **Performance** - Direct access to individual functions without parsing
2. **Flexibility** - Each function is a separate element for easier analysis
3. **Maintainability** - Cleaner code logic without string manipulation
4. **Compatibility** - Output formats maintain backward compatibility

## Test Coverage

The test suite covers:
- Stack trace construction with user and kernel stacks
- Delimiter handling between user and kernel space
- Stack depth calculations
- Sample aggregation and percentage calculations
- Output format generation (both folded and readable)
- Function-level statistics
- Sorting and top-stack retrieval

## Adding New Tests

To add new tests, create a new `TEST_CASE` in an appropriate file:

```cpp
TEST_CASE("Your test description", "[tag]") {
    SECTION("Specific functionality") {
        // Your test code here
        REQUIRE(condition == expected_value);
    }
}
```

For more information about Catch2, see: https://github.com/catchorg/Catch2 