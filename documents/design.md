
Looking at your current codebase structure and the linter errors, I can see several organizational and architectural issues. Here's a comprehensive design plan to make it more well-organized:

## 1. Core Architecture Redesign

### **Proposed Directory Structure:**
```
src/
├── core/                    # Core abstractions and interfaces
│   ├── interfaces/
│   ├── data/
│   ├── config/
│   └── utils/
├── collectors/              # Data collection layer
│   ├── base/
│   ├── bpf/
│   └── plugins/
├── analyzers/               # Analysis layer
│   ├── base/
│   └── implementations/
├── views/                   # Data presentation layer
│   ├── flamegraph/
│   └── text/
├── server/                  # Web server and API
│   ├── api/
│   ├── handlers/
│   └── static/
├── cli/                     # Command line interface
├── third_party/             # External dependencies
└── main.cpp
```

## 2. Layer Separation and Dependencies

### **Dependency Flow:**
```
CLI/Server → Analyzers → Collectors → Core
     ↓            ↓           ↓        ↓
   Views    ←     Views  ←    Data  ←  Interfaces
```

### **Core Layer (`src/core/`):**
- **`interfaces/`**: Pure abstract interfaces
  - `ICollector` - data collection interface
  - `IAnalyzer` - analysis interface  
  - `IDataView` - data presentation interface
  - `ISymbolResolver` - symbol resolution interface

- **`data/`**: Core data structures
  - `sampling_data.hpp` - unified sampling data structures
  - `stack_trace.hpp` - stack trace representations
  - `profiling_event.hpp` - base event structures

- **`config/`**: Configuration management
  - `base_config.hpp` - base configuration classes
  - `collector_config.hpp` - collector-specific configs
  - `analyzer_config.hpp` - analyzer-specific configs

- **`utils/`**: Core utilities
  - `types.hpp` - common type definitions
  - `time_utils.hpp` - timing utilities
  - `error_handling.hpp` - error types and handling

## 3. Collectors Layer Redesign

### **Structure:**
```
collectors/
├── base/
│   ├── base_collector.hpp        # Abstract base collector
│   ├── bpf_collector_base.hpp    # BPF-specific base
│   └── sampling_collector.hpp    # Sampling-specific base
├── bpf/
│   ├── common/
│   │   ├── bpf_event.h           # Move here from current location
│   │   ├── bpf_helpers.hpp
│   │   └── bpf_loader.hpp
│   ├── oncpu/
│   │   ├── profile_collector.hpp
│   │   ├── profile_collector.cpp
│   │   └── profile.bpf.c
│   └── offcpu/
│       ├── offcpu_collector.hpp
│       ├── offcpu_collector.cpp
│       └── offcputime.bpf.c
└── plugins/                      # Future extensibility
    └── custom_collectors/
```

### **Key Changes:**
- Move `bpf_event.h` to `collectors/bpf/common/`
- Create proper base classes with clear interfaces
- Separate BPF program management from data collection logic

## 4. Analyzers Layer Restructure

### **Structure:**
```
analyzers/
├── base/
│   ├── base_analyzer.hpp         # Pure interface
│   ├── sampling_analyzer.hpp     # Sampling-specific base
│   └── multi_collector_analyzer.hpp
├── implementations/
│   ├── profile_analyzer.hpp/.cpp
│   ├── offcpu_analyzer.hpp/.cpp
│   └── wallclock_analyzer.hpp/.cpp
└── utils/
    ├── symbol_resolver.hpp/.cpp  # Move from current location
    └── thread_analyzer.hpp/.cpp
```

### **Key Changes:**
- Move `symbol_resolver` to `analyzers/utils/`
- Create cleaner separation between interface and implementation
- Remove circular dependencies with collectors

## 5. Views Layer (New)

### **Structure:**
```
views/
├── base/
│   ├── data_view_interface.hpp
│   └── export_format.hpp
├── flamegraph/
│   ├── flamegraph_view.hpp/.cpp     # Move from analyzers/
│   ├── flamegraph_generator.hpp/.cpp # Move from root
│   └── flamegraph_renderer.hpp
└── text/
    ├── table_view.hpp
    └── summary_view.hpp
```

### **Key Changes:**
- Extract presentation logic from analyzers
- Create pluggable view system
- Move `flamegraph_view` and `flamegraph_generator` here

## 6. Server Layer Enhancement

### **Structure:**
```
server/
├── api/
│   ├── profiler_api.hpp/.cpp       # Core API logic
│   ├── collector_endpoints.hpp     # Collector management
│   └── analyzer_endpoints.hpp      # Analysis endpoints
├── handlers/
│   ├── static_handler.hpp/.cpp
│   ├── file_handler.hpp/.cpp
│   └── websocket_handler.hpp/.cpp  # Future real-time updates
├── static/
│   ├── html/
│   ├── css/
│   └── js/
└── profile_server.hpp/.cpp         # Main server orchestration
```

### **API Design:**
```
GET  /api/v1/status                 # Server status
GET  /api/v1/collectors             # Available collectors
POST /api/v1/profile/start          # Start profiling
GET  /api/v1/profile/{id}/status    # Profile status
GET  /api/v1/profile/{id}/data      # Get profile data
GET  /api/v1/profile/{id}/flamegraph # Get flamegraph
POST /api/v1/profile/{id}/stop      # Stop profiling
```

## 7. Configuration System Redesign

### **Hierarchical Configuration:**
```cpp
// Base configuration
class BaseConfig {
    int duration;
    std::vector<pid_t> pids;
    std::vector<pid_t> tids;
    LogLevel log_level;
};

// Collector-specific
class CollectorConfig : public BaseConfig {
    virtual ~CollectorConfig() = default;
};

class SamplingCollectorConfig : public CollectorConfig {
    int frequency;
    int stack_depth;
};

// Analyzer-specific  
class AnalyzerConfig : public BaseConfig {
    std::vector<std::unique_ptr<CollectorConfig>> collector_configs;
};
```

## 8. Dependency Injection and Factory Pattern

### **Factory System:**
```cpp
class CollectorFactory {
public:
    static std::unique_ptr<ICollector> create(
        const std::string& type, 
        std::unique_ptr<CollectorConfig> config
    );
};

class AnalyzerFactory {
public:
    static std::unique_ptr<IAnalyzer> create(
        const std::string& type,
        std::unique_ptr<AnalyzerConfig> config,
        std::vector<std::unique_ptr<ICollector>> collectors
    );
};
```

## 9. Error Handling and Logging

### **Centralized Error Handling:**
```cpp
namespace profiler {
    enum class ErrorCode {
        SUCCESS,
        BPF_LOAD_FAILED,
        PERMISSION_DENIED,
        INVALID_CONFIG,
        // ...
    };
    
    class Result<T> {
        // Result type for error handling
    };
}
```

## 10. Build System Organization

### **CMake Structure:**
```
CMakeLists.txt                    # Root
src/
├── core/CMakeLists.txt
├── collectors/CMakeLists.txt
├── analyzers/CMakeLists.txt
├── views/CMakeLists.txt
├── server/CMakeLists.txt
└── cli/CMakeLists.txt
```

## 11. Testing Structure

### **Test Organization:**
```
tests/
├── unit/
│   ├── core/
│   ├── collectors/
│   ├── analyzers/
│   └── views/
├── integration/
│   ├── end_to_end/
│   └── api/
└── fixtures/
    └── test_data/
```

## 12. Migration Strategy

### **Phase 1: Core Abstractions**
1. Create `src/core/` with interfaces and base classes
2. Move common types and utilities
3. Fix circular dependencies

### **Phase 2: Layer Separation**
1. Move collectors to new structure
2. Move views out of analyzers  
3. Update include paths

### **Phase 3: Enhanced Features**
1. Implement factory pattern
2. Add proper error handling
3. Enhance server API

### **Phase 4: Testing and Documentation**
1. Add comprehensive tests
2. Update documentation
3. Add examples

This design addresses the current issues by:
- **Eliminating circular dependencies**
- **Clear separation of concerns**
- **Modular, extensible architecture**
- **Proper abstraction layers**
- **Consistent interfaces**
- **Better testability**

The new structure makes it easier to add new collectors, analyzers, and views without affecting existing code, and provides a solid foundation for future enhancements.

