
Looking at your analyzers and the `FlameGraphView` structure, I'll design a more extensible API that can easily accommodate new analyzers while using your predefined views. Here's a simplified, extensible approach:

## Core Extensible API Design

### 1. Generic Analyzer Endpoints

```
POST /api/v1/analyzers/{analyzer_type}/start
GET  /api/v1/analyzers/{analyzer_type}/{session_id}/status  
GET  /api/v1/analyzers/{analyzer_type}/{session_id}/views
POST /api/v1/analyzers/{analyzer_type}/{session_id}/stop
DELETE /api/v1/analyzers/{analyzer_type}/{session_id}
```

**Supported analyzer_type values:**
- `profile` (ProfileAnalyzer)
- `offcpu` (OffCPUTimeAnalyzer) 
- `wallclock` (WallClockAnalyzer)
- `memory` (future MemoryAnalyzer)
- `io` (future IOAnalyzer)

### 2. Unified Configuration Format

**Generic Start Request:**
```json
{
  "session_name": "my_analysis_session",
  "duration": 30,
  "targets": {
    "pids": [1234, 5678],
    "tids": [9012, 3456]
  },
  "config": {
    // Analyzer-specific config goes here
  }
}
```

**Analyzer-Specific Configs:**

**Profile Analyzer:**
```json
{
  "config": {
    "frequency": 99
  }
}
```

**OffCPU Analyzer:**
```json
{
  "config": {
    "min_block_us": 1000
  }
}
```

**WallClock Analyzer:**
```json
{
  "config": {
    "cpu_frequency": 99,
    "offcpu_min_block_us": 1000,
    "auto_discover_threads": true
  }
}
```

### 3. Standardized View Response Format

**Single View Response (Profile/OffCPU):**
```json
{
  "session_id": "uuid-here",
  "analyzer_type": "profile",
  "views": {
    "flamegraph": {
      "type": "flamegraph",
      "data": {
        "analyzer_name": "profile_analyzer",
        "success": true,
        "total_samples": 15000,
        "total_time_seconds": 30.0,
        "time_unit": "samples",
        "entries": [
          {
            "folded_stack": ["main", "process_data", "compute"],
            "command": "myapp",
            "pid": 1234,
            "sample_count": 1500,
            "percentage": 10.0,
            "stack_depth": 3,
            "is_oncpu": true
          }
        ],
        "formats": {
          "folded": "/api/v1/analyzers/profile/uuid-here/views/flamegraph/folded",
          "summary": "/api/v1/analyzers/profile/uuid-here/views/flamegraph/summary", 
          "readable": "/api/v1/analyzers/profile/uuid-here/views/flamegraph/readable"
        }
      }
    }
  }
}
```

**Multi-View Response (WallClock):**
```json
{
  "session_id": "uuid-here", 
  "analyzer_type": "wallclock",
  "views": {
    "per_thread_flamegraphs": {
      "type": "multi_flamegraph",
      "data": {
        "threads": {
          "1234": {
            "analyzer_name": "wallclock_main_thread",
            "success": true,
            "total_samples": 8000,
            "entries": [...]
          },
          "1235": {
            "analyzer_name": "wallclock_worker_thread", 
            "success": true,
            "total_samples": 7000,
            "entries": [...]
          }
        }
      }
    },
    "thread_discovery": {
      "type": "thread_info",
      "data": {
        "is_multithreaded": true,
        "threads": [
          {
            "tid": 1234,
            "pid": 1234, 
            "role": "main",
            "command": "/usr/bin/myapp"
          }
        ]
      }
    }
  }
}
```

### 4. Extensible Handler Design

**Base Handler Interface:**
```cpp
// Base analyzer handler interface
class IAnalyzerHandler {
public:
    virtual ~IAnalyzerHandler() = default;
    virtual void handle_start(const httplib::Request& req, httplib::Response& res) = 0;
    virtual void handle_status(const httplib::Request& req, httplib::Response& res) = 0;
    virtual void handle_views(const httplib::Request& req, httplib::Response& res) = 0;
    virtual void handle_stop(const httplib::Request& req, httplib::Response& res) = 0;
    virtual void handle_delete(const httplib::Request& req, httplib::Response& res) = 0;
    virtual std::string get_analyzer_type() const = 0;
};

// Registry for easy extension
class AnalyzerHandlerRegistry {
private:
    std::map<std::string, std::unique_ptr<IAnalyzerHandler>> handlers_;
    
public:
    void register_handler(std::unique_ptr<IAnalyzerHandler> handler) {
        handlers_[handler->get_analyzer_type()] = std::move(handler);
    }
    
    IAnalyzerHandler* get_handler(const std::string& analyzer_type) {
        auto it = handlers_.find(analyzer_type);
        return (it != handlers_.end()) ? it->second.get() : nullptr;
    }
    
    std::vector<std::string> get_supported_types() const {
        std::vector<std::string> types;
        for (const auto& [type, handler] : handlers_) {
            types.push_back(type);
        }
        return types;
    }
};
```

**Concrete Handler Example:**
```cpp
class ProfileAnalyzerHandler : public IAnalyzerHandler {
private:
    std::map<std::string, std::unique_ptr<ProfileAnalyzer>> active_sessions_;
    
public:
    std::string get_analyzer_type() const override { return "profile"; }
    
    void handle_start(const httplib::Request& req, httplib::Response& res) override {
        // Parse generic config, create ProfileAnalyzerConfig
        // Start ProfileAnalyzer, return session_id
    }
    
    void handle_views(const httplib::Request& req, httplib::Response& res) override {
        // Get FlameGraphView from analyzer
        // Convert to standardized JSON format
    }
    
    // ... other methods
};
```

### 5. Simple Registration System

**Main API Setup:**
```cpp
class AnalyzerAPI {
private:
    AnalyzerHandlerRegistry registry_;
    
public:
    void setup_routes(httplib::Server& server) {
        // Generic routes that delegate to registered handlers
        server.Post("/api/v1/analyzers/(.*)/start", [this](const httplib::Request& req, httplib::Response& res) {
            auto analyzer_type = req.matches[1];
            auto handler = registry_.get_handler(analyzer_type);
            if (handler) {
                handler->handle_start(req, res);
            } else {
                res.status = 404;
                res.set_content("{\"error\": \"Analyzer type not supported\"}", "application/json");
            }
        });
        
        // Similar patterns for other endpoints...
    }
    
    void register_default_analyzers() {
        registry_.register_handler(std::make_unique<ProfileAnalyzerHandler>());
        registry_.register_handler(std::make_unique<OffCPUAnalyzerHandler>());
        registry_.register_handler(std::make_unique<WallClockAnalyzerHandler>());
    }
};
```

### 6. Discovery Endpoint

```
GET /api/v1/analyzers/types
```

**Response:**
```json
{
  "supported_analyzers": [
    {
      "type": "profile",
      "name": "CPU Profiler",
      "description": "On-CPU sampling profiler",
      "config_schema": {
        "frequency": {"type": "integer", "default": 99, "range": [1, 1000]}
      },
      "view_types": ["flamegraph"]
    },
    {
      "type": "wallclock", 
      "name": "Wall-Clock Analyzer",
      "description": "Combined CPU and off-CPU analysis",
      "config_schema": {
        "cpu_frequency": {"type": "integer", "default": 99},
        "offcpu_min_block_us": {"type": "integer", "default": 1000},
        "auto_discover_threads": {"type": "boolean", "default": true}
      },
      "view_types": ["multi_flamegraph", "thread_info"]
    }
  ]
}
```

## Benefits of This Design:

1. **Easy Extension**: Just implement `IAnalyzerHandler` and register it
2. **Consistent API**: Same endpoints work for all analyzer types  
3. **Flexible Config**: Analyzer-specific config within standard structure
4. **Standard Views**: All return FlameGraphView data in consistent format
5. **Future-Proof**: Can easily add new view types and analyzers
6. **Simple Start**: Begin with basic profile analyzer, add others incrementally

This design leverages your existing `FlameGraphView` structure while making the API highly extensible for future analyzers.
