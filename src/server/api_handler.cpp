#include "api_handler.hpp"
#include "../third_party/json/single_include/nlohmann/json.hpp"
#include "../third_party/spdlog/include/spdlog/spdlog.h"
#include <chrono>
#include <random>
#include <sstream>
#include <iomanip>

using json = nlohmann::json;

namespace server {

StatusHandler::StatusHandler() : start_time(std::time(nullptr)) {}

void StatusHandler::handle(const httplib::Request& req, httplib::Response& res) {
    spdlog::debug("Status request received");
    
    json status = {
        {"status", "running"},
        {"version", "1.0"},
        {"uptime_seconds", std::time(nullptr) - start_time}
    };
    
    res.set_content(status.dump(2), "application/json");
    spdlog::debug("Status response sent");
}

// ProfileAnalyzerHandler Implementation

std::string ProfileAnalyzerHandler::generate_session_id() {
    auto now = std::chrono::high_resolution_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1000, 9999);
    
    std::stringstream ss;
    ss << "profile_" << timestamp << "_" << dis(gen);
    return ss.str();
}

std::unique_ptr<ProfileAnalyzerConfig> ProfileAnalyzerHandler::parse_config(const std::string& json_body) {
    try {
        auto j = json::parse(json_body);
        auto config = std::make_unique<ProfileAnalyzerConfig>();
        
        // Parse required fields
        config->duration = j.value("duration", 30);
        
        // Parse targets
        if (j.contains("targets")) {
            auto targets = j["targets"];
            if (targets.contains("pids") && targets["pids"].is_array()) {
                for (auto pid : targets["pids"]) {
                    config->pids.push_back(pid.get<pid_t>());
                }
            }
            if (targets.contains("tids") && targets["tids"].is_array()) {
                for (auto tid : targets["tids"]) {
                    config->tids.push_back(tid.get<pid_t>());
                }
            }
        }
        
        // Parse analyzer-specific config
        if (j.contains("config")) {
            auto analyzer_config = j["config"];
            config->frequency = analyzer_config.value("frequency", 99);
        }
        
        return config;
    } catch (const std::exception& e) {
        spdlog::error("Failed to parse config: {}", e.what());
        return nullptr;
    }
}

void ProfileAnalyzerHandler::handle_start(const httplib::Request& req, httplib::Response& res) {
    spdlog::info("Profile analyzer start request received");
    
    auto config = parse_config(req.body);
    if (!config) {
        json error = {
            {"error", {
                {"code", "INVALID_CONFIG"},
                {"message", "Failed to parse configuration"}
            }}
        };
        res.status = 400;
        res.set_content(error.dump(2), "application/json");
        return;
    }
    
    std::string session_id = generate_session_id();
    
    try {
        auto analyzer = std::make_unique<ProfileAnalyzer>(std::move(config));
        
        if (!analyzer->start()) {
            json error = {
                {"error", {
                    {"code", "START_FAILED"},
                    {"message", "Failed to start profiler"}
                }}
            };
            res.status = 500;
            res.set_content(error.dump(2), "application/json");
            return;
        }
        
        active_sessions_[session_id] = std::move(analyzer);
        
        json response = {
            {"session_id", session_id},
            {"status", "started"},
            {"message", "Profiling session started successfully"}
        };
        
        res.set_content(response.dump(2), "application/json");
        spdlog::info("Started profile session: {}", session_id);
        
    } catch (const std::exception& e) {
        json error = {
            {"error", {
                {"code", "INTERNAL_ERROR"},
                {"message", e.what()}
            }}
        };
        res.status = 500;
        res.set_content(error.dump(2), "application/json");
    }
}

void ProfileAnalyzerHandler::handle_status(const httplib::Request& req, httplib::Response& res) {
    std::string session_id = req.matches[2]; // Assuming URL pattern captures session_id
    
    auto it = active_sessions_.find(session_id);
    if (it == active_sessions_.end()) {
        json error = {
            {"error", {
                {"code", "SESSION_NOT_FOUND"},
                {"message", "Session not found"}
            }}
        };
        res.status = 404;
        res.set_content(error.dump(2), "application/json");
        return;
    }
    
    // For simplicity, we'll return a basic status
    // In a real implementation, you'd track the actual profiling status
    json response = {
        {"session_id", session_id},
        {"status", "running"},
        {"message", "Profiling in progress"}
    };
    
    res.set_content(response.dump(2), "application/json");
}

void ProfileAnalyzerHandler::handle_views(const httplib::Request& req, httplib::Response& res) {
    std::string session_id = req.matches[2]; // Assuming URL pattern captures session_id
    
    auto it = active_sessions_.find(session_id);
    if (it == active_sessions_.end()) {
        json error = {
            {"error", {
                {"code", "SESSION_NOT_FOUND"},
                {"message", "Session not found"}
            }}
        };
        res.status = 404;
        res.set_content(error.dump(2), "application/json");
        return;
    }
    
    try {
        auto flamegraph = it->second->get_flamegraph();
        if (!flamegraph || !flamegraph->success) {
            json error = {
                {"error", {
                    {"code", "NO_DATA"},
                    {"message", "No flamegraph data available"}
                }}
            };
            res.status = 500;
            res.set_content(error.dump(2), "application/json");
            return;
        }
        
        // Convert FlameGraphView to d3-flame-graph format
        json d3_data = json::object();
        d3_data["name"] = flamegraph->analyzer_name;
        d3_data["value"] = flamegraph->total_samples;
        d3_data["children"] = json::array();
        
        // Group entries by their first stack frame to create hierarchy
        std::map<std::string, json> root_functions;
        
        for (const auto& entry : flamegraph->entries) {
            if (entry.folded_stack.empty()) continue;
            
            std::string root_func = entry.folded_stack[0];
            if (root_functions.find(root_func) == root_functions.end()) {
                root_functions[root_func] = json::object();
                root_functions[root_func]["name"] = root_func;
                root_functions[root_func]["value"] = 0;
                root_functions[root_func]["children"] = json::array();
            }
            root_functions[root_func]["value"] = root_functions[root_func]["value"].get<uint64_t>() + entry.sample_count;
        }
        
        // Add root functions to children
        for (auto& [func_name, func_data] : root_functions) {
            d3_data["children"].push_back(func_data);
        }
        
        json response = {
            {"session_id", session_id},
            {"analyzer_type", "profile"},
            {"views", {
                {"flamegraph", {
                    {"type", "flamegraph"},
                    {"data", {
                        {"d3_format", d3_data},
                        {"folded_format", flamegraph->to_folded_format()},
                        {"analyzer_name", flamegraph->analyzer_name},
                        {"success", flamegraph->success},
                        {"total_samples", flamegraph->total_samples},
                        {"time_unit", flamegraph->time_unit}
                    }}
                }}
            }}
        };
        
        res.set_content(response.dump(2), "application/json");
        spdlog::info("Returned flamegraph data for session: {}", session_id);
        
    } catch (const std::exception& e) {
        json error = {
            {"error", {
                {"code", "INTERNAL_ERROR"},
                {"message", e.what()}
            }}
        };
        res.status = 500;
        res.set_content(error.dump(2), "application/json");
    }
}

void ProfileAnalyzerHandler::handle_stop(const httplib::Request& req, httplib::Response& res) {
    std::string session_id = req.matches[2];
    
    auto it = active_sessions_.find(session_id);
    if (it == active_sessions_.end()) {
        json error = {
            {"error", {
                {"code", "SESSION_NOT_FOUND"},
                {"message", "Session not found"}
            }}
        };
        res.status = 404;
        res.set_content(error.dump(2), "application/json");
        return;
    }
    
    // For now, we just mark it as stopped
    // In a real implementation, you'd stop the actual profiling
    json response = {
        {"session_id", session_id},
        {"status", "stopped"},
        {"message", "Profiling session stopped"}
    };
    
    res.set_content(response.dump(2), "application/json");
    spdlog::info("Stopped profile session: {}", session_id);
}

void ProfileAnalyzerHandler::handle_delete(const httplib::Request& req, httplib::Response& res) {
    std::string session_id = req.matches[2];
    
    auto it = active_sessions_.find(session_id);
    if (it == active_sessions_.end()) {
        json error = {
            {"error", {
                {"code", "SESSION_NOT_FOUND"},
                {"message", "Session not found"}
            }}
        };
        res.status = 404;
        res.set_content(error.dump(2), "application/json");
        return;
    }
    
    active_sessions_.erase(it);
    
    json response = {
        {"session_id", session_id},
        {"status", "deleted"},
        {"message", "Session deleted successfully"}
    };
    
    res.set_content(response.dump(2), "application/json");
    spdlog::info("Deleted profile session: {}", session_id);
}

// AnalyzerHandlerRegistry Implementation

void AnalyzerHandlerRegistry::register_handler(std::unique_ptr<IAnalyzerHandler> handler) {
    handlers_[handler->get_analyzer_type()] = std::move(handler);
}

IAnalyzerHandler* AnalyzerHandlerRegistry::get_handler(const std::string& analyzer_type) {
    auto it = handlers_.find(analyzer_type);
    return (it != handlers_.end()) ? it->second.get() : nullptr;
}

std::vector<std::string> AnalyzerHandlerRegistry::get_supported_types() const {
    std::vector<std::string> types;
    for (const auto& [type, handler] : handlers_) {
        types.push_back(type);
    }
    return types;
}

} // namespace server 