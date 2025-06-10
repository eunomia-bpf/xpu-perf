#include <httplib.h>
#include <nlohmann/json.hpp>
#include <iostream>
#include <string>
#include <memory>
#include <thread>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <signal.h>

#include "../analyzers/analyzer.hpp"
#include "../analyzers/wallclock_analyzer.hpp"
#include "../args_parser.hpp"
#include "../collectors/utils.hpp"
#include "../collectors/oncpu/profile.hpp"
#include "../collectors/offcpu/offcputime.hpp"
#include "../flamegraph_generator.hpp"

using json = nlohmann::json;
namespace fs = std::filesystem;

class ProfilerServer {
private:
    httplib::Server server;
    int port;
    std::string host;
    volatile bool running = true;
    std::unique_ptr<IAnalyzer> active_analyzer;
    std::string output_dir;
    ProfilerArgs current_args;
    
public:
    ProfilerServer(const std::string& host = "localhost", int port = 8080) 
        : host(host), port(port) {
        setup_routes();
    }
    
    void setup_routes() {
        // Enable CORS for all routes
        server.set_pre_routing_handler([](const httplib::Request& req, httplib::Response& res) {
            res.set_header("Access-Control-Allow-Origin", "*");
            res.set_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS");
            res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");
            return httplib::Server::HandlerResponse::Unhandled;
        });
        
        // Handle OPTIONS requests for CORS
        server.Options(".*", [](const httplib::Request&, httplib::Response& res) {
            return;
        });
        
        // Root endpoint
        server.Get("/", [this](const httplib::Request&, httplib::Response& res) {
            json response = {
                {"message", "BPF Profiler Server"},
                {"version", "1.0"},
                {"status", "running"},
                {"active_analyzer", active_analyzer ? active_analyzer->get_name() : "none"}
            };
            res.set_content(response.dump(2), "application/json");
        });
        
        // Health check endpoint
        server.Get("/health", [](const httplib::Request&, httplib::Response& res) {
            json response = {
                {"status", "healthy"},
                {"timestamp", std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count()}
            };
            res.set_content(response.dump(2), "application/json");
        });
        
        // Start profiling endpoint
        server.Post("/profile/start", [this](const httplib::Request& req, httplib::Response& res) {
            if (active_analyzer) {
                json error = {{"error", "Profiler already running"}};
                res.status = 400;
                res.set_content(error.dump(2), "application/json");
                return;
            }
            
            try {
                json request_body = json::parse(req.body);
                ProfilerArgs args = parse_profile_request(request_body);
                
                if (start_profiling(args)) {
                    json response = {
                        {"message", "Profiling started"},
                        {"analyzer", args.analyzer_type},
                        {"duration", args.duration}
                    };
                    res.set_content(response.dump(2), "application/json");
                } else {
                    json error = {{"error", "Failed to start profiler"}};
                    res.status = 500;
                    res.set_content(error.dump(2), "application/json");
                }
            } catch (const std::exception& e) {
                json error = {{"error", std::string("Invalid request: ") + e.what()}};
                res.status = 400;
                res.set_content(error.dump(2), "application/json");
            }
        });
        
        // Stop profiling endpoint
        server.Post("/profile/stop", [this](const httplib::Request&, httplib::Response& res) {
            if (!active_analyzer) {
                json error = {{"error", "No active profiler"}};
                res.status = 400;
                res.set_content(error.dump(2), "application/json");
                return;
            }
            
            std::string result_dir = stop_profiling();
            json response = {
                {"message", "Profiling stopped"},
                {"output_directory", result_dir}
            };
            res.set_content(response.dump(2), "application/json");
        });
        
        // Get profiling status
        server.Get("/profile/status", [this](const httplib::Request&, httplib::Response& res) {
            json response = {
                {"running", active_analyzer != nullptr},
                {"analyzer", active_analyzer ? active_analyzer->get_name() : "none"},
                {"output_dir", output_dir}
            };
            res.set_content(response.dump(2), "application/json");
        });
        
        // List available results
        server.Get("/results", [this](const httplib::Request&, httplib::Response& res) {
            json results = list_results();
            res.set_content(results.dump(2), "application/json");
        });
        
        // Serve specific result file
        server.Get(R"(/results/([^/]+)/(.*))", [this](const httplib::Request& req, httplib::Response& res) {
            std::string result_dir = req.matches[1];
            std::string filename = req.matches[2];
            serve_result_file(result_dir, filename, res);
        });
        
        // Serve static files from results directory
        server.set_mount_point("/static", "./");
    }
    
    ProfilerArgs parse_profile_request(const json& request) {
        ProfilerArgs args;
        
        // Required fields
        args.analyzer_type = request.value("analyzer", "profile");
        args.duration = request.value("duration", 30);
        
        // Optional fields
        if (request.contains("pids")) {
            for (const auto& pid : request["pids"]) {
                args.pids.push_back(pid.get<pid_t>());
            }
        }
        
        if (request.contains("tids")) {
            for (const auto& tid : request["tids"]) {
                args.tids.push_back(tid.get<pid_t>());
            }
        }
        
        args.frequency = request.value("frequency", 49);
        args.min_block_us = request.value("min_block_us", 1000);
        args.cpu = request.value("cpu", -1);
        args.user_stacks_only = request.value("user_stacks_only", false);
        args.kernel_stacks_only = request.value("kernel_stacks_only", false);
        args.user_threads_only = request.value("user_threads_only", false);
        args.kernel_threads_only = request.value("kernel_threads_only", false);
        
        return args;
    }
    
    bool start_profiling(const ProfilerArgs& args) {
        current_args = args;
        
        // Create appropriate analyzer
        if (args.analyzer_type == "profile") {
            auto config = ArgsParser::create_profile_config(args);
            active_analyzer = std::make_unique<ProfileAnalyzer>(std::move(config));
        } else if (args.analyzer_type == "offcputime") {
            auto config = ArgsParser::create_offcpu_config(args);
            active_analyzer = std::make_unique<OffCPUTimeAnalyzer>(std::move(config));
        } else if (args.analyzer_type == "wallclock") {
            auto config = ArgsParser::create_wallclock_config(args);
            active_analyzer = std::make_unique<WallClockAnalyzer>(std::move(config));
        } else {
            return false;
        }
        
        if (!active_analyzer || !active_analyzer->start()) {
            active_analyzer.reset();
            return false;
        }
        
        return true;
    }
    
    std::string stop_profiling() {
        if (!active_analyzer) {
            return "";
        }
        
        // Generate output directory name
        auto now = std::time(nullptr);
        std::stringstream ss;
        ss << current_args.analyzer_type << "_profile_" << now;
        output_dir = ss.str();
        
        // Generate flamegraph
        FlamegraphGenerator fg_gen(output_dir, current_args.frequency, current_args.duration);
        
        if (current_args.analyzer_type == "wallclock") {
            auto* wallclock_analyzer = dynamic_cast<WallClockAnalyzer*>(active_analyzer.get());
            if (wallclock_analyzer) {
                auto per_thread_data = wallclock_analyzer->get_per_thread_flamegraphs();
                fg_gen.generate_files_from_flamegraphs(per_thread_data, "thread");
            }
        } else {
            std::unique_ptr<FlameGraphView> flamegraph;
            
            if (current_args.analyzer_type == "profile") {
                auto* profile_analyzer = dynamic_cast<ProfileAnalyzer*>(active_analyzer.get());
                if (profile_analyzer) {
                    flamegraph = profile_analyzer->get_flamegraph();
                }
            } else if (current_args.analyzer_type == "offcputime") {
                auto* offcpu_analyzer = dynamic_cast<OffCPUTimeAnalyzer*>(active_analyzer.get());
                if (offcpu_analyzer) {
                    flamegraph = offcpu_analyzer->get_flamegraph();
                }
            }
            
            if (flamegraph) {
                fg_gen.generate_files_from_flamegraph(*flamegraph, current_args.analyzer_type + "_profile");
            }
        }
        
        active_analyzer.reset();
        return output_dir;
    }
    
    json list_results() {
        json results = json::array();
        
        try {
            for (const auto& entry : fs::directory_iterator(".")) {
                if (entry.is_directory()) {
                    std::string dirname = entry.path().filename().string();
                    if (dirname.find("_profile_") != std::string::npos) {
                        json result = {
                            {"name", dirname},
                            {"path", entry.path().string()},
                            {"created", fs::last_write_time(entry).time_since_epoch().count()}
                        };
                        
                        // List files in the directory
                        json files = json::array();
                        for (const auto& file : fs::directory_iterator(entry)) {
                            if (file.is_regular_file()) {
                                files.push_back(file.path().filename().string());
                            }
                        }
                        result["files"] = files;
                        results.push_back(result);
                    }
                }
            }
        } catch (const std::exception& e) {
            // Return empty array if there's an error
        }
        
        return results;
    }
    
    void serve_result_file(const std::string& result_dir, const std::string& filename, httplib::Response& res) {
        fs::path filepath = fs::path(result_dir) / filename;
        
        if (!fs::exists(filepath) || !fs::is_regular_file(filepath)) {
            res.status = 404;
            json error = {{"error", "File not found"}};
            res.set_content(error.dump(2), "application/json");
            return;
        }
        
        std::ifstream file(filepath, std::ios::binary);
        if (!file) {
            res.status = 500;
            json error = {{"error", "Could not read file"}};
            res.set_content(error.dump(2), "application/json");
            return;
        }
        
        // Determine content type
        std::string ext = filepath.extension().string();
        std::string content_type = "application/octet-stream";
        if (ext == ".html") content_type = "text/html";
        else if (ext == ".svg") content_type = "image/svg+xml";
        else if (ext == ".json") content_type = "application/json";
        else if (ext == ".txt") content_type = "text/plain";
        
        // Read file content
        std::ostringstream ss;
        ss << file.rdbuf();
        res.set_content(ss.str(), content_type.c_str());
    }
    
    void run() {
        std::cout << "Starting BPF Profiler Server on " << host << ":" << port << std::endl;
        std::cout << "Available endpoints:" << std::endl;
        std::cout << "  GET  /                  - Server info" << std::endl;
        std::cout << "  GET  /health            - Health check" << std::endl;
        std::cout << "  POST /profile/start     - Start profiling" << std::endl;
        std::cout << "  POST /profile/stop      - Stop profiling" << std::endl;
        std::cout << "  GET  /profile/status    - Get profiling status" << std::endl;
        std::cout << "  GET  /results           - List available results" << std::endl;
        std::cout << "  GET  /results/{dir}/{file} - Get specific result file" << std::endl;
        std::cout << std::endl;
        
        server.listen(host.c_str(), port);
    }
    
    void stop() {
        running = false;
        server.stop();
        if (active_analyzer) {
            stop_profiling();
        }
    }
};

static ProfilerServer* server_instance = nullptr;

static void signal_handler(int signal) {
    if (server_instance) {
        std::cout << "\nShutting down server..." << std::endl;
        server_instance->stop();
    }
    exit(0);
}

int main(int argc, char** argv) {
    ProfilerArgs args = ArgsParser::parse(argc, argv);
    
    if (args.analyzer_type != "server") {
        std::cerr << "Server main should only be called with server subcommand" << std::endl;
        return 1;
    }
    
    // Set up signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    // Create and start server
    ProfilerServer server("0.0.0.0", 8080);
    server_instance = &server;
    
    server.run();
    
    return 0;
} 