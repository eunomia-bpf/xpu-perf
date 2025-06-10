#include "profile_server.hpp"
#include <httplib.h>
#include <nlohmann/json.hpp>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <sstream>

using json = nlohmann::json;
namespace fs = std::filesystem;

class ProfileServerImpl {
private:
    httplib::Server server;
    std::string host;
    int port;
    bool running;
    
public:
    ProfileServerImpl(const std::string& host, int port) 
        : host(host), port(port), running(false) {
        setup_routes();
    }
    
    bool start() {
        if (running) return false;
        
        running = true;
        return server.listen(host.c_str(), port);
    }
    
    void stop() {
        if (running) {
            running = false;
            server.stop();
        }
    }
    
    bool is_running() const {
        return running;
    }
    
private:
    void setup_routes() {
        // Enable CORS
        server.set_pre_routing_handler([](const httplib::Request&, httplib::Response& res) {
            res.set_header("Access-Control-Allow-Origin", "*");
            res.set_header("Access-Control-Allow-Methods", "GET, OPTIONS");
            res.set_header("Access-Control-Allow-Headers", "Content-Type");
            return httplib::Server::HandlerResponse::Unhandled;
        });
        
        // Handle OPTIONS for CORS
        server.Options(".*", [](const httplib::Request&, httplib::Response&) {
            return;
        });
        
        // Root endpoint - list files
        server.Get("/", [this](const httplib::Request&, httplib::Response& res) {
            handle_root_request(res);
        });
        
        // Status endpoint
        server.Get("/api/status", [this](const httplib::Request&, httplib::Response& res) {
            handle_status_request(res);
        });
        
        // Serve static files
        server.Get(R"(/files/(.*))", [this](const httplib::Request& req, httplib::Response& res) {
            std::string filepath = req.matches[1];
            serve_file(filepath, res);
        });
    }
    
    void handle_root_request(httplib::Response& res) {
        std::ostringstream html;
        html << "<!DOCTYPE html>\n";
        html << "<html><head><title>BPF Profiler Server</title>";
        html << "<style>body{font-family:Arial,sans-serif;margin:40px;} ";
        html << "h1{color:#333;} ul{list-style-type:none;padding:0;} ";
        html << "li{padding:8px;border-bottom:1px solid #eee;} ";
        html << "a{text-decoration:none;color:#0066cc;} ";
        html << "a:hover{text-decoration:underline;}</style></head>";
        html << "<body><h1>BPF Profiler Server</h1>";
        html << "<h2>Available Files</h2><ul>";
        
        try {
            for (const auto& entry : fs::directory_iterator(".")) {
                if (entry.is_regular_file()) {
                    std::string filename = entry.path().filename().string();
                    html << "<li><a href=\"/files/" << filename << "\">" << filename << "</a></li>";
                }
            }
        } catch (const std::exception& e) {
            html << "<li>Error reading directory: " << e.what() << "</li>";
        }
        
        html << "</ul>";
        html << "<h2>API Endpoints</h2>";
        html << "<ul>";
        html << "<li><a href=\"/api/status\">GET /api/status</a> - Server status</li>";
        html << "</ul>";
        html << "</body></html>";
        
        res.set_content(html.str(), "text/html");
    }
    
    void handle_status_request(httplib::Response& res) {
        json status = {
            {"status", "running"},
            {"host", host},
            {"port", port},
            {"version", "1.0"},
            {"uptime_seconds", std::time(nullptr) - start_time}
        };
        
        res.set_content(status.dump(2), "application/json");
    }
    
    void serve_file(const std::string& filename, httplib::Response& res) {
        fs::path filepath = fs::current_path() / filename;
        
        // Security check - don't allow path traversal
        if (filename.find("..") != std::string::npos || filename.find("/") != std::string::npos) {
            res.status = 403;
            res.set_content("Forbidden", "text/plain");
            return;
        }
        
        if (!fs::exists(filepath) || !fs::is_regular_file(filepath)) {
            res.status = 404;
            res.set_content("File not found", "text/plain");
            return;
        }
        
        std::ifstream file(filepath, std::ios::binary);
        if (!file) {
            res.status = 500;
            res.set_content("Could not read file", "text/plain");
            return;
        }
        
        // Determine content type
        std::string ext = filepath.extension().string();
        std::string content_type = "application/octet-stream";
        if (ext == ".html") content_type = "text/html";
        else if (ext == ".css") content_type = "text/css";
        else if (ext == ".js") content_type = "application/javascript";
        else if (ext == ".json") content_type = "application/json";
        else if (ext == ".txt") content_type = "text/plain";
        else if (ext == ".svg") content_type = "image/svg+xml";
        else if (ext == ".png") content_type = "image/png";
        else if (ext == ".jpg" || ext == ".jpeg") content_type = "image/jpeg";
        
        std::ostringstream buffer;
        buffer << file.rdbuf();
        res.set_content(buffer.str(), content_type.c_str());
    }
    
    static std::time_t start_time;
};

// Static member definition
std::time_t ProfileServerImpl::start_time = std::time(nullptr);

// ProfileServer public interface implementation
ProfileServer::ProfileServer(const std::string& host, int port) 
    : host(host), port(port), running(false), impl(nullptr) {}

ProfileServer::~ProfileServer() {
    stop();
}

bool ProfileServer::start() {
    if (running || impl) {
        return false;  // Already running
    }
    
    impl = std::make_unique<ProfileServerImpl>(host, port);
    
    std::cout << "Starting BPF Profiler Server on " << host << ":" << port << std::endl;
    std::cout << "Available endpoints:" << std::endl;
    std::cout << "  GET  /           - List files in current directory" << std::endl;
    std::cout << "  GET  /api/status - Server status" << std::endl;
    std::cout << std::endl;
    
    running = true;
    
    // server.listen() is blocking, so this will block until server stops
    bool result = impl->start();
    
    // When we get here, server has stopped
    running = false;
    impl.reset();
    
    return result;
}

void ProfileServer::stop() {
    if (running && impl) {
        running = false;
        impl->stop();
        impl.reset();
    }
} 