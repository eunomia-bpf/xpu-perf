#include "frontend_handler.hpp"
#include "../third_party/spdlog/include/spdlog/spdlog.h"
#include <filesystem>
#include <fstream>
#include <sstream>

namespace fs = std::filesystem;

namespace server {

FrontendHandler::FrontendHandler(const std::string& frontend_dir) 
    : frontend_directory(frontend_dir) {}

void FrontendHandler::handle_static_file(const httplib::Request& req, httplib::Response& res) {
    std::string requested_path = req.matches[1];
    spdlog::debug("Static file request: {}", requested_path);
    
    if (!is_safe_path(requested_path)) {
        spdlog::warn("Unsafe path blocked: {}", requested_path);
        res.status = 403;
        res.set_content("Forbidden", "text/plain");
        return;
    }
    
    fs::path full_path = fs::path(frontend_directory) / requested_path;
    serve_file(full_path.string(), res);
}

void FrontendHandler::handle_index(const httplib::Request& req, httplib::Response& res) {
    spdlog::debug("Index page request");
    fs::path index_path = fs::path(frontend_directory) / "index.html";
    serve_file(index_path.string(), res);
}

void FrontendHandler::serve_file(const std::string& filepath, httplib::Response& res) {
    fs::path file_path(filepath);
    
    if (!fs::exists(file_path) || !fs::is_regular_file(file_path)) {
        spdlog::debug("File not found: {}", filepath);
        res.status = 404;
        res.set_content("File not found", "text/plain");
        return;
    }
    
    std::ifstream file(file_path, std::ios::binary);
    if (!file) {
        spdlog::error("Could not read file: {}", filepath);
        res.status = 500;
        res.set_content("Internal server error", "text/plain");
        return;
    }
    
    std::string content_type = get_content_type(file_path.extension().string());
    
    std::ostringstream buffer;
    buffer << file.rdbuf();
    res.set_content(buffer.str(), content_type);
    
    spdlog::debug("Served file: {} ({})", filepath, content_type);
}

std::string FrontendHandler::get_content_type(const std::string& extension) {
    if (extension == ".html") return "text/html";
    else if (extension == ".css") return "text/css";
    else if (extension == ".js") return "application/javascript";
    else if (extension == ".json") return "application/json";
    else if (extension == ".png") return "image/png";
    else if (extension == ".jpg" || extension == ".jpeg") return "image/jpeg";
    else if (extension == ".svg") return "image/svg+xml";
    else if (extension == ".ico") return "image/x-icon";
    else if (extension == ".woff" || extension == ".woff2") return "font/woff";
    return "application/octet-stream";
}

bool FrontendHandler::is_safe_path(const std::string& path) {
    // Reject paths with .. or absolute paths
    return path.find("..") == std::string::npos && 
           path[0] != '/' && 
           path.find("\\") == std::string::npos;
}

} // namespace server 