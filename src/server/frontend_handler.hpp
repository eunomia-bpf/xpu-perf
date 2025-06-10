#ifndef SERVER_FRONTEND_HANDLER_HPP
#define SERVER_FRONTEND_HANDLER_HPP

#include "../third_party/cpp-httplib/httplib.h"
#include <string>

namespace server {

class FrontendHandler {
private:
    std::string frontend_directory;

public:
    explicit FrontendHandler(const std::string& frontend_dir = "frontend");
    
    // Serve static files from frontend directory
    void handle_static_file(const httplib::Request& req, httplib::Response& res);
    
    // Serve asset files specifically
    void handle_asset_file(const httplib::Request& req, httplib::Response& res);
    
    // Serve the main index.html (SPA support)
    void handle_index(const httplib::Request& req, httplib::Response& res);

private:
    void serve_file(const std::string& filepath, httplib::Response& res);
    std::string get_content_type(const std::string& extension);
    bool is_safe_path(const std::string& path);
};

} // namespace server

#endif // SERVER_FRONTEND_HANDLER_HPP 