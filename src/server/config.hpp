#ifndef SERVER_CONFIG_HPP
#define SERVER_CONFIG_HPP

#include <string>

namespace server {

struct ServerConfig {
    std::string host = "0.0.0.0";
    int port = 8080;
    std::string log_level = "info";
    bool enable_cors = true;
    std::string frontend_directory = "frontend";
    
    // Future extensibility
    int max_connections = 100;
    int timeout_seconds = 30;
};

} // namespace server

#endif // SERVER_CONFIG_HPP 