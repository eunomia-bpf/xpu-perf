#ifndef PROFILE_SERVER_HPP
#define PROFILE_SERVER_HPP

#include "config.hpp"
#include "api_handler.hpp"
#include "../third_party/cpp-httplib/httplib.h"
#include <memory>

namespace server {

class ProfileServer {
private:
    ServerConfig config;
    std::unique_ptr<httplib::Server> server;
    bool running;
    
    // Handlers
    std::unique_ptr<StatusHandler> status_handler;
    std::unique_ptr<AnalyzerHandlerRegistry> analyzer_registry;
    
public:
    explicit ProfileServer(const ServerConfig& config = ServerConfig{});
    ~ProfileServer();
    
    // Start the server
    bool start();
    
    // Stop the server
    void stop();
    
    // Check if server is running
    bool is_running() const { return running; }
    
private:
    void setup_handlers();
    void setup_routes();
    void setup_middleware();
    void register_default_analyzers();
};

} // namespace server

#endif // PROFILE_SERVER_HPP 