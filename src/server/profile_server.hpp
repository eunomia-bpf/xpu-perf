#ifndef PROFILE_SERVER_HPP
#define PROFILE_SERVER_HPP

#include <string>
#include <memory>

class ProfileServer {
private:
    std::string host;
    int port;
    bool running;
    
public:
    ProfileServer(const std::string& host = "0.0.0.0", int port = 8080);
    ~ProfileServer();
    
    // Start the server
    bool start();
    
    // Stop the server
    void stop();
    
    // Check if server is running
    bool is_running() const { return running; }
    
private:
    // Setup HTTP routes
    void setup_routes();
    
    // Route handlers
    void handle_root(const std::string& request_path, std::string& response);
    void handle_status(std::string& response);
};

#endif // PROFILE_SERVER_HPP 