#ifndef SERVER_API_HANDLER_HPP
#define SERVER_API_HANDLER_HPP

#include "../third_party/cpp-httplib/httplib.h"
#include "../analyzers/profile_analyzer.hpp"
#include "../analyzers/offcputime_analyzer.hpp"
#include "../analyzers/wallclock_analyzer.hpp"
#include <ctime>
#include <memory>
#include <map>
#include <string>

namespace server {

class StatusHandler {
private:
    time_t start_time;

public:
    StatusHandler();
    void handle(const httplib::Request& req, httplib::Response& res);
};

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

// Profile analyzer handler
class ProfileAnalyzerHandler : public IAnalyzerHandler {
private:
    std::map<std::string, std::unique_ptr<ProfileAnalyzer>> active_sessions_;
    
public:
    std::string get_analyzer_type() const override { return "profile"; }
    void handle_start(const httplib::Request& req, httplib::Response& res) override;
    void handle_status(const httplib::Request& req, httplib::Response& res) override;
    void handle_views(const httplib::Request& req, httplib::Response& res) override;
    void handle_stop(const httplib::Request& req, httplib::Response& res) override;
    void handle_delete(const httplib::Request& req, httplib::Response& res) override;

private:
    std::string generate_session_id();
    std::unique_ptr<ProfileAnalyzerConfig> parse_config(const std::string& json_body);
};

// Registry for analyzer handlers
class AnalyzerHandlerRegistry {
private:
    std::map<std::string, std::unique_ptr<IAnalyzerHandler>> handlers_;
    
public:
    void register_handler(std::unique_ptr<IAnalyzerHandler> handler);
    IAnalyzerHandler* get_handler(const std::string& analyzer_type);
    std::vector<std::string> get_supported_types() const;
};

// Future API handlers can be added here
// class OffCPUAnalyzerHandler, WallClockAnalyzerHandler, etc.

} // namespace server

#endif // SERVER_API_HANDLER_HPP 