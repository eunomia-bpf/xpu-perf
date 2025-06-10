#include "api_handler.hpp"
#include "../third_party/json/single_include/nlohmann/json.hpp"
#include "../third_party/spdlog/include/spdlog/spdlog.h"

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

} // namespace server 