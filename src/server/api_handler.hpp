#ifndef SERVER_API_HANDLER_HPP
#define SERVER_API_HANDLER_HPP

#include "../third_party/cpp-httplib/httplib.h"
#include <ctime>

namespace server {

class StatusHandler {
private:
    time_t start_time;

public:
    StatusHandler();
    void handle(const httplib::Request& req, httplib::Response& res);
};

// Future API handlers can be added here
// class ProfileHandler, etc.

} // namespace server

#endif // SERVER_API_HANDLER_HPP 