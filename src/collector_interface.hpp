#ifndef __COLLECTOR_INTERFACE_HPP
#define __COLLECTOR_INTERFACE_HPP

#include <string>
#include <memory>

struct CollectorData {
    std::string name;
    std::string output;
    bool success;
    
    CollectorData(const std::string& n = "", const std::string& o = "", bool s = false)
        : name(n), output(o), success(s) {}
};

class ICollector {
public:
    virtual ~ICollector() = default;
    
    // Start the collector
    virtual bool start() = 0;
    
    // Get collected data
    virtual CollectorData get_data() = 0;
    
    // Get collector name
    virtual std::string get_name() const = 0;
};

#endif /* __COLLECTOR_INTERFACE_HPP */ 