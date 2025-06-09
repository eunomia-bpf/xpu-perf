#ifndef __COLLECTOR_INTERFACE_HPP
#define __COLLECTOR_INTERFACE_HPP

#include <string>
#include <memory>
#include <chrono>

// Abstract base class for collector data
class CollectorData {
public:
    virtual ~CollectorData() = default;

    std::string name;
    bool success;
    enum class Type {
        SAMPLING,
    };
    Type type;
    unsigned long long timestamp;

    CollectorData(const std::string& name = "", bool success = false, Type type = Type::SAMPLING)
        : name(name), success(success), type(type) {
            // get current timestamp
            timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()
            ).count();
        }
};

class ICollector {
public:
    virtual ~ICollector() = default;
    
    // Start the collector
    virtual bool start() = 0;
    
    // Get collected data as abstract CollectorData
    virtual std::unique_ptr<CollectorData> get_data() = 0;
    
    // Get collector name
    virtual std::string get_name() const = 0;
};

#endif /* __COLLECTOR_INTERFACE_HPP */ 