#ifndef __COLLECTOR_INTERFACE_HPP
#define __COLLECTOR_INTERFACE_HPP

#include <string>
#include <memory>

// Abstract base class for collector data
class CollectorData {
public:
    virtual ~CollectorData() = default;
    
    // Get collector name
    virtual std::string get_name() const = 0;
    
    // Check if collection was successful
    virtual bool is_success() const = 0;
    
    // Get formatted output as string (optional, for backward compatibility)
    virtual std::string get_output() const { return ""; }
    
    // Get data type identifier
    virtual std::string get_type() const = 0;

protected:
    CollectorData() = default;
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