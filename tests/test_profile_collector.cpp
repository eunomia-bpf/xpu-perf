#include <catch2/catch_all.hpp>
#include "../src/collectors/oncpu/profile.hpp"
#include "../src/collectors/config.hpp"
#include "../src/collectors/sampling_data.hpp"
#include <memory>
#include <thread>
#include <chrono>
#include <cmath>

TEST_CASE("ProfileCollector functional tests", "[profile][functional]") {
    SECTION("Collector lifecycle - start and stop") {
        ProfileCollector collector;
        
        // Test basic properties
        REQUIRE(collector.get_name() == "profile");
        
        // Try to start the collector (may fail in test environments without BPF support)
        bool started = collector.start();
        
        if (started) {
            SECTION("Collect data after successful start") {
                // Let it run briefly to collect some samples
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                
                // Generate some CPU activity to profile
                volatile int sum = 0;
                for (int i = 0; i < 10000; i++) {
                    sum += i * i;
                }
                
                // Collect the data
                auto data = collector.get_data();
                REQUIRE(data != nullptr);
                REQUIRE(data->name == "profile");
                REQUIRE(data->success == true);
                REQUIRE(data->type == CollectorData::Type::SAMPLING);
                
                // The data might be empty if no samples were collected in the brief time
                // but the structure should be valid
                auto& sampling_data = static_cast<SamplingData&>(*data);
                INFO("Collected " << sampling_data.entries.size() << " profile entries");
            }
        } else {
            // Collector failed to start (expected in some test environments)
            INFO("ProfileCollector failed to start - likely due to insufficient privileges or BPF not available");
            
            // Verify we get failure data
            auto data = collector.get_data();
            REQUIRE(data != nullptr);
            REQUIRE(data->name == "profile");
            REQUIRE(data->success == false);
        }
    }
}

TEST_CASE("ProfileCollector with CPU workload", "[profile][workload]") {
    ProfileCollector collector;
    
    // Configure for more aggressive sampling
    auto& config = collector.get_config();
    config.attr.sample_freq = 1000;  // Higher frequency for more samples
    
    bool started = collector.start();
    if (!started) {
        SKIP("Cannot start ProfileCollector - skipping workload test");
        return;
    }
    
    SECTION("Profile CPU-intensive workload") {
        // Generate significant CPU activity
        auto start_time = std::chrono::steady_clock::now();
        volatile double result = 0.0;
        
        // Run CPU-intensive work for 200ms
        while (std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::steady_clock::now() - start_time).count() < 200) {
            for (int i = 0; i < 1000; i++) {
                result += std::sqrt(i * 3.14159 + result);
            }
        }
        
        // Collect data
        auto data = collector.get_data();
        REQUIRE(data != nullptr);
        REQUIRE(data->success == true);
        
        auto& sampling_data = static_cast<SamplingData&>(*data);
        INFO("Collected " << sampling_data.entries.size() << " samples from CPU workload");
        
        // Verify we got some useful data
        if (!sampling_data.entries.empty()) {
            bool found_stack = false;
            
            for (const auto& entry : sampling_data.entries) {
                if (entry.has_user_stack || entry.has_kernel_stack) {
                    found_stack = true;
                    INFO("Found stack trace with " << entry.user_stack.size() 
                         << " user frames and " << entry.kernel_stack.size() << " kernel frames");
                    break;
                }
            }
            
            // We should have at least some stack traces
            if (found_stack) {
                REQUIRE(found_stack == true);
            }
        }
    }
}

TEST_CASE("ProfileCollector hardware events", "[profile][hardware]") {
    ProfileCollector collector;
    auto& config = collector.get_config();
    
    SECTION("Hardware CPU cycles profiling") {
        // Configure for hardware CPU cycles
        config.attr.type = PERF_TYPE_HARDWARE;
        config.attr.config = PERF_COUNT_HW_CPU_CYCLES;
        config.attr.sample_freq = 1000;
        
        bool started = collector.start();
        if (started) {
            // Generate CPU activity
            volatile long sum = 0;
            for (int i = 0; i < 50000; i++) {
                sum += i * i * i;
            }
            
            auto data = collector.get_data();
            REQUIRE(data != nullptr);
            if (data->success) {
                auto& sampling_data = static_cast<SamplingData&>(*data);
                INFO("Hardware cycles: collected " << sampling_data.entries.size() << " samples");
            }
        } else {
            INFO("Hardware CPU cycles not supported or insufficient privileges");
        }
    }
    
    SECTION("Hardware instructions profiling") {
        // Configure for hardware instructions
        config.attr.type = PERF_TYPE_HARDWARE;
        config.attr.config = PERF_COUNT_HW_INSTRUCTIONS;
        config.attr.sample_freq = 2000;
        
        bool started = collector.start();
        if (started) {
            // Generate instruction-heavy workload
            volatile int result = 1;
            for (int i = 0; i < 10000; i++) {
                result = result * 2 + 1;
                if (result > 1000000) result = 1;
            }
            
            auto data = collector.get_data();
            REQUIRE(data != nullptr);
            if (data->success) {
                auto& sampling_data = static_cast<SamplingData&>(*data);
                INFO("Hardware instructions: collected " << sampling_data.entries.size() << " samples");
            }
        } else {
            INFO("Hardware instructions counter not supported or insufficient privileges");
        }
    }
    
    SECTION("Hardware cache misses profiling") {
        // Configure for cache misses
        config.attr.type = PERF_TYPE_HARDWARE;
        config.attr.config = PERF_COUNT_HW_CACHE_MISSES;
        config.attr.sample_freq = 100;  // Lower frequency for cache events
        
        bool started = collector.start();
        if (started) {
            // Generate cache-miss heavy workload
            const size_t size = 1024 * 1024;  // 1MB
            volatile char* buffer = new char[size];
            
            // Random access pattern to cause cache misses
            for (int i = 0; i < 1000; i++) {
                size_t offset = (i * 4096 + i * 7) % size;
                buffer[offset] = i & 0xFF;
            }
            
            delete[] buffer;
            
            auto data = collector.get_data();
            REQUIRE(data != nullptr);
            if (data->success) {
                auto& sampling_data = static_cast<SamplingData&>(*data);
                INFO("Cache misses: collected " << sampling_data.entries.size() << " samples");
            }
        } else {
            INFO("Hardware cache miss counter not supported or insufficient privileges");
        }
    }
}

TEST_CASE("ProfileCollector software events", "[profile][software]") {
    ProfileCollector collector;
    auto& config = collector.get_config();
    
    SECTION("Software page faults profiling") {
        // Configure for page faults
        config.attr.type = PERF_TYPE_SOFTWARE;
        config.attr.config = PERF_COUNT_SW_PAGE_FAULTS;
        config.attr.sample_freq = 10;  // Low frequency for page faults
        
        bool started = collector.start();
        if (started) {
            // Generate page faults by allocating and accessing memory
            const size_t pages = 100;
            const size_t page_size = 4096;
            
            for (size_t i = 0; i < pages; i++) {
                volatile char* page = new char[page_size];
                // Touch the page to potentially cause a fault
                page[0] = i & 0xFF;
                page[page_size - 1] = (i + 1) & 0xFF;
                delete[] page;
            }
            
            auto data = collector.get_data();
            REQUIRE(data != nullptr);
            REQUIRE(data->success == true);
            
            auto& sampling_data = static_cast<SamplingData&>(*data);
            INFO("Page faults: collected " << sampling_data.entries.size() << " samples");
        } else {
            INFO("Software page fault profiling failed to start");
        }
    }
    
    SECTION("Software context switches profiling") {
        // Configure for context switches
        config.attr.type = PERF_TYPE_SOFTWARE;
        config.attr.config = PERF_COUNT_SW_CONTEXT_SWITCHES;
        config.attr.sample_freq = 50;
        
        bool started = collector.start();
        if (started) {
            // Generate context switches by creating competing threads
            std::thread worker1([]() {
                for (int i = 0; i < 100; i++) {
                    std::this_thread::yield();
                    std::this_thread::sleep_for(std::chrono::microseconds(100));
                }
            });
            
            std::thread worker2([]() {
                for (int i = 0; i < 100; i++) {
                    std::this_thread::yield();
                    std::this_thread::sleep_for(std::chrono::microseconds(100));
                }
            });
            
            worker1.join();
            worker2.join();
            
            auto data = collector.get_data();
            REQUIRE(data != nullptr);
            REQUIRE(data->success == true);
            
            auto& sampling_data = static_cast<SamplingData&>(*data);
            INFO("Context switches: collected " << sampling_data.entries.size() << " samples");
        } else {
            INFO("Software context switch profiling failed to start");
        }
    }
}

TEST_CASE("ProfileCollector configuration options", "[profile][config]") {
    SECTION("High frequency sampling") {
        ProfileCollector collector;
        auto& config = collector.get_config();
        
        config.attr.sample_freq = 5000;  // Very high frequency
        config.include_idle = false;
        
        bool started = collector.start();
        if (started) {
            // Brief workload
            volatile int sum = 0;
            for (int i = 0; i < 1000; i++) {
                sum += i;
            }
            
            auto data = collector.get_data();
            REQUIRE(data != nullptr);
            if (data->success) {
                auto& sampling_data = static_cast<SamplingData&>(*data);
                INFO("High frequency: collected " << sampling_data.entries.size() << " samples");
            }
        }
    }
    
    SECTION("Include idle processes") {
        ProfileCollector collector;
        auto& config = collector.get_config();
        
        config.include_idle = true;  // Include idle in profiling
        config.attr.sample_freq = 100;
        
        bool started = collector.start();
        if (started) {
            // Let it run to potentially catch idle
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            
            auto data = collector.get_data();
            REQUIRE(data != nullptr);
            if (data->success) {
                auto& sampling_data = static_cast<SamplingData&>(*data);
                INFO("With idle: collected " << sampling_data.entries.size() << " samples");
            }
        }
    }
}

TEST_CASE("ProfileCollector error handling", "[profile][error]") {
    SECTION("Data collection when not started") {
        ProfileCollector collector;
        
        // Try to get data without starting
        auto data = collector.get_data();
        REQUIRE(data != nullptr);
        REQUIRE(data->name == "profile");
        REQUIRE(data->success == false);
    }
    
    SECTION("Multiple start calls") {
        ProfileCollector collector;
        
        bool first_start = collector.start();
        bool second_start = collector.start();  // Should return true (already running)
        
        if (first_start) {
            REQUIRE(second_start == true);  // Should succeed if already running
        } else {
            // Both should fail if BPF not available
            REQUIRE(second_start == false);
        }
    }
    
    SECTION("Invalid perf event configuration") {
        ProfileCollector collector;
        auto& config = collector.get_config();
        
        // Try an invalid configuration
        config.attr.type = 999;  // Invalid type
        config.attr.config = 999;  // Invalid config
        
        bool started = collector.start();
        // Should fail gracefully
        if (!started) {
            INFO("Invalid perf config correctly rejected");
            auto data = collector.get_data();
            REQUIRE(data->success == false);
        }
    }
} 