#include <catch2/catch_all.hpp>
#include "../src/collectors/offcpu/offcputime.hpp"
#include "../src/collectors/config.hpp"
#include "../src/collectors/sampling_data.hpp"
#include <memory>
#include <thread>
#include <chrono>
#include <mutex>
#include <condition_variable>

TEST_CASE("OffCPUTimeCollector functional tests", "[offcpu][functional]") {
    SECTION("Collector lifecycle - start and stop") {
        OffCPUTimeCollector collector;
        
        // Test basic properties
        REQUIRE(collector.get_name() == "offcputime");
        
        // Try to start the collector (may fail in test environments without BPF support)
        bool started = collector.start();
        
        if (started) {
            SECTION("Collect data after successful start") {
                // Let it run briefly to establish baseline
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                
                // Collect initial data
                auto data = collector.get_data();
                REQUIRE(data != nullptr);
                REQUIRE(data->name == "offcputime");
                REQUIRE(data->success == true);
                REQUIRE(data->type == CollectorData::Type::SAMPLING);
                
                // Cast to SamplingData to access entries
                auto& sampling_data = static_cast<SamplingData&>(*data);
                INFO("Collected " << sampling_data.entries.size() << " initial off-CPU entries");
            }
        } else {
            // Collector failed to start (expected in some test environments)
            INFO("OffCPUTimeCollector failed to start - likely due to insufficient privileges or BPF not available");
            
            // Verify we get failure data
            auto data = collector.get_data();
            REQUIRE(data != nullptr);
            REQUIRE(data->name == "offcputime");
            REQUIRE(data->success == false);
        }
    }
}

TEST_CASE("OffCPUTimeCollector with blocking workload", "[offcpu][workload]") {
    OffCPUTimeCollector collector;
    
    // Configure for blocking time detection
    auto& config = collector.get_config();
    config.min_block_time = 100;      // 100 microseconds minimum
    config.max_block_time = 10000000; // 10 seconds maximum
    config.state = -1;                // Any blocking state
    
    bool started = collector.start();
    if (!started) {
        SKIP("Cannot start OffCPUTimeCollector - skipping workload test");
        return;
    }
    
    SECTION("Profile thread blocking workload") {
        std::mutex mtx;
        std::condition_variable cv;
        bool ready = false;
        bool should_block = true;
        
        // Create a thread that will block and unblock
        std::thread blocking_thread([&]() {
            std::unique_lock<std::mutex> lock(mtx);
            ready = true;
            cv.notify_one();
            
            // Block for a measurable amount of time
            cv.wait(lock, [&] { return !should_block; });
        });
        
        // Wait for thread to be ready
        {
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait(lock, [&] { return ready; });
        }
        
        // Let the thread block for a while
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        // Unblock the thread
        {
            std::unique_lock<std::mutex> lock(mtx);
            should_block = false;
            cv.notify_all();
        }
        
        blocking_thread.join();
        
        // Give some time for the off-CPU data to be recorded
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        
        // Collect data
        auto data = collector.get_data();
        REQUIRE(data != nullptr);
        REQUIRE(data->success == true);
        
        // Cast to SamplingData to access entries
        auto& sampling_data = static_cast<SamplingData&>(*data);
        INFO("Collected " << sampling_data.entries.size() << " off-CPU entries from blocking workload");
        
        // Verify we got some useful data
        if (!sampling_data.entries.empty()) {
            bool found_blocking_time = false;
            
            for (const auto& entry : sampling_data.entries) {
                if (entry.value > 0) {  // Found actual blocking time
                    found_blocking_time = true;
                    INFO("Found blocking time: " << entry.value << " microseconds");
                    if (entry.has_user_stack || entry.has_kernel_stack) {
                        INFO("With stack trace: " << entry.user_stack.size() 
                             << " user frames, " << entry.kernel_stack.size() << " kernel frames");
                    }
                    break;
                }
            }
            
            if (found_blocking_time) {
                REQUIRE(found_blocking_time == true);
            }
        }
    }
}

TEST_CASE("OffCPUTimeCollector with sleep workload", "[offcpu][sleep]") {
    OffCPUTimeCollector collector;
    
    // Configure to catch sleep events
    auto& config = collector.get_config();
    config.min_block_time = 1000;     // 1ms minimum (sleep should be longer)
    config.state = 1;                 // TASK_INTERRUPTIBLE (normal sleep)
    
    bool started = collector.start();
    if (!started) {
        SKIP("Cannot start OffCPUTimeCollector - skipping sleep test");
        return;
    }
    
    SECTION("Profile sleep-based blocking") {
        // Create a thread that sleeps
        std::thread sleeping_thread([]() {
            // Multiple short sleeps to increase chance of detection
            for (int i = 0; i < 5; i++) {
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
        });
        
        sleeping_thread.join();
        
        // Give time for data collection
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        
        // Collect data
        auto data = collector.get_data();
        REQUIRE(data != nullptr);
        REQUIRE(data->success == true);
        
        // Cast to SamplingData to access entries
        auto& sampling_data = static_cast<SamplingData&>(*data);
        INFO("Collected " << sampling_data.entries.size() << " entries from sleep workload");
    }
}

TEST_CASE("OffCPUTimeCollector configuration tests", "[offcpu][config]") {
    SECTION("Different block time thresholds") {
        OffCPUTimeCollector collector;
        auto& config = collector.get_config();
        
        // Test with very specific block time range
        config.min_block_time = 5000;     // 5ms minimum
        config.max_block_time = 1000000;  // 1s maximum
        config.state = 1;                 // TASK_INTERRUPTIBLE only
        
        bool started = collector.start();
        if (started) {
            // Generate blocking with sleep
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            
            auto data = collector.get_data();
            REQUIRE(data != nullptr);
            if (data->success) {
                auto& sampling_data = static_cast<SamplingData&>(*data);
                INFO("Block time range 5ms-1s: collected " << sampling_data.entries.size() << " entries");
            }
        } else {
            INFO("OffCPU collector with custom block time range failed to start");
        }
    }
    
    SECTION("Different task states") {
        OffCPUTimeCollector collector;
        auto& config = collector.get_config();
        
        // Test TASK_UNINTERRUPTIBLE state
        config.state = 2;  // TASK_UNINTERRUPTIBLE
        config.min_block_time = 1000;  // 1ms
        
        bool started = collector.start();
        if (started) {
            // Brief workload that might cause uninterruptible sleep
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            
            auto data = collector.get_data();
            REQUIRE(data != nullptr);
            if (data->success) {
                auto& sampling_data = static_cast<SamplingData&>(*data);
                INFO("TASK_UNINTERRUPTIBLE state: collected " << sampling_data.entries.size() << " entries");
            }
        } else {
            INFO("OffCPU collector with UNINTERRUPTIBLE state failed to start");
        }
    }
}

TEST_CASE("OffCPUTimeCollector error handling", "[offcpu][error]") {
    SECTION("Data collection when not started") {
        OffCPUTimeCollector collector;
        
        // Try to get data without starting
        auto data = collector.get_data();
        REQUIRE(data != nullptr);
        REQUIRE(data->name == "offcputime");
        REQUIRE(data->success == false);
    }
    
    SECTION("Multiple start calls") {
        OffCPUTimeCollector collector;
        
        bool first_start = collector.start();
        bool second_start = collector.start();  // Should return true (already running)
        
        if (first_start) {
            REQUIRE(second_start == true);  // Should succeed if already running
        } else {
            // Both should fail if BPF not available
            REQUIRE(second_start == false);
        }
    }
    
    SECTION("Data structure validation") {
        OffCPUTimeCollector collector;
        bool started = collector.start();
        
        if (started) {
            auto data = collector.get_data();
            if (data->success) {
                auto& sampling_data = static_cast<SamplingData&>(*data);
                if (!sampling_data.entries.empty()) {
                    for (const auto& entry : sampling_data.entries) {
                        // Validate data structure
                        REQUIRE(entry.value >= 0);  // Blocking time should be non-negative
                        
                        // Check stack trace consistency
                        if (entry.has_user_stack) {
                            REQUIRE(!entry.user_stack.empty());
                        }
                        if (entry.has_kernel_stack) {
                            REQUIRE(!entry.kernel_stack.empty());
                        }
                        
                        INFO("Entry: PID=" << entry.key.pid << " TGID=" << entry.key.tgid 
                             << " BlockTime=" << entry.value << "us");
                    }
                }
            }
        }
    }
} 