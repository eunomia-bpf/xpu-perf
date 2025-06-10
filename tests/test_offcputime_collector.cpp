#include <catch2/catch_all.hpp>
#include "../src/collectors/offcpu/offcputime.hpp"
#include "../src/collectors/config.hpp"
#include "../src/collectors/sampling_data.hpp"
#include <memory>
#include <stdexcept>
#include <climits>

TEST_CASE("OffCPUTimeConfig validation and defaults", "[offcpu][config]") {
    SECTION("Default configuration values") {
        OffCPUTimeConfig config;
        
        REQUIRE(config.min_block_time == 1);
        REQUIRE(config.max_block_time == (__u64)(-1));
        REQUIRE(config.state == -1);
    }

    SECTION("Configuration constructor validation") {
        // Test that constructor doesn't throw with default values
        REQUIRE_NOTHROW(OffCPUTimeConfig{});
        
        // Test custom configuration
        OffCPUTimeConfig config;
        config.min_block_time = 100;
        config.max_block_time = 50000;
        config.state = 1;
        
        REQUIRE(config.min_block_time == 100);
        REQUIRE(config.max_block_time == 50000);
        REQUIRE(config.state == 1);
    }
}

TEST_CASE("OffCPUTimeCollector basic functionality", "[offcpu][collector]") {
    SECTION("Constructor and destructor") {
        REQUIRE_NOTHROW([]() {
            OffCPUTimeCollector collector;
        }());
    }

    SECTION("Get collector name") {
        OffCPUTimeCollector collector;
        REQUIRE(collector.get_name() == "offcputime");
    }

    SECTION("Config access") {
        OffCPUTimeCollector collector;
        
        // Test const access
        const auto& const_config = collector.get_config();
        REQUIRE(const_config.min_block_time == 1);
        REQUIRE(const_config.max_block_time == (__u64)(-1));
        REQUIRE(const_config.state == -1);
        
        // Test mutable access
        auto& config = collector.get_config();
        config.min_block_time = 100;
        config.max_block_time = 50000;
        config.state = 2;
        
        REQUIRE(collector.get_config().min_block_time == 100);
        REQUIRE(collector.get_config().max_block_time == 50000);
        REQUIRE(collector.get_config().state == 2);
    }

    SECTION("Libbpf output buffer functionality") {
        OffCPUTimeCollector collector;
        
        // Initially empty
        REQUIRE(collector.get_libbpf_output().empty());
        
        // Test append functionality
        collector.append_libbpf_output("offcpu debug 1\n");
        collector.append_libbpf_output("offcpu debug 2\n");
        
        std::string expected = "offcpu debug 1\noffcpu debug 2\n";
        REQUIRE(collector.get_libbpf_output() == expected);
    }
}

TEST_CASE("OffCPUTimeCollector configuration scenarios", "[offcpu][collector]") {
    SECTION("Custom block time range") {
        OffCPUTimeCollector collector;
        auto& config = collector.get_config();
        
        config.min_block_time = 500;  // 500 microseconds
        config.max_block_time = 1000000;  // 1 second
        
        REQUIRE(config.min_block_time == 500);
        REQUIRE(config.max_block_time == 1000000);
    }

    SECTION("State filtering") {
        OffCPUTimeCollector collector;
        auto& config = collector.get_config();
        
        // TASK_INTERRUPTIBLE = 1
        config.state = 1;
        REQUIRE(config.state == 1);
        
        // TASK_UNINTERRUPTIBLE = 2  
        config.state = 2;
        REQUIRE(config.state == 2);
        
        // Any state = -1
        config.state = -1;
        REQUIRE(config.state == -1);
    }

    SECTION("Min/Max block time configuration") {
        OffCPUTimeCollector collector;
        auto& config = collector.get_config();
        
        // Test minimum value
        config.min_block_time = 1;
        REQUIRE(config.min_block_time == 1);
        
        // Test large value
        config.max_block_time = 1000000000ULL;
        REQUIRE(config.max_block_time == 1000000000ULL);
    }
}

TEST_CASE("OffCPUData and OffCPUEntry type aliases", "[offcpu][data]") {
    SECTION("OffCPUData is SamplingData") {
        auto data = std::make_unique<OffCPUData>("test_offcpu");
        REQUIRE(data->name == "test_offcpu");
        REQUIRE(data->success == true);
        REQUIRE(data->type == CollectorData::Type::SAMPLING);
        REQUIRE(data->entries.empty());
    }

    SECTION("OffCPUEntry is SamplingEntry") {
        OffCPUEntry entry;
        entry.value = 12345;  // microseconds blocked
        entry.has_user_stack = true;
        entry.has_kernel_stack = true;
        entry.user_stack = {0xabc, 0xdef};
        entry.kernel_stack = {0x123, 0x456, 0x789};
        
        REQUIRE(entry.value == 12345);
        REQUIRE(entry.has_user_stack == true);
        REQUIRE(entry.has_kernel_stack == true);
        REQUIRE(entry.user_stack.size() == 2);
        REQUIRE(entry.kernel_stack.size() == 3);
    }
}

TEST_CASE("OffCPUTimeCollector edge cases", "[offcpu][collector][edge]") {
    SECTION("Multiple collectors can coexist") {
        OffCPUTimeCollector collector1;
        OffCPUTimeCollector collector2;
        
        REQUIRE(collector1.get_name() == "offcputime");
        REQUIRE(collector2.get_name() == "offcputime");
        
        // Modify one config, other should be unaffected
        collector1.get_config().min_block_time = 100;
        collector2.get_config().min_block_time = 200;
        
        REQUIRE(collector1.get_config().min_block_time == 100);
        REQUIRE(collector2.get_config().min_block_time == 200);
    }

    SECTION("Libbpf output isolation between collectors") {
        OffCPUTimeCollector collector1;
        OffCPUTimeCollector collector2;
        
        collector1.append_libbpf_output("collector1 offcpu\n");
        collector2.append_libbpf_output("collector2 offcpu\n");
        
        REQUIRE(collector1.get_libbpf_output() == "collector1 offcpu\n");
        REQUIRE(collector2.get_libbpf_output() == "collector2 offcpu\n");
    }

    SECTION("Extreme block time values") {
        OffCPUTimeCollector collector;
        auto& config = collector.get_config();
        
        // Test with very small min_block_time
        config.min_block_time = 1;
        config.max_block_time = (__u64)(-1);  // Maximum possible value
        
        REQUIRE(config.min_block_time == 1);
        REQUIRE(config.max_block_time == (__u64)(-1));
        
        // Test with reasonable range
        config.min_block_time = 1000;  // 1ms
        config.max_block_time = 1000000000ULL;  // 1000 seconds
        
        REQUIRE(config.min_block_time == 1000);
        REQUIRE(config.max_block_time == 1000000000ULL);
    }
}

TEST_CASE("OffCPUTimeConfig state values", "[offcpu][config][state]") {
    SECTION("Valid state values") {
        OffCPUTimeConfig config;
        
        // Test all documented valid state values
        config.state = -1;  // Any state
        REQUIRE(config.state == -1);
        
        config.state = 0;   // TASK_RUNNING (though not typically useful)
        REQUIRE(config.state == 0);
        
        config.state = 1;   // TASK_INTERRUPTIBLE
        REQUIRE(config.state == 1);
        
        config.state = 2;   // TASK_UNINTERRUPTIBLE
        REQUIRE(config.state == 2);
    }

    SECTION("State configuration semantics") {
        OffCPUTimeConfig config;
        
        // Default should be "any state"
        REQUIRE(config.state == -1);
        
        // Common blocking states
        config.state = 1;  // Interruptible sleep
        REQUIRE(config.state == 1);
        
        config.state = 2;  // Uninterruptible sleep  
        REQUIRE(config.state == 2);
    }
} 