#include <catch2/catch_all.hpp>
#include "../src/collectors/oncpu/profile.hpp"
#include "../src/collectors/config.hpp"
#include "../src/collectors/sampling_data.hpp"
#include <memory>
#include <stdexcept>
#include <climits>

TEST_CASE("ProfileConfig validation and defaults", "[profile][config]") {
    SECTION("Default configuration values") {
        ProfileConfig config;
        
        REQUIRE(config.include_idle == false);
        REQUIRE(config.attr.type == PERF_TYPE_SOFTWARE);
        REQUIRE(config.attr.config == PERF_COUNT_SW_CPU_CLOCK);
        REQUIRE(config.attr.sample_freq == 50);
        REQUIRE(config.attr.freq == 1);
        
        // Base Config defaults that should exist
        REQUIRE(config.pids.empty());
        REQUIRE(config.tids.empty());
        REQUIRE(config.verbose == false);
        REQUIRE(config.folded == true);
        REQUIRE(config.cpu == -1);
    }

    SECTION("Configuration constructor validation") {
        // Test that constructor doesn't throw with default values
        REQUIRE_NOTHROW(ProfileConfig{});
        
        // Test custom configuration
        ProfileConfig config;
        config.attr.sample_freq = 100;
        REQUIRE(config.attr.sample_freq == 100);
    }
}

TEST_CASE("ProfileCollector basic functionality", "[profile][collector]") {
    SECTION("Constructor and destructor") {
        REQUIRE_NOTHROW([]() {
            ProfileCollector collector;
        }());
    }

    SECTION("Get collector name") {
        ProfileCollector collector;
        REQUIRE(collector.get_name() == "profile");
    }

    SECTION("Config access") {
        ProfileCollector collector;
        
        // Test const access
        const auto& const_config = collector.get_config();
        REQUIRE(const_config.attr.sample_freq == 50);
        
        // Test mutable access
        auto& config = collector.get_config();
        config.attr.sample_freq = 200;
        REQUIRE(collector.get_config().attr.sample_freq == 200);
    }

    SECTION("Libbpf output buffer functionality") {
        ProfileCollector collector;
        
        // Initially empty
        REQUIRE(collector.get_libbpf_output().empty());
        
        // Test append functionality
        collector.append_libbpf_output("test output 1\n");
        collector.append_libbpf_output("test output 2\n");
        
        std::string expected = "test output 1\ntest output 2\n";
        REQUIRE(collector.get_libbpf_output() == expected);
    }
}

TEST_CASE("ProfileCollector configuration scenarios", "[profile][collector]") {
    SECTION("PID filtering configuration") {
        ProfileCollector collector;
        auto& config = collector.get_config();
        
        config.pids = {1234, 5678};
        REQUIRE(config.pids.size() == 2);
        REQUIRE(config.pids[0] == 1234);
        REQUIRE(config.pids[1] == 5678);
    }

    SECTION("TID filtering configuration") {
        ProfileCollector collector;
        auto& config = collector.get_config();
        
        config.tids = {9999, 8888, 7777};
        REQUIRE(config.tids.size() == 3);
        REQUIRE(config.tids[0] == 9999);
        REQUIRE(config.tids[1] == 8888);
        REQUIRE(config.tids[2] == 7777);
    }

    SECTION("Custom sampling frequency") {
        ProfileCollector collector;
        auto& config = collector.get_config();
        
        config.attr.sample_freq = 999;
        REQUIRE(config.attr.sample_freq == 999);
    }

    SECTION("CPU configuration") {
        ProfileCollector collector;
        auto& config = collector.get_config();
        
        config.cpu = 2;
        REQUIRE(config.cpu == 2);
    }

    SECTION("Include idle configuration") {
        ProfileCollector collector;
        auto& config = collector.get_config();
        
        config.include_idle = true;
        REQUIRE(config.include_idle == true);
    }
}

TEST_CASE("ProfileData and ProfileEntry type aliases", "[profile][data]") {
    SECTION("ProfileData is SamplingData") {
        auto data = std::make_unique<ProfileData>("test_profile");
        REQUIRE(data->name == "test_profile");
        REQUIRE(data->success == true);
        REQUIRE(data->type == CollectorData::Type::SAMPLING);
        REQUIRE(data->entries.empty());
    }

    SECTION("ProfileEntry is SamplingEntry") {
        ProfileEntry entry;
        entry.value = 42;
        entry.has_user_stack = true;
        entry.has_kernel_stack = false;
        entry.user_stack = {0x1234, 0x5678};
        entry.kernel_stack = {};
        
        REQUIRE(entry.value == 42);
        REQUIRE(entry.has_user_stack == true);
        REQUIRE(entry.has_kernel_stack == false);
        REQUIRE(entry.user_stack.size() == 2);
        REQUIRE(entry.kernel_stack.empty());
    }
}

TEST_CASE("ProfileCollector edge cases", "[profile][collector][edge]") {
    SECTION("Multiple collectors can coexist") {
        ProfileCollector collector1;
        ProfileCollector collector2;
        
        REQUIRE(collector1.get_name() == "profile");
        REQUIRE(collector2.get_name() == "profile");
        
        // Modify one config, other should be unaffected
        collector1.get_config().attr.sample_freq = 100;
        collector2.get_config().attr.sample_freq = 200;
        
        REQUIRE(collector1.get_config().attr.sample_freq == 100);
        REQUIRE(collector2.get_config().attr.sample_freq == 200);
    }

    SECTION("Libbpf output isolation between collectors") {
        ProfileCollector collector1;
        ProfileCollector collector2;
        
        collector1.append_libbpf_output("collector1 output\n");
        collector2.append_libbpf_output("collector2 output\n");
        
        REQUIRE(collector1.get_libbpf_output() == "collector1 output\n");
        REQUIRE(collector2.get_libbpf_output() == "collector2 output\n");
    }
}

TEST_CASE("ProfileConfig perf_event_attr configuration", "[profile][config][perf]") {
    SECTION("Default perf_event_attr values") {
        ProfileConfig config;
        
        REQUIRE(config.attr.type == PERF_TYPE_SOFTWARE);
        REQUIRE(config.attr.config == PERF_COUNT_SW_CPU_CLOCK);
        REQUIRE(config.attr.sample_freq == 50);
        REQUIRE(config.attr.freq == 1);
    }

    SECTION("Custom perf_event_attr configuration") {
        ProfileConfig config;
        
        config.attr.sample_freq = 1000;
        config.attr.type = PERF_TYPE_HARDWARE;
        config.attr.config = PERF_COUNT_HW_CPU_CYCLES;
        
        REQUIRE(config.attr.sample_freq == 1000);
        REQUIRE(config.attr.type == PERF_TYPE_HARDWARE);
        REQUIRE(config.attr.config == PERF_COUNT_HW_CPU_CYCLES);
    }
}

TEST_CASE("ProfileConfig inheritance from Config", "[profile][config][inheritance]") {
    SECTION("Inherits base Config properties") {
        ProfileConfig config;
        
        // These should be accessible from base Config class
        config.verbose = true;
        config.folded = false;
        config.cpu = 3;
        
        REQUIRE(config.verbose == true);
        REQUIRE(config.folded == false);
        REQUIRE(config.cpu == 3);
    }

    SECTION("PID and TID vectors") {
        ProfileConfig config;
        
        // Test PID configuration
        config.pids.push_back(1234);
        config.pids.push_back(5678);
        REQUIRE(config.pids.size() == 2);
        REQUIRE(config.pids[0] == 1234);
        REQUIRE(config.pids[1] == 5678);
        
        // Test TID configuration
        config.tids.push_back(9999);
        REQUIRE(config.tids.size() == 1);
        REQUIRE(config.tids[0] == 9999);
    }
} 