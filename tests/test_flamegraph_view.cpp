#include <catch2/catch.hpp>
#include "../src/analyzers/flamegraph_view.hpp"
#include "../src/analyzers/symbol_resolver.hpp"
#include <vector>
#include <string>

TEST_CASE("FlameGraphView vector-based folded stack functionality", "[flamegraph]") {
    SECTION("FlameGraphEntry structure") {
        FlameGraphEntry entry;
        entry.folded_stack = {"myapp", "main", "process_data", "parse_input"};
        entry.command = "myapp";
        entry.pid = 1234;
        entry.sample_count = 42;
        entry.percentage = 25.5;
        entry.stack_depth = 3;
        entry.is_oncpu = true;
        
        REQUIRE(entry.folded_stack.size() == 4);
        REQUIRE(entry.folded_stack[0] == "myapp");
        REQUIRE(entry.folded_stack[1] == "main");
        REQUIRE(entry.folded_stack[2] == "process_data");
        REQUIRE(entry.folded_stack[3] == "parse_input");
        REQUIRE(entry.command == "myapp");
        REQUIRE(entry.sample_count == 42);
        REQUIRE(entry.stack_depth == 3);
        REQUIRE(entry.is_oncpu == true);
    }
    
    SECTION("FlameGraphView basic functionality") {
        FlameGraphView flamegraph("test_analyzer", true);
        
        REQUIRE(flamegraph.analyzer_name == "test_analyzer");
        REQUIRE(flamegraph.total_samples == 0);
        REQUIRE(flamegraph.entries.empty());
    }
    
    SECTION("Add stack trace and build folded stack") {
        FlameGraphView flamegraph("test_analyzer", true);
        
        std::vector<std::string> user_stack = {"function_a", "function_b", "function_c"};
        std::vector<std::string> kernel_stack = {"sys_call", "kernel_func"};
        
        flamegraph.add_stack_trace(user_stack, kernel_stack, "test_app", 1234, 10, true);
        flamegraph.finalize();
        
        REQUIRE(flamegraph.total_samples == 10);
        REQUIRE(flamegraph.entries.size() == 1);
        
        const auto& entry = flamegraph.entries[0];
        // Check the expected order: command, user_stack (reversed), delimiter, kernel_stack (reversed)
        REQUIRE(entry.folded_stack[0] == "test_app");  // command
        REQUIRE(entry.folded_stack[1] == "function_c"); // user stack reversed
        REQUIRE(entry.folded_stack[2] == "function_b");
        REQUIRE(entry.folded_stack[3] == "function_a");
        REQUIRE(entry.folded_stack[4] == "--");        // delimiter
        REQUIRE(entry.folded_stack[5] == "kernel_func"); // kernel stack reversed
        REQUIRE(entry.folded_stack[6] == "sys_call");
        
        REQUIRE(entry.sample_count == 10);
        REQUIRE(entry.percentage == 100.0); // Only one entry, so 100%
    }
    
    SECTION("Stack depth calculation") {
        FlameGraphView flamegraph("test_analyzer", true);
        
        std::vector<std::string> user_stack = {"func1", "func2"};
        std::vector<std::string> kernel_stack = {"sys1"};
        
        flamegraph.add_stack_trace(user_stack, kernel_stack, "app", 1234, 5, true);
        flamegraph.finalize();
        
        const auto& entry = flamegraph.entries[0];
        // Stack depth should be total elements - 1 (excluding command)
        // Expected: app;func2;func1;--;sys1 = 5 elements, depth = 4
        REQUIRE(entry.stack_depth == 4);
    }
    
    SECTION("Folded format output") {
        FlameGraphView flamegraph("test_analyzer", true);
        
        std::vector<std::string> user_stack = {"main", "helper"};
        std::vector<std::string> kernel_stack;
        
        flamegraph.add_stack_trace(user_stack, kernel_stack, "simple_app", 999, 25, true);
        flamegraph.finalize();
        
        std::string folded_output = flamegraph.to_folded_format();
        // Expected format: simple_app;helper;main 25
        REQUIRE(folded_output == "simple_app;helper;main 25\n");
    }
    
    SECTION("Readable format output") {
        FlameGraphView flamegraph("test_analyzer", true);
        
        std::vector<std::string> user_stack = {"main"};
        std::vector<std::string> kernel_stack;
        
        flamegraph.add_stack_trace(user_stack, kernel_stack, "readable_app", 777, 15, true);
        flamegraph.finalize();
        
        std::string readable_output = flamegraph.to_readable_format();
        REQUIRE(readable_output.find("readable_app") != std::string::npos);
        REQUIRE(readable_output.find("15 samples") != std::string::npos);
        REQUIRE(readable_output.find("On-CPU") != std::string::npos);
    }
    
    SECTION("Function totals aggregation") {
        FlameGraphView flamegraph("test_analyzer", true);
        
        // Add multiple stacks with overlapping functions
        std::vector<std::string> stack1 = {"common_func", "func_a"};
        std::vector<std::string> stack2 = {"common_func", "func_b"};
        std::vector<std::string> empty_kernel;
        
        flamegraph.add_stack_trace(stack1, empty_kernel, "app1", 100, 10, true);
        flamegraph.add_stack_trace(stack2, empty_kernel, "app2", 200, 20, true);
        flamegraph.finalize();
        
        auto function_totals = flamegraph.get_function_totals();
        
        // common_func should appear in both stacks
        REQUIRE(function_totals["common_func"] == 30); // 10 + 20
        REQUIRE(function_totals["func_a"] == 10);
        REQUIRE(function_totals["func_b"] == 20);
        REQUIRE(function_totals["app1"] == 10);
        REQUIRE(function_totals["app2"] == 20);
    }
    
    SECTION("Top stacks retrieval") {
        FlameGraphView flamegraph("test_analyzer", true);
        
        std::vector<std::string> user_stack1 = {"high_freq_func"};
        std::vector<std::string> user_stack2 = {"low_freq_func"};
        std::vector<std::string> empty_kernel;
        
        flamegraph.add_stack_trace(user_stack1, empty_kernel, "app", 1, 100, true); // High frequency
        flamegraph.add_stack_trace(user_stack2, empty_kernel, "app", 1, 10, true);  // Low frequency
        flamegraph.finalize();
        
        auto top_stacks = flamegraph.get_top_stacks(2);
        REQUIRE(top_stacks.size() == 2);
        
        // Should be sorted by sample count (descending)
        REQUIRE(top_stacks[0].sample_count == 100);
        REQUIRE(top_stacks[1].sample_count == 10);
        REQUIRE(top_stacks[0].folded_stack[1] == "high_freq_func");
        REQUIRE(top_stacks[1].folded_stack[1] == "low_freq_func");
    }
} 