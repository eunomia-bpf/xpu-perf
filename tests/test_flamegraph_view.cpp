#include <catch2/catch_all.hpp>
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
} 