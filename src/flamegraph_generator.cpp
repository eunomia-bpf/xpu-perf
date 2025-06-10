#include "flamegraph_generator.hpp"
#include "analyzers/flamegraph_view.hpp"
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <algorithm>
#include <ctime>
#include <iomanip>
#include <sys/stat.h>

FlamegraphGenerator::FlamegraphGenerator(const std::string& output_dir, int freq, int duration)
    : output_dir_(output_dir), sampling_freq_(freq), duration_(duration), actual_wall_clock_time_(duration) {
}

void FlamegraphGenerator::set_actual_wall_clock_time(double actual_time_seconds) {
    actual_wall_clock_time_ = actual_time_seconds;
}

bool FlamegraphGenerator::create_output_directory() {
    try {
        std::filesystem::create_directories(output_dir_);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to create output directory: " << e.what() << std::endl;
        return false;
    }
}

uint64_t FlamegraphGenerator::normalize_offcpu_time(uint64_t microseconds) {
    if (sampling_freq_ <= 0) return microseconds;
    
    // Calculate microseconds per sample: (1.0 / freq) * 1,000,000
    double us_per_sample = (1.0 / sampling_freq_) * 1000000.0;
    
    // Convert microseconds to equivalent samples
    uint64_t normalized_value = static_cast<uint64_t>(std::max(1.0, microseconds / us_per_sample));
    return normalized_value;
}

std::string FlamegraphGenerator::add_annotation(const std::string& stack, bool is_oncpu) {
    // Remove the first part (process name) and add annotation
    std::string clean_stack = stack;
    size_t first_semicolon = stack.find(';');
    if (first_semicolon != std::string::npos) {
        clean_stack = stack.substr(first_semicolon + 1);
    }
    
    // Add annotation
    return clean_stack + (is_oncpu ? "_[c]" : "_[o]");
}

std::string FlamegraphGenerator::generate_folded_file(const std::vector<FlamegraphEntry>& entries, 
                                                    const std::string& filename_prefix) {
    if (!create_output_directory()) {
        return "";
    }
    
    // Create timestamp for unique filename
    auto now = std::time(nullptr);
    std::stringstream ss;
    ss << output_dir_ << "/" << filename_prefix << "_" << now << ".folded";
    std::string folded_file = ss.str();
    
    // Combine and sort entries
    std::map<std::string, uint64_t> combined_stacks;
    
    for (const auto& entry : entries) {
        std::string annotated_stack = add_annotation(entry.stack_trace, entry.is_oncpu);
        uint64_t value = entry.value;
        
        // Normalize off-CPU values
        if (!entry.is_oncpu) {
            value = normalize_offcpu_time(value);
        }
        
        combined_stacks[annotated_stack] += value;
    }
    
    // Sort by value (descending)
    std::vector<std::pair<std::string, uint64_t>> sorted_stacks(combined_stacks.begin(), combined_stacks.end());
    std::sort(sorted_stacks.begin(), sorted_stacks.end(), 
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // Write folded file
    try {
        std::ofstream file(folded_file);
        if (!file.is_open()) {
            std::cerr << "Failed to open file: " << folded_file << std::endl;
            return "";
        }
        
        for (const auto& [stack, value] : sorted_stacks) {
            file << stack << " " << value << "\n";
        }
        
        file.close();
        std::cout << "Folded data written to: " << folded_file << std::endl;
        return folded_file;
        
    } catch (const std::exception& e) {
        std::cerr << "Error writing folded file: " << e.what() << std::endl;
        return "";
    }
}

bool FlamegraphGenerator::setup_flamegraph_tools() {
    std::string script_path = get_flamegraph_script_path();
    
    // Check if flamegraph.pl already exists
    if (std::filesystem::exists(script_path)) {
        return true;
    }
    
    std::cout << "FlameGraph tools not found, downloading..." << std::endl;
    return download_flamegraph_tools();
}

std::string FlamegraphGenerator::get_flamegraph_script_path() {
    return "FlameGraph/flamegraph.pl";
}

bool FlamegraphGenerator::download_flamegraph_tools() {
    // Clone FlameGraph repository
    std::string cmd = "git clone https://github.com/brendangregg/FlameGraph.git FlameGraph --depth=1 2>/dev/null";
    int result = system(cmd.c_str());
    
    if (result == 0) {
        // Make flamegraph.pl executable
        chmod("FlameGraph/flamegraph.pl", 0755);
        std::cout << "FlameGraph tools downloaded successfully" << std::endl;
        return true;
    } else {
        std::cerr << "Failed to download FlameGraph tools" << std::endl;
        return false;
    }
}

void FlamegraphGenerator::create_custom_flamegraph_script() {
    // No longer needed - using standard flamegraph.pl
}

std::string FlamegraphGenerator::generate_svg_from_folded(const std::string& folded_file, const std::string& title) {
    if (!setup_flamegraph_tools()) {
        return "";
    }
    
    std::string svg_file = folded_file;
    svg_file.replace(svg_file.find(".folded"), 7, ".svg");
    
    std::stringstream cmd;
    cmd << "perl FlameGraph/flamegraph.pl";
    if (!title.empty()) {
        cmd << " --title \"" << title << "\"";
    } else {
        cmd << " --title \"CPU Profile Flamegraph\"";
    }
    cmd << " --colors hot";  // Use hot colors (red/orange for CPU-intensive)
    cmd << " --width 1200";  // Set a reasonable width
    cmd << " " << folded_file << " > " << svg_file << " 2>&1";
    
    std::cout << "Running flamegraph command: " << cmd.str() << std::endl;
    
    int result = system(cmd.str().c_str());
    
    if (result == 0 && std::filesystem::exists(svg_file)) {
        // Check if the SVG file actually contains data
        std::ifstream check_file(svg_file);
        if (check_file.is_open()) {
            std::string line;
            bool has_content = false;
            while (std::getline(check_file, line)) {
                if (line.find("<svg") != std::string::npos) {
                    has_content = true;
                    break;
                }
            }
            check_file.close();
            
            if (has_content) {
                std::cout << "SVG flamegraph generated: " << svg_file << std::endl;
                return svg_file;
            } else {
                std::cerr << "SVG file was created but appears empty or invalid" << std::endl;
            }
        }
    } else {
        std::cerr << "Failed to generate SVG flamegraph (exit code: " << result << ")" << std::endl;
        // Try to read any error output
        std::ifstream error_file(svg_file);
        if (error_file.is_open()) {
            std::string error_content((std::istreambuf_iterator<char>(error_file)),
                                     std::istreambuf_iterator<char>());
            error_file.close();
            if (!error_content.empty()) {
                std::cerr << "Error output: " << error_content << std::endl;
            }
        }
    }
    
    return "";
}

void FlamegraphGenerator::generate_analysis_file(const std::string& filename,
                                                const std::vector<FlamegraphEntry>& entries,
                                                const std::string& analysis_type) {
    if (!create_output_directory()) {
        return;
    }
    
    std::string analysis_file = output_dir_ + "/" + filename + "_analysis.txt";
    
    // Calculate statistics
    uint64_t oncpu_samples = 0;
    uint64_t offcpu_us = 0;
    size_t oncpu_count = 0;
    size_t offcpu_count = 0;
    
    for (const auto& entry : entries) {
        if (entry.is_oncpu) {
            oncpu_samples += entry.value;
            oncpu_count++;
        } else {
            offcpu_us += entry.value;
            offcpu_count++;
        }
    }
    
    // Convert to wall clock times
    double oncpu_time_sec = static_cast<double>(oncpu_samples) / sampling_freq_;
    double offcpu_time_sec = static_cast<double>(offcpu_us) / 1000000.0;
    double measured_time_sec = oncpu_time_sec + offcpu_time_sec;
    
    // Wall clock time should be the actual duration that was profiled
    double wall_clock_time_sec = actual_wall_clock_time_;
    double coverage_pct = (measured_time_sec / wall_clock_time_sec) * 100.0;
    
    try {
        std::ofstream file(analysis_file);
        file << std::fixed << std::setprecision(3);
        
        file << analysis_type << " Analysis Report\n";
        file << std::string(50, '=') << "\n\n";
        
        file << "Profiling Parameters:\n";
        file << "Duration: " << duration_ << " seconds\n";
        file << "Sampling frequency: " << sampling_freq_ << " Hz\n\n";
        
        file << "Time Analysis:\n";
        file << std::string(40, '-') << "\n";
        file << "On-CPU time: " << oncpu_time_sec << "s (" << oncpu_samples << " samples)\n";
        file << "Off-CPU time: " << offcpu_time_sec << "s (" << offcpu_us << " μs)\n";
        file << "Total measured time: " << wall_clock_time_sec << "s (wall clock)\n";
        file << "Active CPU time: " << measured_time_sec << "s (on-CPU + off-CPU)\n";
        file << "Activity coverage: " << coverage_pct << "% of wall clock time\n\n";
        
        file << "Stack Trace Summary:\n";
        file << std::string(40, '-') << "\n";
        file << "On-CPU stack traces: " << oncpu_count << "\n";
        file << "Off-CPU stack traces: " << offcpu_count << "\n";
        file << "Total unique stacks: " << (oncpu_count + offcpu_count) << "\n\n";
        
        file << "Coverage Assessment:\n";
        file << std::string(40, '-') << "\n";
        if (coverage_pct < 10) {
            file << "⚠️  Very low activity - process mostly idle\n";
        } else if (coverage_pct < 50) {
            file << "⚠️  Low activity - process partially idle\n";
        } else if (coverage_pct > 150) {
            file << "⚠️  High activity - possible measurement anomaly\n";
        } else {
            file << "✓ Activity level appears reasonable for active process\n";
        }
        
        file << "\nTime Verification Notes:\n";
        file << std::string(40, '-') << "\n";
        file << "• On-CPU time = samples / sampling_frequency (" << sampling_freq_ << " Hz)\n";
        file << "• Off-CPU time = blocking_time_microseconds / 1,000,000\n";
        file << "• Total measured time = wall clock time (actual profiling duration)\n";
        file << "• Active CPU time = on-CPU + off-CPU time (time spent executing)\n";
        file << "• Activity coverage shows what % of wall clock time was active\n";
        
        file.close();
        std::cout << "Analysis saved to: " << analysis_file << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error writing analysis file: " << e.what() << std::endl;
    }
}

void FlamegraphGenerator::generate_single_flamegraph(const std::map<pid_t, std::unique_ptr<FlameGraphView>>& per_thread_data,
                                                    const std::vector<pid_t>& pids) {
    // Combine all thread data into a single flamegraph
    std::vector<FlamegraphEntry> all_entries;
    
    for (const auto& [tid, flamegraph] : per_thread_data) {
        if (flamegraph && flamegraph->entries.size() > 0) {
            auto thread_entries = convert_flamegraph_to_entries(*flamegraph);
            all_entries.insert(all_entries.end(), thread_entries.begin(), thread_entries.end());
        }
    }
    
    if (all_entries.empty()) {
        std::cout << "No stack traces collected from either tool" << std::endl;
        return;
    }
    
    // Generate files
    std::string prefix = "process_profile";
    if (!pids.empty()) {
        prefix += "_pid" + std::to_string(pids[0]);
    }
    
    std::string folded_file = generate_folded_file(all_entries, prefix);
    if (!folded_file.empty()) {
        std::string title = "Process " + std::to_string(pids.empty() ? 0 : pids[0]) + " Combined Profile";
        std::string svg_file = generate_svg_from_folded(folded_file, title);
        generate_analysis_file(prefix, all_entries, "Process-Level");
        
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "PROCESS PROFILING COMPLETE" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        std::cout << "📊 Folded data: " << folded_file << std::endl;
        if (!svg_file.empty()) {
            std::cout << "🔥 Flamegraph:  " << svg_file << std::endl;
            std::cout << "   Open " << svg_file << " in a web browser to view the interactive flamegraph" << std::endl;
        }
        
        std::cout << "\n📝 Interpretation guide:" << std::endl;
        std::cout << "   • Red frames show CPU-intensive code paths (on-CPU) with actual function names" << std::endl;
        std::cout << "   • Blue frames show blocking/waiting operations (off-CPU) with actual function names" << std::endl;
        std::cout << "   • Wider sections represent more time spent in those functions" << std::endl;
        std::cout << "   • Values are normalized to make on-CPU and off-CPU time comparable" << std::endl;
    }
}

void FlamegraphGenerator::generate_multithread_flamegraphs(const std::map<pid_t, std::unique_ptr<FlameGraphView>>& per_thread_data,
                                                          const std::vector<ThreadInfo>& detected_threads) {
    if (per_thread_data.empty()) {
        std::cerr << "No per-thread data available" << std::endl;
        return;
    }
    
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "MULTI-THREAD PROFILING COMPLETE" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "Generating flamegraphs for " << per_thread_data.size() << " threads..." << std::endl;
    
    // Generate flamegraph for each thread
    for (const auto& [tid, flamegraph] : per_thread_data) {
        if (!flamegraph || flamegraph->entries.empty()) {
            std::cout << "⚠️  Thread " << tid << ": No data available" << std::endl;
            continue;
        }
        
        // Convert FlameGraphView entries to FlamegraphEntry format for this thread
        auto thread_entries = convert_flamegraph_to_entries(*flamegraph);
        
        // Determine thread role for better naming
        std::string thread_role = get_thread_role(tid, "", {});
        for (const auto& thread_info : detected_threads) {
            if (thread_info.tid == tid) {
                thread_role = thread_info.role;
                break;
            }
        }
        
        std::string thread_prefix = "thread_" + std::to_string(tid) + "_" + thread_role;
        
        // Generate files for this thread
        std::string folded_file = generate_folded_file(thread_entries, thread_prefix);
        if (!folded_file.empty()) {
            std::string title = "Thread " + std::to_string(tid) + " (" + thread_role + ") Profile";
            std::string svg_file = generate_svg_from_folded(folded_file, title);
            generate_analysis_file(thread_prefix, thread_entries, "Thread-Level");
            
            std::cout << "\n📊 Thread " << tid << " (" << thread_role << "):" << std::endl;
            std::cout << "   📄 Folded data: " << folded_file << std::endl;
            if (!svg_file.empty()) {
                std::cout << "   🔥 Flamegraph:  " << svg_file << std::endl;
            }
            std::cout << "   📈 Samples: " << thread_entries.size() << " entries" << std::endl;
        } else {
            std::cout << "❌ Thread " << tid << ": Failed to generate flamegraph" << std::endl;
        }
    }
    
    // Also generate a combined process-level flamegraph for overview
    std::cout << "\n📊 Generating combined process flamegraph..." << std::endl;
    generate_single_flamegraph(per_thread_data, {});
    
    std::cout << "\n📝 Multi-threading interpretation guide:" << std::endl;
    std::cout << "   • Each thread has its own flamegraph showing its specific behavior" << std::endl;
    std::cout << "   • Red frames show CPU-intensive code paths (on-CPU) with actual function names" << std::endl;
    std::cout << "   • Blue frames show blocking/waiting operations (off-CPU) with actual function names" << std::endl;
    std::cout << "   • Compare thread flamegraphs to identify thread roles and bottlenecks" << std::endl;
    std::cout << "   • The combined process flamegraph shows overall application behavior" << std::endl;
    
    std::cout << "\n🎯 Thread Summary:" << std::endl;
    for (const auto& [tid, flamegraph] : per_thread_data) {
        if (flamegraph && !flamegraph->entries.empty()) {
            std::string role = get_thread_role(tid, "", {});
            for (const auto& thread_info : detected_threads) {
                if (thread_info.tid == tid) {
                    role = thread_info.role;
                    break;
                }
            }
            std::cout << "   • Thread " << tid << " (" << role << "): " 
                      << flamegraph->entries.size() << " unique stacks" << std::endl;
        }
    }
}

std::vector<FlamegraphEntry> FlamegraphGenerator::convert_flamegraph_to_entries(const FlameGraphView& flamegraph) {
    std::vector<FlamegraphEntry> entries;
    
    for (const auto& entry : flamegraph.entries) {
        FlamegraphEntry fg_entry;
        
        // Build stack trace from folded_stack vector
        std::string stack_trace;
        if (!entry.folded_stack.empty()) {
            stack_trace = entry.folded_stack[0];
            for (size_t i = 1; i < entry.folded_stack.size(); ++i) {
                stack_trace += ";" + entry.folded_stack[i];
            }
        }
        
        fg_entry.stack_trace = stack_trace;
        fg_entry.value = entry.sample_count;
        fg_entry.is_oncpu = entry.is_oncpu;
        
        entries.push_back(fg_entry);
    }
    
    return entries;
}

std::string FlamegraphGenerator::get_thread_role(pid_t tid, const std::string& cmd, const std::vector<pid_t>& pids) {
    if (!pids.empty() && tid == pids[0]) {
        return "main";
    } else {
        return "thread_" + std::to_string(tid);
    }
} 