#include "flamegraph_generator.hpp"
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