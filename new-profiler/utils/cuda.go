package utils

import (
	"fmt"
	"os"
	"os/exec"
	"strings"
)

// DetectCudaLibraryFromBinary detects the CUDA library path used by the target binary
func DetectCudaLibraryFromBinary(binaryPath string) string {
	cmd := exec.Command("ldd", binaryPath)
	output, err := cmd.Output()
	if err != nil {
		fmt.Printf("Failed to run ldd on %s: %v\n", binaryPath, err)
		return ""
	}

	// Parse ldd output to find libcudart.so
	lines := strings.Split(string(output), "\n")
	for _, line := range lines {
		if strings.Contains(line, "libcudart.so") {
			// Extract the path (format: "libcudart.so.13 => /path/to/libcudart.so.13 (0xaddr)")
			parts := strings.Fields(line)
			for i, part := range parts {
				if strings.HasPrefix(part, "/") && strings.Contains(part, "libcudart.so") {
					fmt.Printf("Detected CUDA library from binary: %s\n", part)
					return part
				}
				// Handle symlinks - sometimes the resolved path is after "=>"
				if part == "=>" && i+1 < len(parts) && strings.HasPrefix(parts[i+1], "/") {
					fmt.Printf("Detected CUDA library from binary: %s\n", parts[i+1])
					return parts[i+1]
				}
			}
		}
	}

	return ""
}

// BuildUprobeList builds a list of uprobes for CUDA kernel launch symbols used by the target binary
func BuildUprobeList(cudaLibPath string, binaryPath string) []string {
	var uprobes []string

	// First, check which cudaLaunchKernel symbols the target binary actually uses
	cmd := exec.Command("nm", "-D", binaryPath)
	output, err := cmd.Output()

	usedSymbols := make(map[string]bool)
	if err == nil {
		lines := strings.Split(string(output), "\n")
		for _, line := range lines {
			if strings.Contains(line, "cudaLaunchKernel") && strings.Contains(line, " U ") {
				// Extract symbol name (format: "U symbol_name@version")
				parts := strings.Fields(line)
				if len(parts) >= 2 {
					symbol := parts[1]
					// Remove version suffix if present (e.g., "@libcudart.so.13")
					if idx := strings.Index(symbol, "@"); idx != -1 {
						symbol = symbol[:idx]
					}
					usedSymbols[symbol] = true
				}
			}
		}
	}

	// If we found symbols used by the binary, attach only to those
	if len(usedSymbols) > 0 {
		fmt.Printf("Target binary uses %d cudaLaunchKernel symbol(s):\n", len(usedSymbols))
		for symbol := range usedSymbols {
			uprobes = append(uprobes, cudaLibPath+":"+symbol)
			fmt.Printf("  - %s\n", symbol)
		}
		return uprobes
	}

	// Fallback: if we couldn't detect from binary, check what's available in the library
	fmt.Printf("Could not detect symbols from binary, checking library...\n")
	cmd = exec.Command("nm", "-D", cudaLibPath)
	output, err = cmd.Output()

	foundSymbols := make(map[string]bool)
	if err == nil {
		// Parse nm output to find cudaLaunchKernel symbols
		lines := strings.Split(string(output), "\n")
		for _, line := range lines {
			if strings.Contains(line, "cudaLaunchKernel") && strings.Contains(line, " T ") {
				// Extract symbol name (format: "address T symbol_name")
				parts := strings.Fields(line)
				if len(parts) >= 3 {
					symbol := parts[2]
					// Remove version suffix if present (e.g., "@@libcudart.so.13")
					if idx := strings.Index(symbol, "@@"); idx != -1 {
						symbol = symbol[:idx]
					}
					foundSymbols[symbol] = true
				}
			}
		}
	}

	if len(foundSymbols) > 0 {
		fmt.Printf("Found %d cudaLaunchKernel symbols in %s\n", len(foundSymbols), cudaLibPath)
		for symbol := range foundSymbols {
			uprobes = append(uprobes, cudaLibPath+":"+symbol)
		}
	} else {
		// Last resort: try common symbol names
		fmt.Printf("Could not detect symbols, trying common symbol names\n")
		commonSymbols := []string{
			"__cudaLaunchKernel",
		}
		for _, symbol := range commonSymbols {
			uprobes = append(uprobes, cudaLibPath+":"+symbol)
		}
	}

	return uprobes
}

// FindCudaLibrary searches for CUDA runtime library in common locations
func FindCudaLibrary() string {
	// Search for CUDA runtime library in common locations
	cudaPaths := []string{
		"/usr/local/cuda-12.9/lib64/libcudart.so.13",
		"/usr/local/cuda-12.9/lib64/libcudart.so.12",
		"/usr/local/cuda-13.0/lib64/libcudart.so.13",
		"/usr/local/cuda-13.0/lib64/libcudart.so.12",
		"/usr/local/cuda/lib64/libcudart.so.13",
		"/usr/local/cuda/lib64/libcudart.so.12",
		"/usr/local/cuda/lib64/libcudart.so",
		"/usr/local/cuda-12.8/lib64/libcudart.so.12",
	}

	for _, path := range cudaPaths {
		if _, err := os.Stat(path); err == nil {
			return path
		}
	}

	return ""
}
