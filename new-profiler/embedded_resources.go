package main

import (
	_ "embed"
	"fmt"
	"os"
	"path/filepath"
)

//go:embed libcupti_trace_injection.so
var cuptiLibraryData []byte

// ExtractEmbeddedCUPTILibrary extracts the embedded CUPTI library to a temporary location
// and returns the path to the extracted file
func ExtractEmbeddedCUPTILibrary() (string, error) {
	// Create temp directory if it doesn't exist
	tmpDir := filepath.Join(os.TempDir(), "xpu-perf")
	if err := os.MkdirAll(tmpDir, 0755); err != nil {
		return "", fmt.Errorf("failed to create temp directory: %v", err)
	}

	// Create the library file with a unique name based on PID
	libPath := filepath.Join(tmpDir, fmt.Sprintf("libcupti_trace_injection_%d.so", os.Getpid()))

	// Check if file already exists and is valid
	if stat, err := os.Stat(libPath); err == nil && stat.Size() == int64(len(cuptiLibraryData)) {
		return libPath, nil
	}

	// Write the embedded library data to the temp file
	if err := os.WriteFile(libPath, cuptiLibraryData, 0755); err != nil {
		return "", fmt.Errorf("failed to write CUPTI library: %v", err)
	}

	return libPath, nil
}

// CleanupEmbeddedLibrary removes the extracted CUPTI library
func CleanupEmbeddedLibrary(libPath string) {
	if libPath != "" {
		os.Remove(libPath)
	}
}
