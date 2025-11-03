// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/ianlancetaylor/demangle"
	"go.opentelemetry.io/ebpf-profiler/libpf"
	"go.opentelemetry.io/ebpf-profiler/libpf/pfelf"
)

// Symbolizer provides comprehensive symbol resolution for native frames
// using pfelf (ELF parsing) and demangling for C++/Rust symbols
type Symbolizer struct {
	// Cache for ELF file handles
	elfCache map[string]*pfelf.File
	elfMu    sync.RWMutex

	// Cache for symbol maps (both regular and dynamic)
	symbolCache map[string]*libpf.SymbolMap
	symbolMu    sync.RWMutex

	// Track files that failed to open
	failedFiles map[string]bool
	failedMu    sync.RWMutex
}

// NewSymbolizer creates a new symbolizer instance
func NewSymbolizer() *Symbolizer {
	return &Symbolizer{
		elfCache:    make(map[string]*pfelf.File),
		symbolCache: make(map[string]*libpf.SymbolMap),
		failedFiles: make(map[string]bool),
	}
}

// Close closes all cached ELF files
func (s *Symbolizer) Close() {
	s.elfMu.Lock()
	defer s.elfMu.Unlock()

	for _, ef := range s.elfCache {
		ef.Close()
	}
	s.elfCache = make(map[string]*pfelf.File)
}

// getOrOpenELF opens an ELF file or returns cached handle
func (s *Symbolizer) getOrOpenELF(path string) (*pfelf.File, error) {
	// Check if we already failed to open this file
	s.failedMu.RLock()
	if s.failedFiles[path] {
		s.failedMu.RUnlock()
		return nil, fmt.Errorf("previously failed to open")
	}
	s.failedMu.RUnlock()

	// Check cache first
	s.elfMu.RLock()
	if ef, ok := s.elfCache[path]; ok {
		s.elfMu.RUnlock()
		return ef, nil
	}
	s.elfMu.RUnlock()

	// Open new file
	ef, err := pfelf.Open(path)
	if err != nil {
		s.failedMu.Lock()
		s.failedFiles[path] = true
		s.failedMu.Unlock()
		return nil, err
	}

	// Cache the handle
	s.elfMu.Lock()
	s.elfCache[path] = ef
	s.elfMu.Unlock()

	return ef, nil
}

// getSymbolMap loads and caches symbol map for a file
func (s *Symbolizer) getSymbolMap(path string) (*libpf.SymbolMap, error) {
	// Check cache first
	s.symbolMu.RLock()
	if symmap, ok := s.symbolCache[path]; ok {
		s.symbolMu.RUnlock()
		return symmap, nil
	}
	s.symbolMu.RUnlock()

	// Open ELF file
	ef, err := s.getOrOpenELF(path)
	if err != nil {
		return nil, err
	}

	// Try dynamic symbols first (works with stripped binaries)
	symmap, err := ef.ReadDynamicSymbols()
	if err != nil || symmap == nil || symmap.Len() == 0 {
		// Fall back to regular symbol table
		symmap, err = ef.ReadSymbols()
		if err != nil || symmap == nil {
			return nil, fmt.Errorf("no symbols available")
		}
	}

	// Cache the symbol map
	s.symbolMu.Lock()
	s.symbolCache[path] = symmap
	s.symbolMu.Unlock()

	return symmap, nil
}

// Symbolize resolves an address to a symbol name with offset
// Returns the symbolized name or empty string if not found
func (s *Symbolizer) Symbolize(fileName string, fileOffset uint64) string {
	symmap, err := s.getSymbolMap(fileName)
	if err != nil {
		return ""
	}

	// Look up symbol by address
	symbolName, offset, found := symmap.LookupByAddress(libpf.SymbolValue(fileOffset))
	if !found || symbolName == "" {
		return ""
	}

	// Demangle C++/Rust symbols
	demangledName := s.demangle(string(symbolName))

	// Format with offset if non-zero
	if offset == 0 {
		return demangledName
	}
	return fmt.Sprintf("%s+0x%x", demangledName, offset)
}

// demangle attempts to demangle C++/Rust symbol names
func (s *Symbolizer) demangle(symbol string) string {
	// Skip if it doesn't look like a mangled name
	if !strings.HasPrefix(symbol, "_Z") && !strings.HasPrefix(symbol, "_R") {
		return symbol
	}

	// Try to demangle
	demangled, err := demangle.ToString(symbol, demangle.NoParams)
	if err != nil {
		// If demangling fails, return original
		return symbol
	}

	return demangled
}

// SymbolizeFrame symbolizes a frame name (filename+offset format)
// Handles various formats: "file+0x123", "file+0x123abc", "file"
func (s *Symbolizer) SymbolizeFrame(frameName string) string {
	// Parse frame name
	parts := strings.Split(frameName, "+")
	if len(parts) != 2 {
		// No offset, return as-is
		return frameName
	}

	fileName := parts[0]
	offsetStr := parts[1]

	// Parse offset (hex format: 0x123)
	var offset uint64
	if _, err := fmt.Sscanf(offsetStr, "0x%x", &offset); err != nil {
		// Invalid offset format
		return frameName
	}

	// Try to symbolize
	symbol := s.Symbolize(fileName, offset)
	if symbol != "" {
		return symbol
	}

	// Return original if symbolization failed
	return frameName
}

// ResolveExecutablePath resolves a basename to full path using /proc
func ResolveExecutablePath(pid int, basename string) string {
	// Check common library paths first
	commonPaths := []string{
		"/usr/lib/x86_64-linux-gnu",
		"/usr/lib",
		"/lib/x86_64-linux-gnu",
		"/lib",
		"/usr/local/lib",
		"/usr/local/cuda/lib64",
		"/usr/local/cuda-12.9/lib64",
		"/usr/local/cuda-13.0/lib64",
	}

	// Try common paths
	for _, dir := range commonPaths {
		fullPath := filepath.Join(dir, basename)
		if _, err := os.Stat(fullPath); err == nil {
			return fullPath
		}
	}

	// Try /proc/<pid>/exe symlink
	exePath, err := os.Readlink(fmt.Sprintf("/proc/%d/exe", pid))
	if err == nil && filepath.Base(exePath) == basename {
		return exePath
	}

	// Try /proc/<pid>/maps to find the actual path
	mapsPath := fmt.Sprintf("/proc/%d/maps", pid)
	data, err := os.ReadFile(mapsPath)
	if err == nil {
		for _, line := range strings.Split(string(data), "\n") {
			fields := strings.Fields(line)
			if len(fields) >= 6 {
				mappedPath := strings.Join(fields[5:], " ")
				if filepath.Base(mappedPath) == basename {
					return mappedPath
				}
			}
		}
	}

	// Fall back to basename
	return basename
}

// Global symbolizer instance
var globalSymbolizer *Symbolizer
var symbolizerOnce sync.Once

// GetSymbolizer returns the global symbolizer instance
func GetSymbolizer() *Symbolizer {
	symbolizerOnce.Do(func() {
		globalSymbolizer = NewSymbolizer()
	})
	return globalSymbolizer
}
