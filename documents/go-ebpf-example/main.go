package main

//go:generate go run github.com/cilium/ebpf/cmd/bpf2go -target amd64 counter counter.c -- -I/usr/include/x86_64-linux-gnu

import (
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/cilium/ebpf/link"
	"github.com/cilium/ebpf/rlimit"
)

func main() {
	// Remove memory lock limit
	if err := rlimit.RemoveMemlock(); err != nil {
		log.Fatal(err)
	}

	// Load eBPF objects
	objs := counterObjects{}
	if err := loadCounterObjects(&objs, nil); err != nil {
		log.Fatalf("loading objects: %v", err)
	}
	defer objs.Close()

	// Find bash binary
	bashPath := "/usr/bin/bash"
	if _, err := os.Stat(bashPath); err != nil {
		log.Fatalf("bash not found at %s: %v", bashPath, err)
	}

	// Attach uprobe to readline function in bash
	ex, err := link.OpenExecutable(bashPath)
	if err != nil {
		log.Fatalf("opening executable: %v", err)
	}

	up, err := ex.Uprobe("readline", objs.UprobeBashReadline, nil)
	if err != nil {
		log.Fatalf("attaching uprobe: %v", err)
	}
	defer up.Close()

	log.Println("Successfully attached uprobe to bash readline function")
	log.Println("Monitoring function calls per process...")
	log.Println("Open a bash shell and run commands to see activity")
	log.Println("Press Ctrl+C to exit")

	// Setup signal handler
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)

	// Read map periodically
	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-sig:
			log.Println("\nExiting...")
			return
		case <-ticker.C:
			fmt.Println("\n=== Function Call Stats ===")
			var (
				key   uint32
				value uint64
			)
			iter := objs.FunctionCalls.Iterate()
			count := 0
			for iter.Next(&key, &value) {
				fmt.Printf("PID %d: %d calls\n", key, value)
				count++
			}
			if err := iter.Err(); err != nil {
				log.Printf("Error iterating map: %v\n", err)
			}
			if count == 0 {
				fmt.Println("No activity yet...")
			}
		}
	}
}
