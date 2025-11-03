package main

//go:generate go run github.com/cilium/ebpf/cmd/bpf2go -target amd64 counter counter.c -- -I/usr/include/x86_64-linux-gnu

import (
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/cilium/ebpf"
	"github.com/cilium/ebpf/link"
	"github.com/cilium/ebpf/rlimit"
	"github.com/mmat11/usdt"
)

func main() {
	// Parse command line flags
	exePath := flag.String("exe", "./example", "Path to the executable to attach to")
	pidFlag := flag.Int("pid", 0, "PID of running process to attach USDT (required for USDT)")
	flag.Parse()

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

	// Check if executable exists
	if _, err := os.Stat(*exePath); err != nil {
		log.Fatalf("executable not found at %s: %v", *exePath, err)
	}

	// Attach uprobe to add_numbers function
	ex, err := link.OpenExecutable(*exePath)
	if err != nil {
		log.Fatalf("opening executable: %v", err)
	}

	up, err := ex.Uprobe("_Z11add_numbersii", objs.UprobeAddNumbers, nil)
	if err != nil {
		log.Fatalf("attaching uprobe: %v", err)
	}
	defer up.Close()

	log.Printf("Successfully attached uprobe to add_numbers function in %s\n", *exePath)

	// Attach USDT probe if PID provided
	var usdtProbe *usdt.USDT
	if *pidFlag > 0 {
		usdtProbe, err = usdt.New(objs.UsdtAddOperation, "myapp", "add_operation", *pidFlag)
		if err != nil {
			log.Printf("Warning: Failed to attach USDT probe: %v\n", err)
			log.Println("Continuing with uprobe only...")
		} else {
			defer usdtProbe.Close()
			log.Printf("Successfully attached USDT probe to PID %d\n", *pidFlag)
		}
	} else {
		log.Println("Note: No PID provided, USDT probe not attached. Use -pid flag to attach USDT.")
	}

	log.Println("Monitoring function calls...")
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
			fmt.Println("\n=== Uprobe Stats ===")
			printMap(objs.UprobeCalls)

			fmt.Println("\n=== USDT Stats ===")
			printMap(objs.UsdtCalls)
		}
	}
}

func printMap(m *ebpf.Map) {
	var (
		key   uint32
		value uint64
	)
	iter := m.Iterate()
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
