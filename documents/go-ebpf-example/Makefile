.PHONY: all clean example ebpf run

all: example ebpf

example:
	g++ -o example example.cpp -lpthread -g

ebpf:
	go generate
	go build -o ebpf-demo

run-example:
	./example

run-ebpf:
	sudo ./ebpf-demo

clean:
	rm -f example ebpf-demo counter_x86_bpfel.go counter_x86_bpfel.o
