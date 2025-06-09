.PHONY: build clean install test

all: build

build:
	cmake -B build
	cmake --build build

test:
	cmake -B build
	cmake --build build --target profiler_tests
	ctest --test-dir build

clean:
	rm -rf build

install:
	sudo apt update
	sudo apt-get install -y --no-install-recommends \
        libelf1 libelf-dev zlib1g-dev \
        make clang llvm