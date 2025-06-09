.PHONY: build clean install

all: build

build:
	cmake -B build
	cmake --build build

clean:
	rm -rf build

install:
	sudo apt update
	sudo apt-get install -y --no-install-recommends \
        libelf1 libelf-dev zlib1g-dev \
        make clang llvm