cmake_minimum_required(VERSION 3.16)
project(bpf_profiler)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_program(CLANG_EXECUTABLE NAMES clang REQUIRED)
find_program(CARGO_EXECUTABLE NAMES cargo REQUIRED)

# Detect architecture
execute_process(
    COMMAND uname -m
    OUTPUT_VARIABLE ARCH_RAW
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Convert architecture names to match kernel conventions
if(ARCH_RAW STREQUAL "x86_64")
    set(ARCH "x86")
elseif(ARCH_RAW MATCHES "^arm.*")
    set(ARCH "arm")
elseif(ARCH_RAW STREQUAL "aarch64")
    set(ARCH "arm64")
elseif(ARCH_RAW STREQUAL "ppc64le")
    set(ARCH "powerpc")
elseif(ARCH_RAW MATCHES "^mips.*")
    set(ARCH "mips")
elseif(ARCH_RAW STREQUAL "riscv64")
    set(ARCH "riscv")
elseif(ARCH_RAW STREQUAL "loongarch64")
    set(ARCH "loongarch")
else()
    set(ARCH ${ARCH_RAW})
endif()

# Set paths
set(LIBBPF_SRC ${CMAKE_CURRENT_SOURCE_DIR}/libbpf/src)
set(BPFTOOL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/bpftool/src)
set(LIBBLAZESYM_SRC ${CMAKE_CURRENT_SOURCE_DIR}/blazesym)
set(VMLINUX_H ${CMAKE_CURRENT_SOURCE_DIR}/vmlinux/${ARCH}/vmlinux.h)
set(OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

# Get clang system includes for BPF compilation (simplified approach)
execute_process(
    COMMAND bash -c "${CLANG_EXECUTABLE} -v -E - </dev/null 2>&1 | sed -n '/<...> search starts here:/,/End of search list./{ s| \\(/.*\\)|-idirafter \\1|p }'"
    OUTPUT_VARIABLE CLANG_BPF_SYS_INCLUDES_RAW
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Convert to list
string(REPLACE "\n" ";" CLANG_BPF_SYS_INCLUDES "${CLANG_BPF_SYS_INCLUDES_RAW}")

# Include directories
set(INCLUDES 
    -I${OUTPUT_DIR}
    -I${CMAKE_CURRENT_SOURCE_DIR}/libbpf/include/uapi
    -I${CMAKE_CURRENT_SOURCE_DIR}/vmlinux/${ARCH}
)

# Create output directory
file(MAKE_DIRECTORY ${OUTPUT_DIR})
file(MAKE_DIRECTORY ${OUTPUT_DIR}/libbpf)
file(MAKE_DIRECTORY ${OUTPUT_DIR}/bpftool)

# Build libbpf
set(LIBBPF_OBJ ${OUTPUT_DIR}/libbpf.a)
add_custom_command(
    OUTPUT ${LIBBPF_OBJ}
    COMMAND $(MAKE) -C ${LIBBPF_SRC} BUILD_STATIC_ONLY=1 
            OBJDIR=${OUTPUT_DIR}/libbpf/libbpf DESTDIR=${OUTPUT_DIR}/libbpf
            INCLUDEDIR= LIBDIR= UAPIDIR= install
    COMMAND cp ${OUTPUT_DIR}/libbpf/libbpf.a ${LIBBPF_OBJ}
    DEPENDS ${LIBBPF_SRC}/Makefile
    COMMENT "Building libbpf"
)

# Build bpftool
set(BPFTOOL ${OUTPUT_DIR}/bpftool/bootstrap/bpftool)
add_custom_command(
    OUTPUT ${BPFTOOL}
    COMMAND $(MAKE) ARCH= CROSS_COMPILE= OUTPUT=${OUTPUT_DIR}/bpftool/ -C ${BPFTOOL_SRC} bootstrap
    DEPENDS ${BPFTOOL_SRC}/Makefile
    COMMENT "Building bpftool"
)

# Build blazesym
set(LIBBLAZESYM_RUST ${LIBBLAZESYM_SRC}/target/release/libblazesym.a)
set(LIBBLAZESYM_OBJ ${OUTPUT_DIR}/libblazesym.a)
set(LIBBLAZESYM_HEADER ${OUTPUT_DIR}/blazesym.h)

add_custom_command(
    OUTPUT ${LIBBLAZESYM_RUST}
    COMMAND cd ${LIBBLAZESYM_SRC} && ${CARGO_EXECUTABLE} build --features=cheader,dont-generate-test-files --release
    WORKING_DIRECTORY ${LIBBLAZESYM_SRC}
    COMMENT "Building blazesym with cargo"
)

add_custom_command(
    OUTPUT ${LIBBLAZESYM_OBJ} ${LIBBLAZESYM_HEADER}
    COMMAND cp ${LIBBLAZESYM_RUST} ${LIBBLAZESYM_OBJ}
    COMMAND cp ${LIBBLAZESYM_SRC}/target/release/blazesym.h ${LIBBLAZESYM_HEADER}
    DEPENDS ${LIBBLAZESYM_RUST}
    COMMENT "Copying blazesym library and header"
)

# Function to compile BPF programs
function(compile_bpf_program bpf_source)
    get_filename_component(bpf_name ${bpf_source} NAME_WE)
    set(bpf_obj ${OUTPUT_DIR}/${bpf_name}.bpf.o)
    set(bpf_tmp ${OUTPUT_DIR}/${bpf_name}.tmp.bpf.o)
    set(skel_h ${OUTPUT_DIR}/${bpf_name}.skel.h)
    
    # Compile BPF source to object file
    add_custom_command(
        OUTPUT ${bpf_obj}
        COMMAND ${CLANG_EXECUTABLE} -g -O2 -target bpf -D__TARGET_ARCH_${ARCH}
                ${INCLUDES} ${CLANG_BPF_SYS_INCLUDES}
                -c ${CMAKE_CURRENT_SOURCE_DIR}/src/${bpf_source} -o ${bpf_tmp}
        COMMAND ${BPFTOOL} gen object ${bpf_obj} ${bpf_tmp}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/${bpf_source} ${BPFTOOL} ${VMLINUX_H}
        COMMENT "Compiling BPF program ${bpf_source}"
    )
    
    # Generate skeleton header
    add_custom_command(
        OUTPUT ${skel_h}
        COMMAND ${BPFTOOL} gen skeleton ${bpf_obj} > ${skel_h}
        DEPENDS ${bpf_obj} ${BPFTOOL}
        COMMENT "Generating skeleton for ${bpf_name}"
    )
    
    # Set variables in parent scope
    set(${bpf_name}_BPF_OBJ ${bpf_obj} PARENT_SCOPE)
    set(${bpf_name}_SKEL_H ${skel_h} PARENT_SCOPE)
endfunction()

# Compile BPF programs
compile_bpf_program(offcputime.bpf.c)
compile_bpf_program(profile.bpf.c)

# Function to create userspace program
function(create_userspace_program prog_name cpp_source)
    set(prog_obj ${OUTPUT_DIR}/${prog_name}.o)
    set(prog_executable ${CMAKE_CURRENT_BINARY_DIR}/${prog_name})
    
    # Compile C++ source
    add_custom_command(
        OUTPUT ${prog_obj}
        COMMAND g++ -g -Wall -std=c++11 ${INCLUDES} -c ${CMAKE_CURRENT_SOURCE_DIR}/src/${cpp_source} -o ${prog_obj}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/${cpp_source} ${${prog_name}_SKEL_H} ${LIBBLAZESYM_HEADER}
        COMMENT "Compiling ${cpp_source}"
    )
    
    # Link executable
    add_custom_command(
        OUTPUT ${prog_executable}
        COMMAND g++ -g -Wall -std=c++11 ${prog_obj} ${LIBBPF_OBJ} ${LIBBLAZESYM_OBJ} 
                -lelf -lz -lrt -ldl -lpthread -lm -o ${prog_executable}
        DEPENDS ${prog_obj} ${LIBBPF_OBJ} ${LIBBLAZESYM_OBJ}
        COMMENT "Linking ${prog_name}"
    )
    
    # Create target
    add_custom_target(${prog_name} ALL DEPENDS ${prog_executable})
endfunction()

# Create userspace programs
create_userspace_program(offcputime offcputime.cpp)
create_userspace_program(profile profile.cpp)

# Create clean target
add_custom_target(clean-all
    COMMAND rm -rf ${OUTPUT_DIR}
    COMMAND rm -f ${CMAKE_CURRENT_BINARY_DIR}/offcputime
    COMMAND rm -f ${CMAKE_CURRENT_BINARY_DIR}/profile
    COMMENT "Cleaning all build artifacts"
)

# Create install target
install(PROGRAMS 
    ${CMAKE_CURRENT_BINARY_DIR}/offcputime
    ${CMAKE_CURRENT_BINARY_DIR}/profile
    DESTINATION bin
) 