cmake_minimum_required(VERSION 3.16)
project(systemscope)

# Force Unix Makefiles generator to avoid Ninja issues
if(NOT CMAKE_GENERATOR STREQUAL "Unix Makefiles")
    message(WARNING "This project is designed for Unix Makefiles generator. Current generator: ${CMAKE_GENERATOR}")
endif()

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Sanitizer options
option(ENABLE_ASAN "Enable AddressSanitizer" ON)
option(ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)
option(ENABLE_TSAN "Enable ThreadSanitizer" OFF)
option(ENABLE_MSAN "Enable MemorySanitizer" OFF)

# Set sanitizer flags
set(SANITIZER_FLAGS "")
set(SANITIZER_LINK_FLAGS "")

if(ENABLE_ASAN)
    set(SANITIZER_FLAGS "${SANITIZER_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
    set(SANITIZER_LINK_FLAGS "${SANITIZER_LINK_FLAGS} -fsanitize=address")
    message(STATUS "AddressSanitizer enabled")
endif()

if(ENABLE_UBSAN)
    set(SANITIZER_FLAGS "${SANITIZER_FLAGS} -fsanitize=undefined -fno-omit-frame-pointer")
    set(SANITIZER_LINK_FLAGS "${SANITIZER_LINK_FLAGS} -fsanitize=undefined")
    message(STATUS "UndefinedBehaviorSanitizer enabled")
endif()

if(ENABLE_TSAN)
    if(ENABLE_ASAN)
        message(FATAL_ERROR "ThreadSanitizer cannot be used with AddressSanitizer")
    endif()
    set(SANITIZER_FLAGS "${SANITIZER_FLAGS} -fsanitize=thread -fno-omit-frame-pointer")
    set(SANITIZER_LINK_FLAGS "${SANITIZER_LINK_FLAGS} -fsanitize=thread")
    message(STATUS "ThreadSanitizer enabled")
endif()

if(ENABLE_MSAN)
    if(ENABLE_ASAN OR ENABLE_TSAN)
        message(FATAL_ERROR "MemorySanitizer cannot be used with AddressSanitizer or ThreadSanitizer")
    endif()
    set(SANITIZER_FLAGS "${SANITIZER_FLAGS} -fsanitize=memory -fno-omit-frame-pointer")
    set(SANITIZER_LINK_FLAGS "${SANITIZER_LINK_FLAGS} -fsanitize=memory")
    message(STATUS "MemorySanitizer enabled")
endif()

# Apply sanitizer flags globally if any sanitizer is enabled
if(SANITIZER_FLAGS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SANITIZER_FLAGS}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${SANITIZER_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${SANITIZER_LINK_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${SANITIZER_LINK_FLAGS}")
endif()

# Add local third-party dependencies as subdirectories
add_subdirectory(src/third_party/Catch2)
add_subdirectory(src/third_party/argparse)
add_subdirectory(src/third_party/spdlog)

# Find required packages
find_program(CLANG_EXECUTABLE NAMES clang REQUIRED)
find_program(CARGO_EXECUTABLE NAMES cargo REQUIRED)
find_program(MAKE_EXECUTABLE NAMES make REQUIRED)

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
    COMMAND ${MAKE_EXECUTABLE} -C ${LIBBPF_SRC} BUILD_STATIC_ONLY=1 
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
    COMMAND ${MAKE_EXECUTABLE} ARCH= CROSS_COMPILE= OUTPUT=${OUTPUT_DIR}/bpftool/ -C ${BPFTOOL_SRC} bootstrap
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
compile_bpf_program(collectors/offcpu/offcputime.bpf.c)
compile_bpf_program(collectors/oncpu/profile.bpf.c)

# Create a library for shared source files (for easier testing)
add_library(profiler_lib STATIC
    src/collectors/oncpu/profile.cpp
    src/collectors/offcpu/offcputime.cpp
    src/analyzers/wallclock_analyzer.cpp
    src/analyzers/symbol_resolver.cpp
    src/analyzers/flamegraph_view.cpp
    src/flamegraph_generator.cpp
)

# Set C++ standard for the library
set_target_properties(profiler_lib PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# Add include directories for the library
target_include_directories(profiler_lib PUBLIC
    ${OUTPUT_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/libbpf/include/uapi
    ${CMAKE_CURRENT_SOURCE_DIR}/vmlinux/${ARCH}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Add dependencies for the library
add_dependencies(profiler_lib 
    libbpf_target
    blazesym_target
    profile_bpf_target
    offcputime_bpf_target
)

# Link libraries for the profiler_lib
target_link_libraries(profiler_lib PUBLIC
    ${LIBBPF_OBJ}
    ${LIBBLAZESYM_OBJ}
    elf
    z
    rt
    dl
    pthread
    m
)

# Create the main executable using the library
add_executable(profiler
    src/main.cpp
    src/args_parser.cpp
)

# Set C++ standard for the target
set_target_properties(profiler PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# Add include directories for the main executable
target_include_directories(profiler PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/third_party/spdlog/include
    ${OUTPUT_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/libbpf/include/uapi
    ${CMAKE_CURRENT_SOURCE_DIR}/vmlinux/${ARCH}
)

# Link the main executable with the library
target_link_libraries(profiler PRIVATE 
    profiler_lib
    argparse
    spdlog::spdlog
)

# Enable testing
enable_testing()

# Create test executable
add_executable(profiler_tests
    tests/test_main.cpp
    tests/test_flamegraph_view.cpp
    tests/test_profile_collector.cpp
    tests/test_offcputime_collector.cpp
)

# Set C++ standard for tests
set_target_properties(profiler_tests PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# Add include directories for tests
target_include_directories(profiler_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${OUTPUT_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/libbpf/include/uapi
    ${CMAKE_CURRENT_SOURCE_DIR}/vmlinux/${ARCH}
)

# Link test executable with the library and Catch2
target_link_libraries(profiler_tests PRIVATE 
    profiler_lib
    Catch2::Catch2WithMain
)

# Add test to CTest
add_test(NAME unit_tests COMMAND profiler_tests)

# Create custom targets for dependencies
add_custom_target(libbpf_target DEPENDS ${LIBBPF_OBJ})
add_custom_target(blazesym_target DEPENDS ${LIBBLAZESYM_OBJ} ${LIBBLAZESYM_HEADER})
add_custom_target(profile_bpf_target DEPENDS ${profile_SKEL_H})
add_custom_target(offcputime_bpf_target DEPENDS ${offcputime_SKEL_H})

# Create clean target
add_custom_target(clean-all
    COMMAND rm -rf ${OUTPUT_DIR}
    COMMENT "Cleaning all build artifacts"
)

# Create install target
install(TARGETS profiler
    RUNTIME DESTINATION bin
) 