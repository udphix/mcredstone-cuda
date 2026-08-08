cmake_minimum_required(VERSION 3.18)

# CUDA architecture. Defaults to 8.6 (RTX 30-series — the card this was
# developed on). Set it for your GPU:
#
#     cmake -B build -DCMAKE_CUDA_ARCHITECTURES=89     # e.g. RTX 4090
#
# This MUST come before project(): enabling the CUDA language initializes
# CMAKE_CUDA_ARCHITECTURES to the compiler's own default (observed: 75), so
# a guard placed after project() would never fire.
#
# `native` is deliberately not the default — on CMake 4.3 + CUDA 13.2 it was
# observed resolving to 75 on an sm_86 card. That still runs, because the
# embedded PTX is JIT-compiled, but it silently gives up performance. A
# wrong-but-quiet default is worse than one you set on purpose.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

project(mcredstone-cuda LANGUAGES C CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)

message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES} "
               "(override with -DCMAKE_CUDA_ARCHITECTURES=<cc>)")

# CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/third_party)

# Source files
set(SOURCES
    src/block_registry.c
    src/schem_import.c
    src/schem_export.c
    third_party/miniz.c
    src/world.cu
    src/kernels/signal.cu
    src/kernels/components.cu
    src/kernels/movement.cu
    src/kernels/batch.cu
)

# Main executable
add_executable(redstone-sim
    src/main.cu
    ${SOURCES}
)

target_link_libraries(redstone-sim PRIVATE cudart)

# Test executable
add_executable(redstone-test
    tests/test_dust.cu
    ${SOURCES}
)

target_link_libraries(redstone-test PRIVATE cudart)

# dump_sim: debug tool — load a schem, simulate, dump final block state
add_executable(dump_sim
    tools/dump_sim.cu
    ${SOURCES}
)

target_link_libraries(dump_sim PRIVATE cudart)

# Shared library for Python bindings (ctypes). Loaded by
# bindings/redstone_sim/_core.py via ctypes.CDLL().
#
# WINDOWS_EXPORT_ALL_SYMBOLS auto-exports symbols so ctypes can find them
# without __declspec(dllexport) annotations on every function.
add_library(redstone_sim SHARED ${SOURCES})
set_target_properties(redstone_sim PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)
target_link_libraries(redstone_sim PRIVATE cudart)
