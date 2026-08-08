#include "redstone/kernels.h"
#include "redstone/types.h"
#include "components_device.cuh"
#include <cuda_runtime.h>
#include <stdio.h>

/* ══════════════════════════════════════════════════════════════════════════
 * Torch / repeater state machine — SINGLE-WORLD launch shape.
 *
 * Rules live in `components_device.cuh`, shared with the batched path.
 * ══════════════════════════════════════════════════════════════════════════ */

__global__ void kernel_process_events(
    Block* blocks,
    uint32_t current_tick,
    int size_x, int size_y, int size_z
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = size_x * size_y * size_z;
    if (idx >= total) return;

    component_process_event_cell(blocks, idx, current_tick,
                                 size_x, size_y, size_z);
}

__global__ void kernel_detect_changes(
    Block* blocks,
    uint32_t current_tick,
    int size_x, int size_y, int size_z
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = size_x * size_y * size_z;
    if (idx >= total) return;

    component_detect_change_cell(blocks, idx, current_tick,
                                 size_x, size_y, size_z);
}

/* ── Host-callable wrappers ───────────────────────────────────────────── */

void kernel_process_tick_events(
    Block* d_blocks,
    uint32_t current_tick,
    int size_x, int size_y, int size_z
) {
    int total = size_x * size_y * size_z;
    int threads = 256;
    int grid = (total + threads - 1) / threads;

    kernel_process_events<<<grid, threads>>>(
        d_blocks, current_tick, size_x, size_y, size_z
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[Components] process_events error: %s\n",
                cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
}

void kernel_detect_component_changes(
    Block* d_blocks,
    uint32_t current_tick,
    int size_x, int size_y, int size_z
) {
    int total = size_x * size_y * size_z;
    int threads = 256;
    int grid = (total + threads - 1) / threads;

    kernel_detect_changes<<<grid, threads>>>(
        d_blocks, current_tick, size_x, size_y, size_z
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[Components] detect_changes error: %s\n",
                cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
}
