#include "redstone/kernels.h"
#include "redstone/types.h"
#include "signal_device.cuh"
#include <cuda_runtime.h>
#include <stdio.h>

/* ══════════════════════════════════════════════════════════════════════════
 * Signal propagation — SINGLE-WORLD launch shape.
 *
 * All the actual redstone rules live in `signal_device.cuh` so the batched
 * launch shape (kernels/batch.cu) runs byte-identical semantics. Do not
 * reimplement rules here.
 * ══════════════════════════════════════════════════════════════════════════ */

__global__ void kernel_signal_propagate(
    const Block* __restrict__ src,
    Block* __restrict__ dst,
    int size_x, int size_y, int size_z
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = size_x * size_y * size_z;
    if (i >= total) return;

    signal_update_cell(src, dst, i, size_x, size_y, size_z);
}

/* ── Host wrapper ─────────────────────────────────────────────────────── */

void kernel_propagate_signal(
    const Block* d_src, Block* d_dst,
    int size_x, int size_y, int size_z
) {
    int total = size_x * size_y * size_z;
    int threads = 256;
    int grid = (total + threads - 1) / threads;
    kernel_signal_propagate<<<grid, threads>>>(d_src, d_dst, size_x, size_y, size_z);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        fprintf(stderr, "[Signal] Kernel error: %s\n", cudaGetErrorString(err));
    cudaDeviceSynchronize();
}
