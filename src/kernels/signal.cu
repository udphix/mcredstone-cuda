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
    int32_t* __restrict__ changed,
    int size_x, int size_y, int size_z
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = size_x * size_y * size_z;

    /* No early return before the ballot — every thread of the warp must
     * reach __ballot_sync. */
    int diff = 0;
    if (i < total) {
        signal_update_cell(src, dst, i, size_x, size_y, size_z);
        Block a = dst[i], b = src[i];
        diff = (a.type   != b.type)   | (a.signal != b.signal) |
               (a.facing != b.facing) | (a.flags  != b.flags)  |
               (a.state  != b.state)  | (a.tick_scheduled != b.tick_scheduled);
    }

    unsigned bd = __ballot_sync(0xFFFFFFFFu, diff);
    if ((threadIdx.x & 31u) == 0u && bd) atomicOr(changed, 1);
}

/* ── Host wrapper ─────────────────────────────────────────────────────── */

void kernel_propagate_signal(
    const Block* d_src, Block* d_dst, int32_t* d_changed,
    int size_x, int size_y, int size_z
) {
    int total = size_x * size_y * size_z;
    int threads = 256;
    int grid = (total + threads - 1) / threads;
    kernel_signal_propagate<<<grid, threads>>>(d_src, d_dst, d_changed,
                                               size_x, size_y, size_z);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        fprintf(stderr, "[Signal] Kernel error: %s\n", cudaGetErrorString(err));
    cudaDeviceSynchronize();
}
