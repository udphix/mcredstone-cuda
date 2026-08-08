#include "redstone/batch.h"
#include "signal_device.cuh"
#include "components_device.cuh"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ══════════════════════════════════════════════════════════════════════════
 * WorldBatch — batched launch shape. See include/redstone/batch.h.
 *
 * Kernel indexing: blockIdx.y = instance, (blockIdx.x, threadIdx.x) = cell.
 * That avoids an integer division per thread and keeps the per-cell device
 * code (shared with the single-world path) untouched — we just offset the
 * base pointer by `inst * cells`.
 *
 * There is NO cudaDeviceSynchronize in the tick path. Everything runs on
 * the default stream, so the next blocking cudaMemcpy (probe read, sync,
 * snapshot) is already ordered after the kernels.
 * ══════════════════════════════════════════════════════════════════════════ */

/* Unlike world.cu's CUDA_CHECK, nothing here calls exit() — this library is
 * driven from Python and a long RL run must not be killed by an allocation
 * failure with no traceback. Allocating entry points return NULL/void and
 * report to stderr. */
#define THREADS_PER_BLOCK 256

/* ── Kernels ──────────────────────────────────────────────────────────── */

/* Phase 1: fire scheduled tile ticks (in place on d_current). */
__global__ void kernel_batch_process_events(
    Block* __restrict__ blocks,
    const uint32_t* __restrict__ ticks,
    const uint8_t* __restrict__ done,
    int cells, int size_x, int size_y, int size_z
) {
    int inst = blockIdx.y;
    if (done[inst]) return;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cells) return;

    component_process_event_cell(blocks + (size_t)inst * cells, i,
                                 ticks[inst], size_x, size_y, size_z);
}

/* Phase 2: signal propagation (d_current → d_next).
 * Converged instances still copy src→dst: the buffers are swapped every
 * tick, so skipping the write entirely would leave d_next holding state
 * from two ticks ago. */
__global__ void kernel_batch_propagate(
    const Block* __restrict__ src,
    Block* __restrict__ dst,
    const uint8_t* __restrict__ done,
    const uint8_t* __restrict__ relaxed,
    int32_t* __restrict__ pass_changed,
    int cells, int size_x, int size_y, int size_z
) {
    int inst = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    /* An instance whose wire field already settled earlier in this tick has
     * nothing left to compute: the remaining passes would reproduce the same
     * state. It still has to copy, because the buffers swap every pass.
     *
     * No early return before the ballot below — every thread of the warp must
     * reach __ballot_sync. */
    int active = (i < cells) && !done[inst] && !relaxed[inst];
    int diff = 0;

    if (i < cells) {
        const Block* s = src + (size_t)inst * cells;
        Block*       d = dst + (size_t)inst * cells;

        if (!active) {
            d[i] = s[i];
        } else {
            signal_update_cell(s, d, i, size_x, size_y, size_z);
            Block a = d[i], b = s[i];
            diff = (a.type   != b.type)   | (a.signal != b.signal) |
                   (a.facing != b.facing) | (a.flags  != b.flags)  |
                   (a.state  != b.state)  | (a.tick_scheduled != b.tick_scheduled);
        }
    }

    /* Warp-aggregated so at most cells/32 atomics land per instance. */
    unsigned bd = __ballot_sync(0xFFFFFFFFu, diff);
    if ((threadIdx.x & 31u) == 0u && bd) atomicOr(&pass_changed[inst], 1);
}

/* Between relaxation passes: a pass that changed nothing means this instance's
 * wire field has reached its fixed point, so every later pass in this tick can
 * skip it. Folding per-pass changes into the per-tick flag also keeps the
 * "did this tick change anything" answer that drives convergence. */
__global__ void kernel_batch_fold_relax(
    uint8_t* __restrict__ relaxed,
    int32_t* __restrict__ changed,
    const int32_t* __restrict__ pass_changed,
    const uint8_t* __restrict__ done,
    int32_t* __restrict__ any_active,
    int n_worlds
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_worlds) return;
    if (pass_changed[i]) changed[i] = 1;
    else                 relaxed[i] = 1;

    /* Whether ANY instance still needs another pass. The host reads this to
     * leave the loop early, which matters more than the per-instance skip: a
     * skipped instance still copies its slab, and at batch 2048 that copy
     * traffic dominates the pass. Ending the loop avoids both. */
    if (!relaxed[i] && !done[i]) atomicOr(any_active, 1);
}

/* Phase 4: schedule component events (in place) + its share of convergence
 * detection. The relaxation loop reports its own changes, so this only has to
 * report the ones scheduling itself introduces. */
__global__ void kernel_batch_detect(
    Block* __restrict__ next,
    const uint32_t* __restrict__ ticks,
    const uint8_t* __restrict__ done,
    int32_t* __restrict__ changed,
    int32_t* __restrict__ sched,
    int cells, int size_x, int size_y, int size_z
) {
    int inst = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    /* No early return before the ballot below — every thread of the warp
     * must reach __ballot_sync. */
    int active = (i < cells) && !done[inst];
    int diff = 0, has_sched = 0;

    if (active) {
        Block* n = next + (size_t)inst * cells;

        /* Compare against this cell's own pre-scheduling state rather than
         * against the buffer the tick started from: the relaxation loop has
         * already consumed and swapped that buffer. The propagation passes
         * report their own changes, so the union still answers "did this tick
         * change anything". */
        Block b = n[i];
        component_detect_change_cell(n, i, ticks[inst], size_x, size_y, size_z);

        Block a = n[i];
        diff = (a.type   != b.type)   | (a.signal != b.signal) |
               (a.facing != b.facing) | (a.flags  != b.flags)  |
               (a.state  != b.state)  | (a.tick_scheduled != b.tick_scheduled);
        has_sched = (a.flags & BLOCK_FLAG_SCHEDULED) != 0;
    }

    /* Warp-aggregate so at most cells/32 atomics land per instance. */
    unsigned bd = __ballot_sync(0xFFFFFFFFu, diff);
    unsigned bs = __ballot_sync(0xFFFFFFFFu, has_sched);
    if ((threadIdx.x & 31u) == 0u) {
        if (bd) atomicOr(&changed[inst], 1);
        if (bs) atomicOr(&sched[inst], 1);
    }
}

/* Phase 5: advance per-instance clocks and latch convergence.
 * An instance that changed nothing and holds no pending tile tick is at a
 * genuine fixed point: freeze it (its clock stops too, which is safe
 * precisely because nothing is scheduled against it). */
__global__ void kernel_batch_finalize(
    uint32_t* __restrict__ ticks,
    uint8_t* __restrict__ done,
    const int32_t* __restrict__ changed,
    const int32_t* __restrict__ sched,
    int n_worlds
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_worlds) return;
    if (done[i]) return;

    if (!changed[i] && !sched[i]) { done[i] = 1; return; }
    ticks[i] += 1u;
}

/* Device-side lever writes — mirrors world_set_lever_powered. */
__global__ void kernel_batch_set_levers(
    Block* __restrict__ blocks,
    const int32_t* __restrict__ inst_xyz,
    const uint8_t* __restrict__ on,
    uint8_t* __restrict__ done,
    int count, int cells, int size_x, int size_y, int size_z
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= count) return;

    int inst = inst_xyz[k * 4 + 0];
    int x    = inst_xyz[k * 4 + 1];
    int y    = inst_xyz[k * 4 + 2];
    int z    = inst_xyz[k * 4 + 3];
    if (x < 0 || x >= size_x || y < 0 || y >= size_y || z < 0 || z >= size_z)
        return;

    Block* b = blocks + (size_t)inst * cells + ((size_t)y * size_z + z) * size_x + x;
    if (b->type != BLOCK_LEVER) return;

    if (on[k]) {
        b->signal = 15;
        b->flags |= BLOCK_FLAG_POWERED;
    } else {
        b->signal = 0;
        b->flags &= (uint8_t)~BLOCK_FLAG_POWERED;
    }
    done[inst] = 0;   /* the instance is no longer at a fixed point */
}

__global__ void kernel_batch_read_probes(
    const Block* __restrict__ blocks,
    const int32_t* __restrict__ inst_xyz,
    BatchProbeResult* __restrict__ results,
    int count, int cells, int size_x, int size_y, int size_z
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= count) return;

    int inst = inst_xyz[k * 4 + 0];
    int x    = inst_xyz[k * 4 + 1];
    int y    = inst_xyz[k * 4 + 2];
    int z    = inst_xyz[k * 4 + 3];

    results[k].inst = inst;
    results[k].x = x; results[k].y = y; results[k].z = z;

    if (x >= 0 && x < size_x && y >= 0 && y < size_y && z >= 0 && z < size_z) {
        Block b = blocks[(size_t)inst * cells + ((size_t)y * size_z + z) * size_x + x];
        results[k].type   = b.type;
        results[k].signal = b.signal;
        results[k].flags  = b.flags;
    } else {
        results[k].type = 0; results[k].signal = 0; results[k].flags = 0;
    }
    results[k]._pad = 0;
}

/* ── Host helpers ─────────────────────────────────────────────────────── */

static void batch_report(const char* where) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        fprintf(stderr, "[Batch] %s: %s\n", where, cudaGetErrorString(err));
}

static inline int batch_valid_inst(const WorldBatch* b, int inst) {
    return b && inst >= 0 && inst < b->n_worlds;
}

int world_batch_index(const WorldBatch* batch, int inst, int x, int y, int z) {
    return inst * batch->cells + (y * batch->size_z + z) * batch->size_x + x;
}

int world_batch_in_bounds(const WorldBatch* batch, int x, int y, int z) {
    return x >= 0 && x < batch->size_x &&
           y >= 0 && y < batch->size_y &&
           z >= 0 && z < batch->size_z;
}

/* ── Lifecycle ────────────────────────────────────────────────────────── */

WorldBatch* world_batch_create(int n_worlds, int size_x, int size_y, int size_z) {
    if (n_worlds <= 0 || size_x <= 0 || size_y <= 0 || size_z <= 0) {
        fprintf(stderr, "[Batch] invalid dimensions\n");
        return NULL;
    }
    /* blockIdx.y carries the instance index. */
    if (n_worlds > 65535) {
        fprintf(stderr, "[Batch] n_worlds %d exceeds gridDim.y limit (65535)\n",
                n_worlds);
        return NULL;
    }

    WorldBatch* b = (WorldBatch*)calloc(1, sizeof(WorldBatch));
    if (!b) return NULL;

    b->n_worlds     = n_worlds;
    b->size_x       = size_x;
    b->size_y       = size_y;
    b->size_z       = size_z;
    b->cells        = size_x * size_y * size_z;
    b->total_blocks = n_worlds * b->cells;
    b->dirty        = 0;

    size_t bytes = (size_t)b->total_blocks * sizeof(Block);

    b->h_blocks = (Block*)calloc((size_t)b->total_blocks, sizeof(Block));
    b->h_ticks  = (uint32_t*)calloc((size_t)n_worlds, sizeof(uint32_t));
    b->h_done   = (uint8_t*)calloc((size_t)n_worlds, sizeof(uint8_t));
    if (!b->h_blocks || !b->h_ticks || !b->h_done) {
        fprintf(stderr, "[Batch] host allocation failed (%zu bytes)\n", bytes);
        world_batch_destroy(b);
        return NULL;
    }

    cudaError_t err = cudaSuccess;
    err = err ? err : cudaMalloc(&b->d_blocks_a, bytes);
    err = err ? err : cudaMalloc(&b->d_blocks_b, bytes);
    err = err ? err : cudaMalloc(&b->d_ticks,   (size_t)n_worlds * sizeof(uint32_t));
    err = err ? err : cudaMalloc(&b->d_done,    (size_t)n_worlds * sizeof(uint8_t));
    err = err ? err : cudaMalloc(&b->d_changed, (size_t)n_worlds * sizeof(int32_t));
    err = err ? err : cudaMalloc(&b->d_sched,   (size_t)n_worlds * sizeof(int32_t));
    err = err ? err : cudaMalloc(&b->d_pass_changed,
                                 (size_t)n_worlds * sizeof(int32_t));
    err = err ? err : cudaMalloc(&b->d_relaxed, (size_t)n_worlds * sizeof(uint8_t));
    err = err ? err : cudaMalloc(&b->d_any_active, sizeof(int32_t));
    if (err != cudaSuccess) {
        fprintf(stderr, "[Batch] device allocation failed (%zu bytes x2): %s\n",
                bytes, cudaGetErrorString(err));
        world_batch_destroy(b);
        return NULL;
    }

    cudaMemset(b->d_blocks_a, 0, bytes);
    cudaMemset(b->d_blocks_b, 0, bytes);
    cudaMemset(b->d_ticks, 0, (size_t)n_worlds * sizeof(uint32_t));
    cudaMemset(b->d_done,  0, (size_t)n_worlds * sizeof(uint8_t));

    b->d_current = b->d_blocks_a;
    b->d_next    = b->d_blocks_b;

    return b;
}

void world_batch_destroy(WorldBatch* b) {
    if (!b) return;
    if (b->d_blocks_a)        cudaFree(b->d_blocks_a);
    if (b->d_blocks_b)        cudaFree(b->d_blocks_b);
    if (b->d_ticks)           cudaFree(b->d_ticks);
    if (b->d_done)            cudaFree(b->d_done);
    if (b->d_changed)         cudaFree(b->d_changed);
    if (b->d_sched)           cudaFree(b->d_sched);
    if (b->d_pass_changed)    cudaFree(b->d_pass_changed);
    if (b->d_relaxed)         cudaFree(b->d_relaxed);
    if (b->d_any_active)      cudaFree(b->d_any_active);
    if (b->d_probe_positions) cudaFree(b->d_probe_positions);
    if (b->d_probe_results)   cudaFree(b->d_probe_results);
    if (b->d_lever_positions) cudaFree(b->d_lever_positions);
    if (b->d_lever_values)    cudaFree(b->d_lever_values);
    if (b->h_blocks)          free(b->h_blocks);
    if (b->h_ticks)           free(b->h_ticks);
    if (b->h_done)            free(b->h_done);
    if (b->h_probe_positions) free(b->h_probe_positions);
    if (b->h_probe_results)   free(b->h_probe_results);
    free(b);
}

/* ── Block access ─────────────────────────────────────────────────────── */

void world_batch_set_block(WorldBatch* b, int inst, int x, int y, int z,
                           BlockType type, Direction facing, uint16_t state) {
    if (!batch_valid_inst(b, inst)) {
        fprintf(stderr, "[Batch] set_block: bad instance %d\n", inst);
        return;
    }
    if (!world_batch_in_bounds(b, x, y, z)) {
        fprintf(stderr, "[Batch] set_block out of bounds: (%d,%d,%d)\n", x, y, z);
        return;
    }

    Block* block = &b->h_blocks[world_batch_index(b, inst, x, y, z)];

    /* Identical initialization to world_set_block — keep in lockstep. */
    block->type   = (uint8_t)type;
    block->facing = (uint8_t)facing;
    block->state  = state;
    block->signal = 0;
    block->flags  = 0;
    block->tick_scheduled = 0;

    if (type == BLOCK_LEVER || type == BLOCK_REDSTONE_BLOCK) {
        block->signal = 15;
        block->flags |= BLOCK_FLAG_POWERED;
    }
    if (type == BLOCK_REDSTONE_TORCH) {
        block->signal = 15;
        block->flags |= BLOCK_FLAG_POWERED;
    }
    if (type == BLOCK_REPEATER) {
        uint16_t delay = state & 0xF;
        if (delay < 1) delay = 1;
        if (delay > 4) delay = 4;
        block->state = delay;
        block->signal = 0;
    }

    b->h_done[inst] = 0;
    b->dirty = 1;
}

Block world_batch_get_block(const WorldBatch* b, int inst, int x, int y, int z) {
    Block empty = {0,0,0,0,0,0};
    if (!batch_valid_inst(b, inst)) return empty;
    if (!world_batch_in_bounds(b, x, y, z)) return empty;
    return b->h_blocks[world_batch_index(b, inst, x, y, z)];
}

void world_batch_clear(WorldBatch* b) {
    if (!b) return;
    memset(b->h_blocks, 0, (size_t)b->total_blocks * sizeof(Block));
    memset(b->h_ticks, 0, (size_t)b->n_worlds * sizeof(uint32_t));
    memset(b->h_done,  0, (size_t)b->n_worlds * sizeof(uint8_t));
    b->dirty = 1;
}

void world_batch_clear_instance(WorldBatch* b, int inst) {
    if (!batch_valid_inst(b, inst)) return;
    memset(b->h_blocks + (size_t)inst * b->cells, 0,
           (size_t)b->cells * sizeof(Block));
    b->h_ticks[inst] = 0;
    b->h_done[inst]  = 0;
    b->dirty = 1;
}

void world_batch_copy_instance(WorldBatch* b, int src_inst, int dst_inst) {
    if (!batch_valid_inst(b, src_inst) || !batch_valid_inst(b, dst_inst)) return;
    if (src_inst == dst_inst) return;
    memcpy(b->h_blocks + (size_t)dst_inst * b->cells,
           b->h_blocks + (size_t)src_inst * b->cells,
           (size_t)b->cells * sizeof(Block));
    b->h_ticks[dst_inst] = b->h_ticks[src_inst];
    b->h_done[dst_inst]  = 0;
    b->dirty = 1;
}

/* ── GPU sync ─────────────────────────────────────────────────────────── */

void world_batch_sync_to_gpu(WorldBatch* b) {
    size_t bytes = (size_t)b->total_blocks * sizeof(Block);
    cudaMemcpy(b->d_current, b->h_blocks, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(b->d_next,    b->h_blocks, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(b->d_ticks, b->h_ticks, (size_t)b->n_worlds * sizeof(uint32_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(b->d_done, b->h_done, (size_t)b->n_worlds * sizeof(uint8_t),
               cudaMemcpyHostToDevice);
    b->dirty = 0;
    batch_report("sync_to_gpu");
}

void world_batch_sync_from_gpu(WorldBatch* b) {
    size_t bytes = (size_t)b->total_blocks * sizeof(Block);
    cudaMemcpy(b->h_blocks, b->d_current, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(b->h_ticks, b->d_ticks, (size_t)b->n_worlds * sizeof(uint32_t),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(b->h_done, b->d_done, (size_t)b->n_worlds * sizeof(uint8_t),
               cudaMemcpyDeviceToHost);
    batch_report("sync_from_gpu");
}

void world_batch_copy_host_buffer(const WorldBatch* b, Block* out) {
    if (!b || !out) return;
    memcpy(out, b->h_blocks, (size_t)b->total_blocks * sizeof(Block));
}

/* ── Simulation ───────────────────────────────────────────────────────── */

static void batch_run_ticks(WorldBatch* b, int n) {
    if (n <= 0) return;
    if (b->dirty) world_batch_sync_to_gpu(b);

    dim3 cell_grid((b->cells + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK,
                   b->n_worlds);
    dim3 cell_block(THREADS_PER_BLOCK);
    int  fin_grid = (b->n_worlds + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    size_t flag_bytes = (size_t)b->n_worlds * sizeof(int32_t);

    for (int t = 0; t < n; t++) {
        cudaMemsetAsync(b->d_changed, 0, flag_bytes);
        cudaMemsetAsync(b->d_sched,   0, flag_bytes);

        /* 1. TILE TICKS */
        kernel_batch_process_events<<<cell_grid, cell_block>>>(
            b->d_current, b->d_ticks, b->d_done,
            b->cells, b->size_x, b->size_y, b->size_z);

        /* 2. BLOCK UPDATES — relax the wire field to its fixed point.
         * See DUST_RELAX_PASSES in kernels.h: Minecraft settles a whole dust
         * network inside the disturbing tick, and 16 passes is a hard bound
         * on reaching that fixed point. Each pass swaps the buffers, so the
         * relaxed state ends up in d_current. */
        cudaMemsetAsync(b->d_relaxed, 0, (size_t)b->n_worlds * sizeof(uint8_t));

        for (int k = 0; k < DUST_RELAX_PASSES; k++) {
            cudaMemsetAsync(b->d_pass_changed, 0, flag_bytes);

            kernel_batch_propagate<<<cell_grid, cell_block>>>(
                b->d_current, b->d_next, b->d_done, b->d_relaxed,
                b->d_pass_changed,
                b->cells, b->size_x, b->size_y, b->size_z);

            cudaMemsetAsync(b->d_any_active, 0, sizeof(int32_t));
            kernel_batch_fold_relax<<<fin_grid, THREADS_PER_BLOCK>>>(
                b->d_relaxed, b->d_changed, b->d_pass_changed, b->d_done,
                b->d_any_active, b->n_worlds);

            Block* tmp = b->d_current;
            b->d_current = b->d_next;
            b->d_next = tmp;

            /* One 4-byte readback per pass. It serialises the stream, but a
             * typical circuit settles in 2-4 passes, so it buys back a dozen
             * full-batch passes worth of memory traffic. */
            int32_t any = 0;
            cudaMemcpy(&any, b->d_any_active, sizeof(int32_t),
                       cudaMemcpyDeviceToHost);
            if (!any) break;
        }

        /* 3. BLOCK EVENTS — TODO v0.5 (piston movement) */

        /* 4. SCHEDULING (+ its own share of convergence detection) */
        kernel_batch_detect<<<cell_grid, cell_block>>>(
            b->d_current, b->d_ticks, b->d_done,
            b->d_changed, b->d_sched,
            b->cells, b->size_x, b->size_y, b->size_z);

        /* 5. advance clocks / latch fixed points */
        kernel_batch_finalize<<<fin_grid, THREADS_PER_BLOCK>>>(
            b->d_ticks, b->d_done, b->d_changed, b->d_sched, b->n_worlds);
    }

    batch_report("tick");
}

void world_batch_tick(WorldBatch* b, int n) {
    if (!b) return;
    batch_run_ticks(b, n);
}

void world_batch_refresh_done(WorldBatch* b) {
    if (!b) return;
    cudaMemcpy(b->h_done, b->d_done, (size_t)b->n_worlds * sizeof(uint8_t),
               cudaMemcpyDeviceToHost);
}

int world_batch_done_count(WorldBatch* b) {
    if (!b) return 0;
    int c = 0;
    for (int i = 0; i < b->n_worlds; i++) c += (b->h_done[i] != 0);
    return c;
}

int world_batch_instance_done(const WorldBatch* b, int inst) {
    if (!batch_valid_inst(b, inst)) return 0;
    return b->h_done[inst] != 0;
}

uint32_t world_batch_instance_tick(const WorldBatch* b, int inst) {
    if (!batch_valid_inst(b, inst)) return 0;
    return b->h_ticks[inst];
}

int world_batch_tick_until_stable(WorldBatch* b, int max_ticks, int check_every) {
    if (!b || max_ticks <= 0) return 0;
    if (check_every <= 0) check_every = 4;

    int run = 0;
    while (run < max_ticks) {
        int chunk = check_every;
        if (run + chunk > max_ticks) chunk = max_ticks - run;
        batch_run_ticks(b, chunk);
        run += chunk;

        world_batch_refresh_done(b);
        int all_done = 1;
        for (int i = 0; i < b->n_worlds; i++) {
            if (!b->h_done[i]) { all_done = 0; break; }
        }
        if (all_done) break;
    }
    return run;
}

/* ── Device-side lever writes ─────────────────────────────────────────── */

void world_batch_set_levers(WorldBatch* b, const int32_t* inst_xyz,
                            const uint8_t* on, int count) {
    if (!b || count <= 0) return;
    if (b->dirty) world_batch_sync_to_gpu(b);

    if (count > b->lever_capacity) {
        if (b->d_lever_positions) cudaFree(b->d_lever_positions);
        if (b->d_lever_values)    cudaFree(b->d_lever_values);
        b->d_lever_positions = NULL;
        b->d_lever_values = NULL;
        if (cudaMalloc(&b->d_lever_positions, (size_t)count * 4 * sizeof(int32_t))
                != cudaSuccess ||
            cudaMalloc(&b->d_lever_values, (size_t)count * sizeof(uint8_t))
                != cudaSuccess) {
            fprintf(stderr, "[Batch] set_levers: allocation failed\n");
            b->lever_capacity = 0;
            return;
        }
        b->lever_capacity = count;
    }

    /* Bounds-check instance indices on the host — a bad index would corrupt
     * a neighbouring instance's memory rather than fault. */
    for (int k = 0; k < count; k++) {
        int inst = inst_xyz[k * 4 + 0];
        if (inst < 0 || inst >= b->n_worlds) {
            fprintf(stderr, "[Batch] set_levers: bad instance %d at entry %d\n",
                    inst, k);
            return;
        }
    }

    cudaMemcpy(b->d_lever_positions, inst_xyz,
               (size_t)count * 4 * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(b->d_lever_values, on,
               (size_t)count * sizeof(uint8_t), cudaMemcpyHostToDevice);

    int grid = (count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    kernel_batch_set_levers<<<grid, THREADS_PER_BLOCK>>>(
        b->d_current, b->d_lever_positions, b->d_lever_values, b->d_done,
        count, b->cells, b->size_x, b->size_y, b->size_z);

    /* Host mirror of `done` would otherwise claim these are still frozen. */
    for (int k = 0; k < count; k++) b->h_done[inst_xyz[k * 4 + 0]] = 0;

    batch_report("set_levers");
}

/* ── Probes ───────────────────────────────────────────────────────────── */

void world_batch_set_probes(WorldBatch* b, const int32_t* inst_xyz, int count) {
    if (!b || count < 0) return;

    if (count > b->probe_capacity) {
        if (b->d_probe_positions) cudaFree(b->d_probe_positions);
        if (b->d_probe_results)   cudaFree(b->d_probe_results);
        free(b->h_probe_positions);
        free(b->h_probe_results);
        b->d_probe_positions = NULL; b->d_probe_results = NULL;
        b->h_probe_positions = NULL; b->h_probe_results = NULL;

        if (cudaMalloc(&b->d_probe_positions,
                       (size_t)count * 4 * sizeof(int32_t)) != cudaSuccess ||
            cudaMalloc(&b->d_probe_results,
                       (size_t)count * sizeof(BatchProbeResult)) != cudaSuccess) {
            fprintf(stderr, "[Batch] set_probes: device allocation failed\n");
            b->probe_capacity = 0; b->probe_count = 0;
            return;
        }
        b->h_probe_positions = (int32_t*)calloc((size_t)count * 4, sizeof(int32_t));
        b->h_probe_results   = (BatchProbeResult*)calloc((size_t)count,
                                                    sizeof(BatchProbeResult));
        b->probe_capacity = count;
    }

    b->probe_count = count;
    if (count == 0) return;

    for (int k = 0; k < count; k++) {
        int inst = inst_xyz[k * 4 + 0];
        if (inst < 0 || inst >= b->n_worlds) {
            fprintf(stderr, "[Batch] set_probes: bad instance %d at entry %d\n",
                    inst, k);
            b->probe_count = 0;
            return;
        }
    }
    memcpy(b->h_probe_positions, inst_xyz, (size_t)count * 4 * sizeof(int32_t));
}

const BatchProbeResult* world_batch_read_probes(WorldBatch* b) {
    if (!b || b->probe_count == 0) return b ? b->h_probe_results : NULL;
    if (b->dirty) world_batch_sync_to_gpu(b);

    cudaMemcpy(b->d_probe_positions, b->h_probe_positions,
               (size_t)b->probe_count * 4 * sizeof(int32_t),
               cudaMemcpyHostToDevice);

    int grid = (b->probe_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    kernel_batch_read_probes<<<grid, THREADS_PER_BLOCK>>>(
        b->d_current, b->d_probe_positions, b->d_probe_results,
        b->probe_count, b->cells, b->size_x, b->size_y, b->size_z);

    cudaMemcpy(b->h_probe_results, b->d_probe_results,
               (size_t)b->probe_count * sizeof(BatchProbeResult),
               cudaMemcpyDeviceToHost);
    batch_report("read_probes");
    return b->h_probe_results;
}

/* ── Snapshots ────────────────────────────────────────────────────────── */

WorldBatchSnapshot* world_batch_save(WorldBatch* b) {
    if (!b) return NULL;
    if (b->dirty) world_batch_sync_to_gpu(b);

    WorldBatchSnapshot* s =
        (WorldBatchSnapshot*)calloc(1, sizeof(WorldBatchSnapshot));
    if (!s) return NULL;

    size_t bytes = (size_t)b->total_blocks * sizeof(Block);
    s->n_worlds = b->n_worlds;
    s->cells    = b->cells;

    cudaError_t err = cudaSuccess;
    err = err ? err : cudaMalloc(&s->d_snapshot, bytes);
    err = err ? err : cudaMalloc(&s->d_ticks, (size_t)b->n_worlds * sizeof(uint32_t));
    err = err ? err : cudaMalloc(&s->d_done,  (size_t)b->n_worlds * sizeof(uint8_t));
    if (err != cudaSuccess) {
        fprintf(stderr, "[Batch] save: allocation failed: %s\n",
                cudaGetErrorString(err));
        world_batch_snapshot_destroy(s);
        return NULL;
    }

    cudaMemcpy(s->d_snapshot, b->d_current, bytes, cudaMemcpyDeviceToDevice);
    cudaMemcpy(s->d_ticks, b->d_ticks, (size_t)b->n_worlds * sizeof(uint32_t),
               cudaMemcpyDeviceToDevice);
    cudaMemcpy(s->d_done, b->d_done, (size_t)b->n_worlds * sizeof(uint8_t),
               cudaMemcpyDeviceToDevice);
    batch_report("save");
    return s;
}

void world_batch_restore(WorldBatch* b, const WorldBatchSnapshot* s) {
    if (!b || !s) return;
    if (s->n_worlds != b->n_worlds || s->cells != b->cells) {
        fprintf(stderr, "[Batch] restore: snapshot shape mismatch\n");
        return;
    }
    size_t bytes = (size_t)b->total_blocks * sizeof(Block);
    cudaMemcpy(b->d_current, s->d_snapshot, bytes, cudaMemcpyDeviceToDevice);
    cudaMemcpy(b->d_next,    s->d_snapshot, bytes, cudaMemcpyDeviceToDevice);
    cudaMemcpy(b->d_ticks, s->d_ticks, (size_t)b->n_worlds * sizeof(uint32_t),
               cudaMemcpyDeviceToDevice);
    cudaMemcpy(b->d_done, s->d_done, (size_t)b->n_worlds * sizeof(uint8_t),
               cudaMemcpyDeviceToDevice);
    /* Keep the host view of convergence coherent (n bytes); the block mirror
     * is left stale on purpose — see the contract in batch.h. */
    cudaMemcpy(b->h_done, s->d_done, (size_t)b->n_worlds * sizeof(uint8_t),
               cudaMemcpyDeviceToHost);
    b->dirty = 0;
    batch_report("restore");
}

void world_batch_restore_instance(WorldBatch* b, const WorldBatchSnapshot* s,
                                  int inst) {
    if (!b || !s || !batch_valid_inst(b, inst)) return;
    if (s->cells != b->cells) {
        fprintf(stderr, "[Batch] restore_instance: shape mismatch\n");
        return;
    }
    size_t off   = (size_t)inst * b->cells;
    size_t bytes = (size_t)b->cells * sizeof(Block);
    cudaMemcpy(b->d_current + off, s->d_snapshot + off, bytes,
               cudaMemcpyDeviceToDevice);
    cudaMemcpy(b->d_next + off,    s->d_snapshot + off, bytes,
               cudaMemcpyDeviceToDevice);
    cudaMemcpy(b->d_ticks + inst, s->d_ticks + inst, sizeof(uint32_t),
               cudaMemcpyDeviceToDevice);
    cudaMemcpy(b->d_done + inst, s->d_done + inst, sizeof(uint8_t),
               cudaMemcpyDeviceToDevice);
    cudaMemcpy(b->h_done + inst, s->d_done + inst, sizeof(uint8_t),
               cudaMemcpyDeviceToHost);
    batch_report("restore_instance");
}

void world_batch_snapshot_destroy(WorldBatchSnapshot* s) {
    if (!s) return;
    if (s->d_snapshot) cudaFree(s->d_snapshot);
    if (s->d_ticks)    cudaFree(s->d_ticks);
    if (s->d_done)     cudaFree(s->d_done);
    free(s);
}
