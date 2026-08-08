#ifndef REDSTONE_BATCH_H
#define REDSTONE_BATCH_H

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ══════════════════════════════════════════════════════════════════════════
 * WorldBatch — N independent worlds, one kernel launch.
 *
 * WHY: a single 16^3 world is 4096 cells = 16 warps. On a 3060 Ti that is
 * ~0.1% of the machine, so `World` is 100% launch-latency-bound (~80 us per
 * tick regardless of world size). A batch amortizes the launch over N worlds
 * and gets ~N times the throughput for the same wall clock — which is what
 * RL rollouts and brute-force circuit search actually need.
 *
 * MEMORY LAYOUT: instances are contiguous slabs of `cells = sx*sy*sz` blocks:
 *     block (inst, x, y, z) = d_current[inst * cells + (y*sz + z)*sx + x]
 * Every neighbour lookup is bounds-checked against (sx,sy,sz), so worlds
 * cannot leak signal into each other.
 *
 * SEMANTICS: identical to `World` — same device code paths (see
 * kernels/signal_device.cuh + kernels/components_device.cuh), same tick
 * order (tile ticks → block updates → scheduling). `tests/test_batch_
 * equivalence.py` asserts bit-identical state against the single-world path.
 *
 * PER-INSTANCE CLOCKS: each instance owns its game tick, so restoring one
 * instance from a snapshot taken at a different tick keeps its pending
 * tile ticks (`tick_scheduled`) consistent.
 *
 * CONVERGENCE: an instance is `done` when a full tick produced no state
 * change AND no block holds a pending tile tick. That is a genuine fixed
 * point, so further ticks are no-ops and the instance is skipped.
 * `world_batch_tick_until_stable` uses this to stop early instead of
 * burning a fixed 30-40 ticks per truth-table row.
 *
 * HOST↔GPU CONTRACT (differs from World — read this):
 *   - `set_block` writes the host mirror and marks the batch dirty; the
 *     next tick uploads the WHOLE batch. Use it for scene construction.
 *   - `set_levers` writes GPU memory directly via a tiny kernel, with no
 *     host round-trip. Use it inside evaluation loops (toggling inputs per
 *     truth-table row) — a full re-upload there would dominate runtime.
 *   - `restore` / `restore_instance` touch GPU memory only and leave the
 *     host mirror stale. Call `world_batch_sync_from_gpu` before reading
 *     blocks on the host, or use probes.
 *   - Do NOT call `set_block` after `set_levers`/`restore` without syncing
 *     from GPU first: the dirty-upload would clobber the device state.
 * ══════════════════════════════════════════════════════════════════════════ */

/* Probe: like ProbeResult but carries the instance index. */
typedef struct {
    int32_t  inst;
    int32_t  x, y, z;
    uint8_t  type;
    uint8_t  signal;
    uint8_t  flags;
    uint8_t  _pad;
} BatchProbeResult;

typedef struct {
    Block*    d_snapshot;    /* n_worlds * cells blocks */
    uint32_t* d_ticks;       /* n_worlds per-instance clocks */
    uint8_t*  d_done;        /* n_worlds convergence flags */
    int       n_worlds;
    int       cells;
} WorldBatchSnapshot;

typedef struct {
    /* Dimensions */
    int n_worlds;
    int size_x, size_y, size_z;
    int cells;               /* per instance */
    int total_blocks;        /* n_worlds * cells */

    /* GPU block buffers (double buffered, all instances contiguous) */
    Block* d_blocks_a;
    Block* d_blocks_b;
    Block* d_current;
    Block* d_next;

    /* Host mirror of every instance */
    Block* h_blocks;

    /* Per-instance simulation state */
    uint32_t* d_ticks;       /* game tick per instance */
    uint32_t* h_ticks;
    uint8_t*  d_done;        /* 1 = converged, skipped by kernels */
    uint8_t*  h_done;
    int32_t*  d_changed;     /* scratch: state changed this tick */
    int32_t*  d_sched;       /* scratch: instance holds a pending tile tick */
    int32_t*  d_pass_changed;/* scratch: this relaxation pass changed something */
    uint8_t*  d_relaxed;     /* scratch: wire field settled, skip further passes */
    int32_t*  d_any_active;  /* scratch: any instance still relaxing (1 int) */

    int dirty;               /* host mirror has edits not yet on GPU */

    /* Probes */
    int32_t*          h_probe_positions;  /* 4 ints per probe: inst,x,y,z */
    int32_t*          d_probe_positions;
    BatchProbeResult* d_probe_results;
    BatchProbeResult* h_probe_results;
    int               probe_count;
    int               probe_capacity;

    /* Lever-write staging buffers (device-side writes, no full upload) */
    int32_t* d_lever_positions;
    uint8_t* d_lever_values;
    int      lever_capacity;
} WorldBatch;

/* ── Lifecycle ────────────────────────────────────────────────────────── */
WorldBatch* world_batch_create(int n_worlds, int size_x, int size_y, int size_z);
void        world_batch_destroy(WorldBatch* batch);

/* ── Block access (host mirror; marks batch dirty) ────────────────────── */
void  world_batch_set_block(WorldBatch* batch, int inst, int x, int y, int z,
                            BlockType type, Direction facing, uint16_t state);
Block world_batch_get_block(const WorldBatch* batch, int inst, int x, int y, int z);
void  world_batch_clear(WorldBatch* batch);
void  world_batch_clear_instance(WorldBatch* batch, int inst);

/* Copy `cells` blocks from instance `src_inst` into `dst_inst` (host side).
 * Handy for "one prompt, K sampled circuits" RL groups. */
void  world_batch_copy_instance(WorldBatch* batch, int src_inst, int dst_inst);

/* ── GPU sync ─────────────────────────────────────────────────────────── */
void world_batch_sync_to_gpu(WorldBatch* batch);
void world_batch_sync_from_gpu(WorldBatch* batch);
void world_batch_copy_host_buffer(const WorldBatch* batch, Block* out);

/* ── Simulation ───────────────────────────────────────────────────────── */
void world_batch_tick(WorldBatch* batch, int n);

/* Tick until every instance reaches a fixed point, or `max_ticks` elapse.
 * Polls the device `done` flags every `check_every` ticks (<=0 → 4).
 * Returns the number of ticks actually run. */
int  world_batch_tick_until_stable(WorldBatch* batch, int max_ticks, int check_every);

/* Number of instances that have converged (host-side view; refreshed by
 * tick_until_stable and by world_batch_refresh_done). */
int  world_batch_done_count(WorldBatch* batch);
void world_batch_refresh_done(WorldBatch* batch);
int  world_batch_instance_done(const WorldBatch* batch, int inst);
uint32_t world_batch_instance_tick(const WorldBatch* batch, int inst);

/* ── Device-side lever writes (no host round-trip) ────────────────────── */
/* `inst_xyz` is 4*count ints (inst,x,y,z); `on` is count bytes (0/1).
 * Wakes the touched instances from `done`. */
void world_batch_set_levers(WorldBatch* batch, const int32_t* inst_xyz,
                            const uint8_t* on, int count);

/* ── Probes (read many cells across many instances in one D2H) ────────── */
void world_batch_set_probes(WorldBatch* batch, const int32_t* inst_xyz, int count);
const BatchProbeResult* world_batch_read_probes(WorldBatch* batch);

/* ── Snapshots ────────────────────────────────────────────────────────── */
WorldBatchSnapshot* world_batch_save(WorldBatch* batch);
void world_batch_restore(WorldBatch* batch, const WorldBatchSnapshot* snap);
void world_batch_restore_instance(WorldBatch* batch, const WorldBatchSnapshot* snap,
                                  int inst);
void world_batch_snapshot_destroy(WorldBatchSnapshot* snap);

/* ── Utility ──────────────────────────────────────────────────────────── */
int world_batch_index(const WorldBatch* batch, int inst, int x, int y, int z);
int world_batch_in_bounds(const WorldBatch* batch, int x, int y, int z);

#ifdef __cplusplus
}
#endif

#endif /* REDSTONE_BATCH_H */
