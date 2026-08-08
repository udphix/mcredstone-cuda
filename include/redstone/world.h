#ifndef REDSTONE_WORLD_H
#define REDSTONE_WORLD_H

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ── World ────────────────────────────────────────────────────────────────
 * The World holds the full 3D grid of blocks on GPU memory.
 * Uses double buffering: read from current, write to next, then swap.
 */

#define MAX_PROBES 256

/* Probe result: lightweight read of a specific block position */
typedef struct {
    int32_t  x, y, z;
    uint8_t  type;
    uint8_t  signal;
    uint8_t  flags;
    uint8_t  _pad;
} ProbeResult;

/* Snapshot: saved GPU state for restore */
typedef struct {
    Block*   d_snapshot;    /* GPU buffer copy */
    uint32_t tick;
    int      total_blocks;
} WorldSnapshot;

typedef struct {
    /* Dimensions */
    int size_x;
    int size_y;
    int size_z;
    int total_blocks;

    /* GPU buffers (double buffered) */
    Block* d_blocks_a;
    Block* d_blocks_b;
    Block* d_current;
    Block* d_next;

    /* CPU mirror for host-side operations */
    Block* h_blocks;

    /* Simulation state */
    uint32_t current_tick;
    int      buffer_index;
    int      dirty;

    /* Probes: lightweight GPU reads at specific positions */
    int32_t  probe_positions[MAX_PROBES * 3]; /* x,y,z triples */
    int      probe_count;
    int32_t* d_probe_positions;   /* GPU copy of positions */
    ProbeResult* d_probe_results; /* GPU results buffer */
    ProbeResult* h_probe_results; /* host results buffer */
} World;

/* ── Lifecycle ────────────────────────────────────────────────────────── */
World*  world_create(int size_x, int size_y, int size_z);
void    world_destroy(World* world);

/* ── Block access ─────────────────────────────────────────────────────── */
void    world_set_block(World* world, int x, int y, int z,
                        BlockType type, Direction facing, uint16_t state);
Block   world_get_block(const World* world, int x, int y, int z);

/* Toggle a lever's POWERED state without moving/reinstalling it. Use for
 * RL environments that need to cycle input states across truth-table rows
 * without destroying the circuit being evaluated. No-op if the block at
 * (x,y,z) is not a lever. */
void    world_set_lever_powered(World* world, int x, int y, int z, int powered);

/* ── Simulation ───────────────────────────────────────────────────────── */
void    world_tick(World* world);
void    world_tick_n(World* world, int n);

/* ── Buffer management ────────────────────────────────────────────────── */
void    world_sync_to_gpu(World* world);
void    world_sync_from_gpu(World* world);
void    world_swap_buffers(World* world);

/* ── Snapshots (save/restore GPU state) ───────────────────────────────── */
WorldSnapshot* world_save_state(const World* world);
void           world_restore_state(World* world, const WorldSnapshot* snap);
void           world_snapshot_destroy(WorldSnapshot* snap);

/* ── Probes (read specific block signals without full sync) ───────────── */
void           world_add_probe(World* world, int x, int y, int z);
void           world_clear_probes(World* world);
const ProbeResult* world_read_probes(World* world);

/* ── Schematic I/O ────────────────────────────────────────────────────── */
int     world_export_schematic(const World* world, const char* path); /* .rsschem */
int     world_import_schematic(World* world, const char* path);       /* .rsschem */
int     world_import_schem(World* world, const char* path);           /* .schem (MC) */
int     world_export_schem(const World* world, const char* path);    /* .schem (MC) */

/* ── Utility ──────────────────────────────────────────────────────────── */
int     world_index(const World* world, int x, int y, int z);
int     world_in_bounds(const World* world, int x, int y, int z);
void    world_clear(World* world);
void    world_print_slice(const World* world, int y);

/* Bulk-read the host mirror of the block grid into `out`.
 * `out` must point to at least `total_blocks * sizeof(Block)` bytes.
 * Does NOT call sync_from_gpu — caller must sync first if needed.
 * Used by Python bindings for fast observation building. */
void    world_copy_host_buffer(const World* world, Block* out);

#ifdef __cplusplus
}
#endif

#endif /* REDSTONE_WORLD_H */
