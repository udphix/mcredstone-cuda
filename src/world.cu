#include "redstone/world.h"
#include "redstone/kernels.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── Helpers ──────────────────────────────────────────────────────────── */

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

int world_index(const World* world, int x, int y, int z) {
    return (y * world->size_z + z) * world->size_x + x;
}

int world_in_bounds(const World* world, int x, int y, int z) {
    return x >= 0 && x < world->size_x &&
           y >= 0 && y < world->size_y &&
           z >= 0 && z < world->size_z;
}

/* ── Lifecycle ────────────────────────────────────────────────────────── */

World* world_create(int size_x, int size_y, int size_z) {
    World* world = (World*)calloc(1, sizeof(World));
    if (!world) {
        fprintf(stderr, "Failed to allocate World\n");
        return NULL;
    }

    world->size_x = size_x;
    world->size_y = size_y;
    world->size_z = size_z;
    world->total_blocks = size_x * size_y * size_z;
    world->current_tick = 0;
    world->buffer_index = 0;
    world->dirty = 0;

    size_t buffer_size = (size_t)world->total_blocks * sizeof(Block);

    /* Allocate CPU buffer */
    world->h_blocks = (Block*)calloc(world->total_blocks, sizeof(Block));
    if (!world->h_blocks) {
        fprintf(stderr, "Failed to allocate host buffer (%zu bytes)\n", buffer_size);
        free(world);
        return NULL;
    }

    /* Allocate GPU double buffers */
    CUDA_CHECK(cudaMalloc(&world->d_blocks_a, buffer_size));
    CUDA_CHECK(cudaMalloc(&world->d_blocks_b, buffer_size));

    /* Initialize GPU memory to zero (all AIR blocks) */
    CUDA_CHECK(cudaMemset(world->d_blocks_a, 0, buffer_size));
    CUDA_CHECK(cudaMemset(world->d_blocks_b, 0, buffer_size));

    /* Set buffer pointers */
    world->d_current = world->d_blocks_a;
    world->d_next    = world->d_blocks_b;

    /* Probes */
    world->probe_count = 0;
    CUDA_CHECK(cudaMalloc(&world->d_probe_positions, MAX_PROBES * 3 * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&world->d_probe_results, MAX_PROBES * sizeof(ProbeResult)));
    world->h_probe_results = (ProbeResult*)calloc(MAX_PROBES, sizeof(ProbeResult));

    /* Uncomment for debug:
    printf("[World] Created %dx%dx%d world (%.1f MB GPU x2)\n",
           size_x, size_y, size_z, buffer_size / (1024.0 * 1024.0)); */

    return world;
}

void world_destroy(World* world) {
    if (!world) return;

    if (world->d_blocks_a)        cudaFree(world->d_blocks_a);
    if (world->d_blocks_b)        cudaFree(world->d_blocks_b);
    if (world->d_probe_positions) cudaFree(world->d_probe_positions);
    if (world->d_probe_results)   cudaFree(world->d_probe_results);
    if (world->h_blocks)          free(world->h_blocks);
    if (world->h_probe_results)   free(world->h_probe_results);

    free(world);
}

/* ── Block access (host side) ─────────────────────────────────────────── */

void world_set_block(World* world, int x, int y, int z,
                     BlockType type, Direction facing, uint16_t state) {
    if (!world_in_bounds(world, x, y, z)) {
        fprintf(stderr, "[World] set_block out of bounds: (%d,%d,%d)\n", x, y, z);
        return;
    }

    int idx = world_index(world, x, y, z);
    Block* block = &world->h_blocks[idx];

    block->type   = (uint8_t)type;
    block->facing  = (uint8_t)facing;
    block->state   = state;
    block->signal  = 0;
    block->flags   = 0;
    block->tick_scheduled = 0;

    /* Power sources start with signal 15 */
    if (type == BLOCK_LEVER || type == BLOCK_REDSTONE_BLOCK) {
        block->signal = 15;
        block->flags |= BLOCK_FLAG_POWERED;
    }

    /* Torch starts ON */
    if (type == BLOCK_REDSTONE_TORCH) {
        block->signal = 15;
        block->flags |= BLOCK_FLAG_POWERED;
    }

    /* Repeater: delay stored in lower 4 bits of state (1-4) */
    if (type == BLOCK_REPEATER) {
        uint16_t delay = state & 0xF;
        if (delay < 1) delay = 1;
        if (delay > 4) delay = 4;
        block->state = delay;
        block->signal = 0; /* starts OFF */
    }

    world->dirty = 1;
}

Block world_get_block(const World* world, int x, int y, int z) {
    Block empty = {0};
    if (!world_in_bounds(world, x, y, z)) return empty;
    return world->h_blocks[world_index(world, x, y, z)];
}

void world_set_lever_powered(World* world, int x, int y, int z, int powered) {
    if (!world_in_bounds(world, x, y, z)) return;
    int idx = world_index(world, x, y, z);
    Block* b = &world->h_blocks[idx];
    if (b->type != BLOCK_LEVER) return;
    if (powered) {
        b->signal = 15;
        b->flags |= BLOCK_FLAG_POWERED;
    } else {
        b->signal = 0;
        b->flags &= ~BLOCK_FLAG_POWERED;
    }
    world->dirty = 1;
}

/* ── GPU sync ─────────────────────────────────────────────────────────── */

void world_sync_to_gpu(World* world) {
    size_t size = (size_t)world->total_blocks * sizeof(Block);
    CUDA_CHECK(cudaMemcpy(world->d_current, world->h_blocks, size,
                          cudaMemcpyHostToDevice));
    /* Also copy to the 'next' buffer so both start in sync */
    CUDA_CHECK(cudaMemcpy(world->d_next, world->h_blocks, size,
                          cudaMemcpyHostToDevice));
    world->dirty = 0;
}

void world_sync_from_gpu(World* world) {
    size_t size = (size_t)world->total_blocks * sizeof(Block);
    CUDA_CHECK(cudaMemcpy(world->h_blocks, world->d_current, size,
                          cudaMemcpyDeviceToHost));
}

void world_swap_buffers(World* world) {
    Block* tmp = world->d_current;
    world->d_current = world->d_next;
    world->d_next = tmp;
    world->buffer_index ^= 1;
}

/* ── Simulation ───────────────────────────────────────────────────────── */

void world_tick(World* world) {
    /* Sync host → GPU if dirty */
    if (world->dirty) {
        world_sync_to_gpu(world);
    }

    /* ══════════════════════════════════════════════════════════════════
     * Minecraft tick processing order (each game tick = 50ms):
     *
     *   1. TILE TICKS      — scheduled events fire (torch toggle,
     *                         repeater output, comparator update)
     *   2. BLOCK UPDATES   — signal propagation, neighbor updates
     *                         (dust recalculates, solid blocks check
     *                         power, lamps toggle, quasi-connectivity)
     *   3. BLOCK EVENTS    — piston extend/retract, block movement
     *   4. SCHEDULING      — detect input changes on components,
     *                         schedule future tile ticks
     *
     * This order is critical for Minecraft-faithful timing. Changing
     * it will break circuits that depend on specific tick behavior.
     * ══════════════════════════════════════════════════════════════════ */

    /* 1. TILE TICKS — fire scheduled events from previous ticks.
     * Torch toggles, repeater output changes happen HERE, before
     * signal propagation sees them. Modifies d_current in-place. */
    kernel_process_tick_events(
        world->d_current,
        world->current_tick,
        world->size_x,
        world->size_y,
        world->size_z
    );

    /* 2. BLOCK UPDATES — propagate redstone signal through the world.
     * Reads d_current, writes d_next. Double-buffered to prevent
     * GPU race conditions. Includes quasi-connectivity checks for
     * pistons/dispensers/droppers. */
    kernel_propagate_signal(
        world->d_current,
        world->d_next,
        world->size_x,
        world->size_y,
        world->size_z
    );

    /* 3. BLOCK EVENTS — piston extensions/retractions, block movement.
     * TODO v0.5: kernel_move_blocks() */

    /* 4. SCHEDULING — detect component input changes after propagation,
     * schedule future tile ticks. Modifies d_next in-place. */
    kernel_detect_component_changes(
        world->d_next,
        world->current_tick,
        world->size_x,
        world->size_y,
        world->size_z
    );

    /* Swap buffers: d_next becomes d_current for next tick */
    world_swap_buffers(world);

    world->current_tick++;
}

void world_tick_n(World* world, int n) {
    for (int i = 0; i < n; i++) {
        world_tick(world);
    }
}

/* ── Utility ──────────────────────────────────────────────────────────── */

void world_clear(World* world) {
    memset(world->h_blocks, 0, (size_t)world->total_blocks * sizeof(Block));
    world->current_tick = 0;
    world->dirty = 1;
}

void world_copy_host_buffer(const World* world, Block* out) {
    if (!world || !out) return;
    memcpy(out, world->h_blocks, (size_t)world->total_blocks * sizeof(Block));
}

static const char* block_type_short(uint8_t type) {
    switch (type) {
        case BLOCK_AIR:            return "  .  ";
        case BLOCK_SOLID:          return " [#] ";
        case BLOCK_LEVER:          return " [L] ";
        case BLOCK_REDSTONE_DUST:  return " ";
        case BLOCK_REDSTONE_TORCH: return " [T] ";
        case BLOCK_REPEATER:       return " [R] ";
        case BLOCK_REDSTONE_BLOCK: return " [B] ";
        case BLOCK_COMPARATOR:     return " [C] ";
        case BLOCK_LAMP:           return " [*] ";
        default:                   return " [?] ";
    }
}

void world_print_slice(const World* world, int y) {
    if (y < 0 || y >= world->size_y) return;

    /* First, find the bounding box of non-air blocks in this slice */
    int min_x = world->size_x, max_x = -1;
    int min_z = world->size_z, max_z = -1;

    for (int z = 0; z < world->size_z; z++) {
        for (int x = 0; x < world->size_x; x++) {
            Block b = world->h_blocks[world_index(world, x, y, z)];
            if (b.type != BLOCK_AIR) {
                if (x < min_x) min_x = x;
                if (x > max_x) max_x = x;
                if (z < min_z) min_z = z;
                if (z > max_z) max_z = z;
            }
        }
    }

    if (max_x < 0) {
        printf("[World] Slice y=%d: empty\n", y);
        return;
    }

    /* Add 1 block padding */
    if (min_x > 0) min_x--;
    if (min_z > 0) min_z--;
    if (max_x < world->size_x - 1) max_x++;
    if (max_z < world->size_z - 1) max_z++;

    printf("[World] Slice y=%d (tick %u):\n", y, world->current_tick);
    printf("    ");
    for (int x = min_x; x <= max_x; x++) {
        printf(" x=%-2d", x);
    }
    printf("\n");

    for (int z = min_z; z <= max_z; z++) {
        printf("z=%-2d", z);
        for (int x = min_x; x <= max_x; x++) {
            Block b = world->h_blocks[world_index(world, x, y, z)];
            if (b.type == BLOCK_REDSTONE_DUST) {
                printf(" d%-2d ", b.signal);
            } else {
                printf("%s", block_type_short(b.type));
            }
        }
        printf("\n");
    }
    printf("\n");
}

/* ══════════════════════════════════════════════════════════════════════════
 * Schematic I/O — simple text format (.rsschem)
 *
 * Format:
 *   # comment lines start with #
 *   RSSCHEM 1                     (version)
 *   SIZE 8 4 4                    (world dimensions)
 *   BLOCK x y z type facing state (one per non-air block)
 *   END
 *
 * Block types use the enum names: AIR, SOLID, LEVER, REDSTONE_DUST, etc.
 * ══════════════════════════════════════════════════════════════════════════ */

static const char* blocktype_to_name(uint8_t type) {
    switch (type) {
        case BLOCK_AIR:            return "AIR";
        case BLOCK_SOLID:          return "SOLID";
        case BLOCK_LEVER:          return "LEVER";
        case BLOCK_BUTTON:         return "BUTTON";
        case BLOCK_PRESSURE_PLATE: return "PRESSURE_PLATE";
        case BLOCK_REDSTONE_BLOCK: return "REDSTONE_BLOCK";
        case BLOCK_REDSTONE_DUST:  return "REDSTONE_DUST";
        case BLOCK_REDSTONE_TORCH: return "REDSTONE_TORCH";
        case BLOCK_REPEATER:       return "REPEATER";
        case BLOCK_COMPARATOR:     return "COMPARATOR";
        case BLOCK_PISTON:         return "PISTON";
        case BLOCK_STICKY_PISTON:  return "STICKY_PISTON";
        case BLOCK_LAMP:           return "LAMP";
        case BLOCK_OBSERVER:       return "OBSERVER";
        default:                   return "UNKNOWN";
    }
}

static uint8_t name_to_blocktype(const char* name) {
    if (!name) return BLOCK_AIR;
    if (strcmp(name, "AIR") == 0)            return BLOCK_AIR;
    if (strcmp(name, "SOLID") == 0)          return BLOCK_SOLID;
    if (strcmp(name, "LEVER") == 0)          return BLOCK_LEVER;
    if (strcmp(name, "BUTTON") == 0)         return BLOCK_BUTTON;
    if (strcmp(name, "PRESSURE_PLATE") == 0) return BLOCK_PRESSURE_PLATE;
    if (strcmp(name, "REDSTONE_BLOCK") == 0) return BLOCK_REDSTONE_BLOCK;
    if (strcmp(name, "REDSTONE_DUST") == 0)  return BLOCK_REDSTONE_DUST;
    if (strcmp(name, "REDSTONE_TORCH") == 0) return BLOCK_REDSTONE_TORCH;
    if (strcmp(name, "REPEATER") == 0)       return BLOCK_REPEATER;
    if (strcmp(name, "COMPARATOR") == 0)     return BLOCK_COMPARATOR;
    if (strcmp(name, "PISTON") == 0)         return BLOCK_PISTON;
    if (strcmp(name, "STICKY_PISTON") == 0)  return BLOCK_STICKY_PISTON;
    if (strcmp(name, "LAMP") == 0)           return BLOCK_LAMP;
    if (strcmp(name, "OBSERVER") == 0)       return BLOCK_OBSERVER;
    return BLOCK_AIR;
}

static const char* dir_to_name(uint8_t dir) {
    switch (dir) {
        case DIR_DOWN:  return "DOWN";
        case DIR_UP:    return "UP";
        case DIR_NORTH: return "NORTH";
        case DIR_SOUTH: return "SOUTH";
        case DIR_WEST:  return "WEST";
        case DIR_EAST:  return "EAST";
        default:        return "NONE";
    }
}

static uint8_t name_to_dir(const char* name) {
    if (!name) return DIR_NONE;
    if (strcmp(name, "DOWN") == 0)  return DIR_DOWN;
    if (strcmp(name, "UP") == 0)    return DIR_UP;
    if (strcmp(name, "NORTH") == 0) return DIR_NORTH;
    if (strcmp(name, "SOUTH") == 0) return DIR_SOUTH;
    if (strcmp(name, "WEST") == 0)  return DIR_WEST;
    if (strcmp(name, "EAST") == 0)  return DIR_EAST;
    return DIR_NONE;
}

int world_export_schematic(const World* world, const char* path) {
    FILE* f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "[Schematic] Cannot open %s for writing\n", path);
        return -1;
    }

    fprintf(f, "# Redstone Simulator GPU — Schematic\n");
    fprintf(f, "RSSCHEM 1\n");
    fprintf(f, "SIZE %d %d %d\n", world->size_x, world->size_y, world->size_z);

    int count = 0;
    for (int y = 0; y < world->size_y; y++) {
        for (int z = 0; z < world->size_z; z++) {
            for (int x = 0; x < world->size_x; x++) {
                Block b = world->h_blocks[world_index(world, x, y, z)];
                if (b.type == BLOCK_AIR) continue;
                fprintf(f, "BLOCK %d %d %d %s %s %d\n",
                        x, y, z,
                        blocktype_to_name(b.type),
                        dir_to_name(b.facing),
                        b.state);
                count++;
            }
        }
    }

    fprintf(f, "END\n");
    fclose(f);
    printf("[Schematic] Exported %d blocks to %s\n", count, path);
    return count;
}

int world_import_schematic(World* world, const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "[Schematic] Cannot open %s for reading\n", path);
        return -1;
    }

    char line[256];
    int count = 0;
    int got_header = 0;

    while (fgets(line, sizeof(line), f)) {
        /* Skip comments and empty lines */
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;

        if (strncmp(line, "RSSCHEM", 7) == 0) {
            got_header = 1;
            continue;
        }

        if (strncmp(line, "SIZE", 4) == 0) {
            /* We ignore SIZE since the world is already created.
             * Caller should create a world of appropriate size. */
            continue;
        }

        if (strncmp(line, "END", 3) == 0) break;

        if (strncmp(line, "BLOCK", 5) == 0) {
            int x, y, z, state;
            char type_name[32], dir_name[16];
            if (sscanf(line, "BLOCK %d %d %d %31s %15s %d",
                        &x, &y, &z, type_name, dir_name, &state) == 6) {
                BlockType type = (BlockType)name_to_blocktype(type_name);
                Direction dir = (Direction)name_to_dir(dir_name);
                world_set_block(world, x, y, z, type, dir, (uint16_t)state);
                count++;
            }
        }
    }

    fclose(f);

    if (!got_header) {
        fprintf(stderr, "[Schematic] Warning: no RSSCHEM header in %s\n", path);
    }

    printf("[Schematic] Imported %d blocks from %s\n", count, path);
    return count;
}

/* ══════════════════════════════════════════════════════════════════════════
 * Snapshots — save/restore full GPU state
 * ══════════════════════════════════════════════════════════════════════════ */

WorldSnapshot* world_save_state(const World* world) {
    WorldSnapshot* snap = (WorldSnapshot*)calloc(1, sizeof(WorldSnapshot));
    if (!snap) return NULL;

    size_t size = (size_t)world->total_blocks * sizeof(Block);
    snap->total_blocks = world->total_blocks;
    snap->tick = world->current_tick;

    /* If the host mirror has pending edits (dirty=1), push them to the
     * GPU before snapshotting. Otherwise the snapshot would capture
     * stale d_current without the host's recent set_block edits, and
     * restore() would silently wipe them. This is the semantics the
     * Python bindings and RL rollout code assume. */
    if (world->dirty) {
        World* w = (World*)world;  /* cast away const for sync */
        CUDA_CHECK(cudaMemcpy(w->d_current, w->h_blocks, size,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(w->d_next, w->h_blocks, size,
                              cudaMemcpyHostToDevice));
        w->dirty = 0;
    }

    CUDA_CHECK(cudaMalloc(&snap->d_snapshot, size));
    CUDA_CHECK(cudaMemcpy(snap->d_snapshot, world->d_current, size,
                          cudaMemcpyDeviceToDevice));
    return snap;
}

void world_restore_state(World* world, const WorldSnapshot* snap) {
    if (!snap || snap->total_blocks != world->total_blocks) {
        fprintf(stderr, "[World] restore_state: snapshot size mismatch\n");
        return;
    }

    size_t size = (size_t)world->total_blocks * sizeof(Block);

    /* Restore GPU buffers (both, so double buffering starts clean) */
    CUDA_CHECK(cudaMemcpy(world->d_current, snap->d_snapshot, size,
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(world->d_next, snap->d_snapshot, size,
                          cudaMemcpyDeviceToDevice));

    /* Also update host mirror */
    CUDA_CHECK(cudaMemcpy(world->h_blocks, snap->d_snapshot, size,
                          cudaMemcpyDeviceToHost));

    world->current_tick = snap->tick;
    world->dirty = 0;
}

void world_snapshot_destroy(WorldSnapshot* snap) {
    if (!snap) return;
    if (snap->d_snapshot) cudaFree(snap->d_snapshot);
    free(snap);
}

/* ══════════════════════════════════════════════════════════════════════════
 * Probes — read specific block signals without full GPU sync
 * ══════════════════════════════════════════════════════════════════════════ */

void world_add_probe(World* world, int x, int y, int z) {
    if (world->probe_count >= MAX_PROBES) {
        fprintf(stderr, "[World] Max probes (%d) reached\n", MAX_PROBES);
        return;
    }
    int i = world->probe_count;
    world->probe_positions[i * 3 + 0] = x;
    world->probe_positions[i * 3 + 1] = y;
    world->probe_positions[i * 3 + 2] = z;
    world->probe_count++;
}

void world_clear_probes(World* world) {
    world->probe_count = 0;
}

/* GPU kernel: read N probe positions from the block grid */
__global__ void kernel_read_probes(
    const Block* __restrict__ blocks,
    const int32_t* __restrict__ positions,
    ProbeResult* __restrict__ results,
    int probe_count,
    int size_x, int size_y, int size_z
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= probe_count) return;

    int x = positions[i * 3 + 0];
    int y = positions[i * 3 + 1];
    int z = positions[i * 3 + 2];

    results[i].x = x;
    results[i].y = y;
    results[i].z = z;

    if (x >= 0 && x < size_x && y >= 0 && y < size_y && z >= 0 && z < size_z) {
        int idx = (y * size_z + z) * size_x + x;
        Block b = blocks[idx];
        results[i].type   = b.type;
        results[i].signal = b.signal;
        results[i].flags  = b.flags;
    } else {
        results[i].type   = 0;
        results[i].signal = 0;
        results[i].flags  = 0;
    }
}

const ProbeResult* world_read_probes(World* world) {
    if (world->probe_count == 0) return world->h_probe_results;

    /* Sync dirty host data first */
    if (world->dirty) {
        world_sync_to_gpu(world);
    }

    /* Upload probe positions to GPU */
    CUDA_CHECK(cudaMemcpy(world->d_probe_positions,
                          world->probe_positions,
                          world->probe_count * 3 * sizeof(int32_t),
                          cudaMemcpyHostToDevice));

    /* Launch tiny kernel */
    int threads = 64;
    int grid = (world->probe_count + threads - 1) / threads;
    kernel_read_probes<<<grid, threads>>>(
        world->d_current,
        world->d_probe_positions,
        world->d_probe_results,
        world->probe_count,
        world->size_x, world->size_y, world->size_z
    );
    cudaDeviceSynchronize();

    /* Copy results back (only probe_count results, not the whole world) */
    CUDA_CHECK(cudaMemcpy(world->h_probe_results,
                          world->d_probe_results,
                          world->probe_count * sizeof(ProbeResult),
                          cudaMemcpyDeviceToHost));

    return world->h_probe_results;
}
