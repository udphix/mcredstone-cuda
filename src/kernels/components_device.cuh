#ifndef REDSTONE_COMPONENTS_DEVICE_CUH
#define REDSTONE_COMPONENTS_DEVICE_CUH

/* ══════════════════════════════════════════════════════════════════════════
 * Torch / repeater state machine — DEVICE-SIDE LOGIC (shared).
 *
 * Included by kernels/components.cu (single-world) and kernels/batch.cu
 * (batched). Both call `component_process_event_cell` /
 * `component_detect_change_cell`; neither reimplements the rules.
 * ══════════════════════════════════════════════════════════════════════════ */

#include "redstone/types.h"
#include <cuda_runtime.h>

/* ── Device helpers ───────────────────────────────────────────────────── */

__device__ static inline int comp_index(int x, int y, int z,
                                         int sx, int sy, int sz) {
    (void)sy;
    return (y * sz + z) * sx + x;
}

__device__ static inline int comp_in_bounds(int x, int y, int z,
                                             int sx, int sy, int sz) {
    return x >= 0 && x < sx && y >= 0 && y < sy && z >= 0 && z < sz;
}

__device__ static inline Block comp_get_block(const Block* blocks,
                                               int x, int y, int z,
                                               int sx, int sy, int sz) {
    Block empty;
    empty.type = 0; empty.signal = 0; empty.facing = 0;
    empty.flags = 0; empty.state = 0; empty.tick_scheduled = 0;
    if (!comp_in_bounds(x, y, z, sx, sy, sz)) return empty;
    return blocks[comp_index(x, y, z, sx, sy, sz)];
}

__device__ static inline int comp_dir_dx(int dir) {
    const int d[] = { 0, 0, 0, 0, -1, 1, 0 };
    return d[dir < 7 ? dir : 6];
}
__device__ static inline int comp_dir_dy(int dir) {
    const int d[] = { -1, 1, 0, 0, 0, 0, 0 };
    return d[dir < 7 ? dir : 6];
}
__device__ static inline int comp_dir_dz(int dir) {
    const int d[] = { 0, 0, -1, 1, 0, 0, 0 };
    return d[dir < 7 ? dir : 6];
}
__device__ static inline int comp_dir_opposite(int dir) {
    const int opp[] = { 1, 0, 3, 2, 5, 4, 6 };
    return opp[dir < 7 ? dir : 6];
}

__device__ static inline int comp_is_solid(uint8_t type) {
    return type == BLOCK_SOLID || type == BLOCK_LAMP;
}

/* ── Get the input signal for a repeater's back face ──────────────────── */
__device__ static inline int comp_get_repeater_input(
    const Block* blocks,
    int x, int y, int z,
    int facing,
    int sx, int sy, int sz
) {
    /* Input comes from the opposite direction of facing (the "back") */
    int back = comp_dir_opposite(facing);
    int bx = x + comp_dir_dx(back);
    int by = y + comp_dir_dy(back);
    int bz = z + comp_dir_dz(back);

    Block input = comp_get_block(blocks, bx, by, bz, sx, sy, sz);

    /* Dust */
    if (input.type == BLOCK_REDSTONE_DUST && input.signal > 0)
        return input.signal;

    /* Another repeater pointing into this one's back */
    if (input.type == BLOCK_REPEATER && input.signal > 0) {
        int f = input.facing;
        if (comp_dir_dx(f) == -comp_dir_dx(back) &&
            comp_dir_dz(f) == -comp_dir_dz(back))
            return 15;
    }

    /* Torch */
    if (input.type == BLOCK_REDSTONE_TORCH && input.signal > 0) {
        /* Torch powers this direction unless it's the attached block */
        int f = input.facing;
        if (!(comp_dir_dx(f) == -comp_dir_dx(back) &&
              comp_dir_dy(f) == -comp_dir_dy(back) &&
              comp_dir_dz(f) == -comp_dir_dz(back)))
            return 15;
    }

    /* Power source (lever, redstone_block) */
    if ((input.type == BLOCK_LEVER || input.type == BLOCK_REDSTONE_BLOCK ||
         input.type == BLOCK_BUTTON) && (input.flags & BLOCK_FLAG_POWERED))
        return 15;

    /* Strongly powered solid block */
    if (comp_is_solid(input.type) && (input.flags & BLOCK_FLAG_STRONG_POWER))
        return input.signal;

    return 0;
}

/* ── Check if a repeater is locked by side repeaters ──────────────────── */
__device__ static inline int comp_repeater_is_locked(
    const Block* blocks,
    int x, int y, int z,
    int facing,
    int sx, int sy, int sz
) {
    /* Side directions depend on facing:
     * Facing NORTH/SOUTH → sides are EAST/WEST
     * Facing EAST/WEST   → sides are NORTH/SOUTH */
    int side1, side2;
    if (facing == DIR_NORTH || facing == DIR_SOUTH) {
        side1 = DIR_EAST;
        side2 = DIR_WEST;
    } else {
        side1 = DIR_NORTH;
        side2 = DIR_SOUTH;
    }

    /* Check each side for a powered repeater pointing into this one */
    int sides[] = { side1, side2 };
    for (int i = 0; i < 2; i++) {
        int sd = sides[i];
        int sx2 = x + comp_dir_dx(sd);
        int sy2 = y + comp_dir_dy(sd);
        int sz2 = z + comp_dir_dz(sd);

        Block side_block = comp_get_block(blocks, sx2, sy2, sz2, sx, sy, sz);

        if (side_block.type == BLOCK_REPEATER && side_block.signal > 0) {
            /* Check if this side repeater is pointing towards us */
            int sf = side_block.facing;
            if (comp_dir_dx(sf) == -comp_dir_dx(sd) &&
                comp_dir_dz(sf) == -comp_dir_dz(sd)) {
                return 1; /* Locked! */
            }
        }
    }

    return 0;
}

/* ── Comparator input levels ──────────────────────────────────────────────
 * MC reads a comparator's rear and its two sides differently
 * (ComparatorBlock.getInputSignal vs DiodeBlock.getAlternateSignal):
 *
 *   rear  — the general signal getter, so a strongly-powered conductor counts.
 *   sides — only "alternate inputs", i.e. real signal sources. A plain solid
 *           block carrying strong power does NOT feed a comparator's side.
 *
 * `into` is the direction from the neighbour back toward the comparator, so a
 * repeater/comparator feeds us only when its facing equals `into`.
 *
 * NOT MODELLED: container fullness (no containers exist in this simulator), so
 * a comparator behind a chest reads 0 rather than its fill level.
 * ═══════════════════════════════════════════════════════════════════════ */
__device__ static inline int comp_comparator_level(
    const Block* nb, int into, int sides_only
) {
    switch (nb->type) {
    case BLOCK_REDSTONE_DUST:  return (int)nb->signal;
    case BLOCK_REDSTONE_BLOCK: return 15;
    case BLOCK_LEVER:
    case BLOCK_BUTTON:         return (nb->flags & BLOCK_FLAG_POWERED) ? 15 : 0;
    case BLOCK_REDSTONE_TORCH: return nb->signal > 0 ? 15 : 0;
    case BLOCK_REPEATER:
    case BLOCK_COMPARATOR:     return ((int)nb->facing == into) ? (int)nb->signal : 0;
    case BLOCK_SOLID:
    case BLOCK_LAMP:
        if (sides_only) return 0;
        return (nb->flags & BLOCK_FLAG_STRONG_POWER) ? (int)nb->signal : 0;
    default:                   return 0;
    }
}

/* Output a comparator should be driving right now.
 *   compare  mode: rear if rear >= max(sides), else 0
 *   subtract mode: max(0, rear - max(sides)) */
__device__ static inline int comp_get_comparator_output(
    const Block* blocks,
    int x, int y, int z,
    int facing, int mode_subtract,
    int sx, int sy, int sz
) {
    int back = comp_dir_opposite(facing);
    Block rear = comp_get_block(blocks,
                                x + comp_dir_dx(back),
                                y + comp_dir_dy(back),
                                z + comp_dir_dz(back), sx, sy, sz);
    /* Direction from the rear neighbour back to us is `facing`. */
    int rear_level = comp_comparator_level(&rear, facing, 0);

    /* Comparators are horizontal in MC; a vertical facing has no well-defined
     * sides, so treat it as unpowered rather than guessing. */
    if (facing == DIR_UP || facing == DIR_DOWN || facing == DIR_NONE) return 0;

    int sides[2];
    if (facing == DIR_NORTH || facing == DIR_SOUTH) {
        sides[0] = DIR_EAST;  sides[1] = DIR_WEST;
    } else {
        sides[0] = DIR_NORTH; sides[1] = DIR_SOUTH;
    }

    int side_level = 0;
    for (int i = 0; i < 2; i++) {
        int sd = sides[i];
        Block nb = comp_get_block(blocks,
                                  x + comp_dir_dx(sd),
                                  y + comp_dir_dy(sd),
                                  z + comp_dir_dz(sd), sx, sy, sz);
        int lv = comp_comparator_level(&nb, comp_dir_opposite(sd), 1);
        if (lv > side_level) side_level = lv;
    }

    if (mode_subtract) {
        int out = rear_level - side_level;
        return out > 0 ? out : 0;
    }
    return (rear_level >= side_level) ? rear_level : 0;
}

/* ═══════════════════════════════════════════════════════════════════════
 * PHASE 1 (per cell): Process Tick Events
 *
 * Runs BEFORE signal propagation, in-place on the current buffer.
 * Fires scheduled events whose tick has arrived:
 *   - Torch: toggle ON/OFF
 *   - Repeater: set output to pending value stored in state
 * ═══════════════════════════════════════════════════════════════════════ */
__device__ static inline void component_process_event_cell(
    Block* blocks,
    int idx,
    uint32_t current_tick,
    int size_x, int size_y, int size_z
) {
    (void)size_x; (void)size_y; (void)size_z;

    Block block = blocks[idx];

    /* Only process blocks that have a scheduled event NOW */
    if (!(block.flags & BLOCK_FLAG_SCHEDULED)) return;
    if (block.tick_scheduled != (uint16_t)(current_tick & 0xFFFF)) return;

    /* ── Torch: toggle ────────────────────────────────────────────── */
    if (block.type == BLOCK_REDSTONE_TORCH) {
        if (block.signal > 0) {
            blocks[idx].signal = 0;
            blocks[idx].flags = 0;
        } else {
            blocks[idx].signal = 15;
            blocks[idx].flags = BLOCK_FLAG_POWERED;
        }
        blocks[idx].flags &= ~BLOCK_FLAG_SCHEDULED;
        blocks[idx].tick_scheduled = 0;
    }

    /* ── Repeater: apply pending output ───────────────────────────── */
    if (block.type == BLOCK_REPEATER) {
        /* Pending output is stored in state bits 4-7 */
        uint8_t pending = (block.state >> 4) & 0xF;
        blocks[idx].signal = pending;
        if (pending > 0) {
            blocks[idx].flags = BLOCK_FLAG_POWERED | BLOCK_FLAG_STRONG_POWER;
        } else {
            blocks[idx].flags = 0;
        }
        blocks[idx].flags &= ~BLOCK_FLAG_SCHEDULED;
        blocks[idx].tick_scheduled = 0;
    }

    /* ── Comparator: apply pending output ─────────────────────────── */
    if (block.type == BLOCK_COMPARATOR) {
        /* Same layout as the repeater: pending output in state bits 4-7.
         * Unlike the repeater this is a real level, not just 0/15. */
        uint8_t pending = (block.state >> 4) & 0xF;
        blocks[idx].signal = pending;
        if (pending > 0) {
            blocks[idx].flags = BLOCK_FLAG_POWERED | BLOCK_FLAG_STRONG_POWER;
        } else {
            blocks[idx].flags = 0;
        }
        blocks[idx].flags &= ~BLOCK_FLAG_SCHEDULED;
        blocks[idx].tick_scheduled = 0;
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * PHASE 4 (per cell): Detect Component Changes
 *
 * Runs AFTER signal propagation, in-place on the next buffer.
 * Detects input changes and schedules future events.
 * ═══════════════════════════════════════════════════════════════════════ */
__device__ static inline void component_detect_change_cell(
    Block* blocks,
    int idx,
    uint32_t current_tick,
    int size_x, int size_y, int size_z
) {
    Block block = blocks[idx];

    int x = idx % size_x;
    int z = (idx / size_x) % size_z;
    int y = idx / (size_x * size_z);

    /* ── Torch: check if attached block's power state changed ─────── */
    if (block.type == BLOCK_REDSTONE_TORCH) {
        /* Already has a pending event? Don't schedule another */
        if (block.flags & BLOCK_FLAG_SCHEDULED) return;

        /* Get the block the torch is attached to */
        int f = block.facing;
        int ax = x + comp_dir_dx(f);
        int ay = y + comp_dir_dy(f);
        int az = z + comp_dir_dz(f);

        Block attached = comp_get_block(blocks, ax, ay, az,
                                         size_x, size_y, size_z);
        int attached_powered = (attached.flags & BLOCK_FLAG_POWERED) != 0;
        int torch_on = block.signal > 0;

        /* If attached is powered and torch is ON → schedule OFF */
        /* If attached is NOT powered and torch is OFF → schedule ON */
        if ((attached_powered && torch_on) || (!attached_powered && !torch_on)) {
            uint16_t fire_tick = (uint16_t)((current_tick + 2) & 0xFFFF);
            blocks[idx].tick_scheduled = fire_tick;
            blocks[idx].flags |= BLOCK_FLAG_SCHEDULED;
        }
    }

    /* ── Repeater: check if input changed ─────────────────────────── */
    if (block.type == BLOCK_REPEATER) {
        /* Already scheduled? Don't override */
        if (block.flags & BLOCK_FLAG_SCHEDULED) return;

        int facing = block.facing;

        /* Check locking first */
        int locked = comp_repeater_is_locked(blocks, x, y, z, facing,
                                              size_x, size_y, size_z);
        if (locked) {
            blocks[idx].flags |= BLOCK_FLAG_LOCKED;
            return; /* Locked repeaters don't respond to input */
        }
        blocks[idx].flags &= ~BLOCK_FLAG_LOCKED;

        /* Get input signal from back */
        int input_signal = comp_get_repeater_input(blocks, x, y, z, facing,
                                                    size_x, size_y, size_z);
        int has_input = input_signal > 0;
        int is_outputting = block.signal > 0;

        /* Input changed → schedule output change */
        if (has_input != is_outputting) {
            int delay = block.state & 0xF;
            if (delay < 1) delay = 1;
            if (delay > 4) delay = 4;

            uint16_t fire_tick = (uint16_t)((current_tick + delay * 2) & 0xFFFF);

            /* Store pending output in state bits 4-7 */
            blocks[idx].state = (block.state & 0xF) |
                                ((has_input ? 15 : 0) << 4);
            blocks[idx].tick_scheduled = fire_tick;
            blocks[idx].flags |= BLOCK_FLAG_SCHEDULED;
        }
    }

    /* ── Comparator: check if the computed output changed ──────────── */
    if (block.type == BLOCK_COMPARATOR) {
        if (block.flags & BLOCK_FLAG_SCHEDULED) return;

        /* state bit 0 = mode (0 compare, 1 subtract). Comparators are never
         * locked — that is a repeater-only mechanic. */
        int mode_subtract = block.state & 0x1;
        int want = comp_get_comparator_output(blocks, x, y, z, block.facing,
                                              mode_subtract,
                                              size_x, size_y, size_z);

        if (want != (int)block.signal) {
            /* Comparators are fixed at 1 redstone tick = 2 game ticks. */
            uint16_t fire_tick = (uint16_t)((current_tick + 2) & 0xFFFF);
            blocks[idx].state = (uint16_t)((block.state & 0x000F) |
                                           ((want & 0xF) << 4));
            blocks[idx].tick_scheduled = fire_tick;
            blocks[idx].flags |= BLOCK_FLAG_SCHEDULED;
        }
    }
}

#endif /* REDSTONE_COMPONENTS_DEVICE_CUH */
