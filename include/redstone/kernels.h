#ifndef REDSTONE_KERNELS_H
#define REDSTONE_KERNELS_H

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ── Signal Propagation Kernel ────────────────────────────────────────────
 * Reads from src buffer, writes signal levels to dst buffer.
 * Direction-aware: torch doesn't power attached block, repeater only
 * powers block in front.
 */
/* `d_changed` (may not be NULL) receives 1 if this pass altered any cell.
 * The caller uses it to stop the relaxation loop as soon as the wire field
 * has settled, instead of always running DUST_RELAX_PASSES. */
void kernel_propagate_signal(
    const Block* d_src,
    Block* d_dst,
    int32_t* d_changed,
    int size_x, int size_y, int size_z
);

/* ── Process Tick Events ──────────────────────────────────────────────────
 * Runs BEFORE signal propagation. Modifies blocks in-place.
 * Fires scheduled events for the current tick:
 *   - Torch: toggle ON/OFF
 *   - Repeater: set output to pending value
 */
void kernel_process_tick_events(
    Block* d_blocks,
    uint32_t current_tick,
    int size_x, int size_y, int size_z
);

/* ── Detect Component Changes ─────────────────────────────────────────────
 * Runs AFTER signal propagation. Modifies blocks in-place.
 * Detects input changes and schedules future events:
 *   - Torch: attached block power changed -> schedule toggle (2 tick delay)
 *   - Repeater: back input changed -> schedule output (delay * 2 ticks)
 *   - Repeater locking: side repeater powered -> lock
 */
void kernel_detect_component_changes(
    Block* d_blocks,
    uint32_t current_tick,
    int size_x, int size_y, int size_z
);

/* ── Block Movement Kernel ────────────────────────────────────────────────
 * TODO v0.5: implement
 */
void kernel_move_blocks(
    Block* d_blocks,
    const TickEvent* d_events,
    int event_count,
    int size_x, int size_y, int size_z
);

#ifdef __cplusplus
}
#endif

#endif /* REDSTONE_KERNELS_H */
