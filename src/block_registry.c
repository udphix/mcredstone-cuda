#include "redstone/block_registry.h"
#include <string.h>

/* ── Registry Table ───────────────────────────────────────────────────────
 * Every block type gets an entry here describing its properties.
 * The simulation kernels read these properties to decide behavior.
 *
 * Fields:
 *   id, name,
 *   is_power_source, conducts_power, is_transparent, max_signal,
 *   has_facing, has_tick_behavior, base_tick_delay,
 *   is_movable, quasi_connectivity,
 *   weakly_powers, strongly_powers
 */

static BlockDef g_registry[BLOCK_TYPE_COUNT];
static int g_initialized = 0;

static void register_block(BlockDef def) {
    if (def.id < BLOCK_TYPE_COUNT) {
        g_registry[def.id] = def;
    }
}

void registry_init(void) {
    if (g_initialized) return;
    memset(g_registry, 0, sizeof(g_registry));

    /* ── Air ──────────────────────────────────────────────────────────── */
    register_block((BlockDef){
        .id = BLOCK_AIR, .name = "air",
        .is_transparent = 1, .is_movable = 0
    });

    /* ── Solid ────────────────────────────────────────────────────────── */
    register_block((BlockDef){
        .id = BLOCK_SOLID, .name = "solid",
        .conducts_power = 1, .is_movable = 1
    });

    /* ── Lever ────────────────────────────────────────────────────────── */
    register_block((BlockDef){
        .id = BLOCK_LEVER, .name = "lever",
        .is_power_source = 1, .max_signal = 15,
        .has_facing = 1, .is_transparent = 1,
        .strongly_powers = 1
    });

    /* ── Button ───────────────────────────────────────────────────────── */
    register_block((BlockDef){
        .id = BLOCK_BUTTON, .name = "button",
        .is_power_source = 1, .max_signal = 15,
        .has_facing = 1, .is_transparent = 1,
        .has_tick_behavior = 1, .base_tick_delay = 20, /* 1 second */
        .strongly_powers = 1
    });

    /* ── Redstone Block ───────────────────────────────────────────────── */
    register_block((BlockDef){
        .id = BLOCK_REDSTONE_BLOCK, .name = "redstone_block",
        .is_power_source = 1, .max_signal = 15,
        .is_movable = 1, .weakly_powers = 1
    });

    /* ── Redstone Dust ────────────────────────────────────────────────── */
    register_block((BlockDef){
        .id = BLOCK_REDSTONE_DUST, .name = "redstone_dust",
        .is_transparent = 1, .max_signal = 15,
        .weakly_powers = 1
    });

    /* ── Redstone Torch ───────────────────────────────────────────────── */
    register_block((BlockDef){
        .id = BLOCK_REDSTONE_TORCH, .name = "redstone_torch",
        .is_power_source = 1, .max_signal = 15,
        .has_facing = 1, .is_transparent = 1,
        .has_tick_behavior = 1, .base_tick_delay = 2,
        .strongly_powers = 1
    });

    /* ── Repeater ─────────────────────────────────────────────────────── */
    register_block((BlockDef){
        .id = BLOCK_REPEATER, .name = "repeater",
        .is_power_source = 1, .max_signal = 15,
        .has_facing = 1, .is_transparent = 1,
        .has_tick_behavior = 1, .base_tick_delay = 2,
        .strongly_powers = 1
    });

    /* ── Comparator ───────────────────────────────────────────────────── */
    register_block((BlockDef){
        .id = BLOCK_COMPARATOR, .name = "comparator",
        .is_power_source = 1, .max_signal = 15,
        .has_facing = 1, .is_transparent = 1,
        .has_tick_behavior = 1, .base_tick_delay = 2,
        .strongly_powers = 1
    });

    /* ── Piston ───────────────────────────────────────────────────────── */
    register_block((BlockDef){
        .id = BLOCK_PISTON, .name = "piston",
        .has_facing = 1, .is_transparent = 1,
        .has_tick_behavior = 1, .base_tick_delay = 0,
        .quasi_connectivity = 1
    });

    /* ── Sticky Piston ────────────────────────────────────────────────── */
    register_block((BlockDef){
        .id = BLOCK_STICKY_PISTON, .name = "sticky_piston",
        .has_facing = 1, .is_transparent = 1,
        .has_tick_behavior = 1, .base_tick_delay = 0,
        .quasi_connectivity = 1
    });

    /* ── Lamp ─────────────────────────────────────────────────────────── */
    register_block((BlockDef){
        .id = BLOCK_LAMP, .name = "lamp",
        .conducts_power = 1, .is_movable = 1
    });

    /* ── Observer ─────────────────────────────────────────────────────── */
    register_block((BlockDef){
        .id = BLOCK_OBSERVER, .name = "observer",
        .is_power_source = 1, .max_signal = 15,
        .has_facing = 1,
        .has_tick_behavior = 1, .base_tick_delay = 2,
        .is_movable = 1, .strongly_powers = 1
    });

    g_initialized = 1;
}

const BlockDef* registry_get(BlockType type) {
    if (!g_initialized) registry_init();
    if (type >= BLOCK_TYPE_COUNT) return &g_registry[BLOCK_AIR];
    return &g_registry[type];
}

const char* registry_name(BlockType type) {
    const BlockDef* def = registry_get(type);
    return def->name ? def->name : "unknown";
}

int registry_conducts_power(BlockType type) {
    return registry_get(type)->conducts_power;
}

int registry_is_power_source(BlockType type) {
    return registry_get(type)->is_power_source;
}

int registry_is_transparent(BlockType type) {
    return registry_get(type)->is_transparent;
}
