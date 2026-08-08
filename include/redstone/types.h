#ifndef REDSTONE_TYPES_H
#define REDSTONE_TYPES_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Block Type IDs ─────────────────────────────────────────────────────────
 * Every block type in the simulation gets a unique ID.
 * v0.1 only uses AIR, SOLID, REDSTONE_DUST, LEVER.
 * The rest are reserved so the enum stays stable across versions.
 */
typedef enum {
    BLOCK_AIR             = 0,
    BLOCK_SOLID           = 1,   /* stone, dirt, etc. — opaque, conducts power */

    /* Power sources */
    BLOCK_LEVER           = 10,
    BLOCK_BUTTON          = 11,
    BLOCK_PRESSURE_PLATE  = 12,
    BLOCK_REDSTONE_BLOCK  = 13,

    /* Transmission */
    BLOCK_REDSTONE_DUST   = 20,

    /* Logic */
    BLOCK_REDSTONE_TORCH  = 30,
    BLOCK_REPEATER        = 31,
    BLOCK_COMPARATOR      = 32,

    /* Mechanical */
    BLOCK_PISTON          = 40,
    BLOCK_STICKY_PISTON   = 41,
    BLOCK_PISTON_HEAD     = 42,
    BLOCK_SLIME           = 43,
    BLOCK_HONEY           = 44,

    /* Actionable */
    BLOCK_OBSERVER        = 50,
    BLOCK_DROPPER         = 51,
    BLOCK_DISPENSER       = 52,
    BLOCK_HOPPER          = 53,
    BLOCK_LAMP            = 54,
    BLOCK_NOTE_BLOCK      = 55,
    BLOCK_TNT             = 56,
    BLOCK_TARGET          = 57,

    /* Rails */
    BLOCK_RAIL            = 60,
    BLOCK_POWERED_RAIL    = 61,
    BLOCK_DETECTOR_RAIL   = 62,
    BLOCK_ACTIVATOR_RAIL  = 63,

    BLOCK_TYPE_COUNT      = 128
} BlockType;

/* ── Direction / Facing ─────────────────────────────────────────────────── */
typedef enum {
    DIR_DOWN  = 0,   /* -Y */
    DIR_UP    = 1,   /* +Y */
    DIR_NORTH = 2,   /* -Z */
    DIR_SOUTH = 3,   /* +Z */
    DIR_WEST  = 4,   /* -X */
    DIR_EAST  = 5,   /* +X */
    DIR_NONE  = 6,
    DIR_COUNT = 6
} Direction;

/* Opposite direction lookup */
static const Direction DIR_OPPOSITE[] = {
    DIR_UP, DIR_DOWN, DIR_SOUTH, DIR_NORTH, DIR_EAST, DIR_WEST, DIR_NONE
};

/* Direction offsets: dx, dy, dz for each direction */
static const int DIR_DX[] = {  0,  0,  0,  0, -1,  1,  0 };
static const int DIR_DY[] = { -1,  1,  0,  0,  0,  0,  0 };
static const int DIR_DZ[] = {  0,  0, -1,  1,  0,  0,  0 };

/* ── Block Flags (bitfield) ─────────────────────────────────────────────── */
#define BLOCK_FLAG_POWERED       (1 << 0)  /* receiving power */
#define BLOCK_FLAG_LOCKED        (1 << 1)  /* repeater locked */
#define BLOCK_FLAG_EXTENDED      (1 << 2)  /* piston extended */
#define BLOCK_FLAG_SCHEDULED     (1 << 3)  /* has pending tick event */
#define BLOCK_FLAG_STRONG_POWER  (1 << 4)  /* providing strong power */
#define BLOCK_FLAG_UPDATED       (1 << 5)  /* was updated this tick */

/* ── Block State Encoding ─────────────────────────────────────────────────
 * The 16-bit `state` field is interpreted differently per block type:
 *
 * REPEATER:   [delay:2][locked:1][powered:1][unused:12]
 * COMPARATOR: [mode:1][output:4][unused:11]  (mode: 0=compare, 1=subtract)
 * PISTON:     [extended:1][sticky:1][unused:14]
 * DUST:       [connections:8][unused:8]  (NSEW up/down connectivity)
 *
 * For v0.1 we mostly use `signal` and `flags` directly.
 */

/* ── Block struct ─────────────────────────────────────────────────────────
 * 8 bytes per block. At 256^3 = 16.7M blocks, that's 128 MB on GPU.
 * Designed once, used across all versions.
 */
#pragma pack(push, 1)
typedef struct {
    uint8_t  type;           /* BlockType enum */
    uint8_t  signal;         /* redstone signal level 0-15 */
    uint8_t  facing;         /* Direction enum */
    uint8_t  flags;          /* BLOCK_FLAG_* bitfield */
    uint16_t state;          /* type-specific data */
    uint16_t tick_scheduled; /* game tick when next event fires (0 = none) */
} Block;
#pragma pack(pop)

/* Compile-time check: Block must be exactly 8 bytes */
static_assert(sizeof(Block) == 8, "Block struct must be 8 bytes");

/* ── Block Definition (for registry) ──────────────────────────────────────
 * Data-driven description of each block type's properties.
 * Adding a new component = adding an entry here + writing its kernel.
 */
typedef struct {
    uint8_t  id;
    const char* name;

    /* Signal behavior */
    uint8_t  is_power_source;     /* generates signal? (lever, torch, etc.) */
    uint8_t  conducts_power;      /* opaque block that transmits power? */
    uint8_t  is_transparent;      /* signal passes through? (glass, air) */
    uint8_t  max_signal;          /* max signal this block can output (usually 15) */

    /* Orientation */
    uint8_t  has_facing;          /* needs direction? (repeater, piston) */

    /* Tick behavior */
    uint8_t  has_tick_behavior;   /* needs scheduled tick updates? */
    uint8_t  base_tick_delay;     /* default delay in game ticks */

    /* Mechanical */
    uint8_t  is_movable;          /* can piston push this? */
    uint8_t  quasi_connectivity;  /* affected by block above? (piston, dropper) */

    /* Signal propagation */
    uint8_t  weakly_powers;       /* provides weak power to adjacent blocks? */
    uint8_t  strongly_powers;     /* provides strong power to attached block? */
} BlockDef;

/* ── Tick Event ───────────────────────────────────────────────────────────
 * Represents a scheduled event in the tick queue.
 * Used by repeaters, torches, comparators, pistons, etc.
 */
typedef enum {
    TICK_PRIORITY_HIGH   = 0,  /* tile tick (repeaters, torches) */
    TICK_PRIORITY_NORMAL = 1,  /* block updates (signal propagation) */
    TICK_PRIORITY_LOW    = 2,  /* block events (pistons) */
    TICK_PRIORITY_COUNT  = 3
} TickPriority;

typedef struct {
    int32_t  x, y, z;           /* block position */
    uint16_t tick;               /* game tick when this fires */
    uint8_t  priority;           /* TickPriority */
    uint8_t  type;               /* BlockType that scheduled this */
} TickEvent;

/* ── World dimensions ───────────────────────────────────────────────────── */
#define WORLD_DEFAULT_SIZE_X 256
#define WORLD_DEFAULT_SIZE_Y 256
#define WORLD_DEFAULT_SIZE_Z 256

/* Max tick events per tick (GPU buffer size) */
#define MAX_TICK_EVENTS 65536

#ifdef __cplusplus
}
#endif

#endif /* REDSTONE_TYPES_H */
