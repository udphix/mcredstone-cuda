#ifndef REDSTONE_SIGNAL_DEVICE_CUH
#define REDSTONE_SIGNAL_DEVICE_CUH

/* ══════════════════════════════════════════════════════════════════════════
 * Signal propagation — DEVICE-SIDE LOGIC (shared by all launch shapes)
 *
 * This header holds the *semantics*: how one cell computes its next state.
 * It is included by
 *   - kernels/signal.cu  → single-world kernel (one thread per cell)
 *   - kernels/batch.cu   → batched kernel   (one thread per cell per world)
 *
 * Both launch shapes MUST call `signal_update_cell` rather than reimplement
 * the rules. If you fix a redstone rule, fix it here and every consumer
 * inherits it.
 *
 * Faithful to Minecraft 1.21 (decompiled). Two signal types
 * (from SignalGetter.java):
 *   getSignal(direction)       = weak power emitted toward a direction
 *   getDirectSignal(direction) = strong power delivered INTO a solid block
 *
 * The critical rule: when a solid/conductor block is queried for signal,
 * it returns max(own_getSignal, max_strong_power_received). This is how
 * solid blocks RELAY strong power as weak power to all neighbors.
 * ══════════════════════════════════════════════════════════════════════════ */

#include "redstone/types.h"
#include <cuda_runtime.h>

/* ── Direction helpers ────────────────────────────────────────────────── */

__device__ static inline int dir_dx(int d) {
    const int t[] = { 0, 0, 0, 0, -1, 1, 0 }; return t[d < 7 ? d : 6];
}
__device__ static inline int dir_dy(int d) {
    const int t[] = { -1, 1, 0, 0, 0, 0, 0 }; return t[d < 7 ? d : 6];
}
__device__ static inline int dir_dz(int d) {
    const int t[] = { 0, 0, -1, 1, 0, 0, 0 }; return t[d < 7 ? d : 6];
}
__device__ static inline int dir_opposite(int d) {
    const int t[] = { 1, 0, 3, 2, 5, 4, 6 }; return t[d < 7 ? d : 6];
}

/* Direction from offset */
__device__ static inline int dir_from_offset(int dx, int dy, int dz) {
    if (dy == -1) return DIR_DOWN;  if (dy == 1) return DIR_UP;
    if (dz == -1) return DIR_NORTH; if (dz == 1) return DIR_SOUTH;
    if (dx == -1) return DIR_WEST;  if (dx == 1) return DIR_EAST;
    return DIR_NONE;
}

/* ── Block access helpers ─────────────────────────────────────────────── */

__device__ static inline int idx3d(int x, int y, int z, int sx, int sz) {
    return (y * sz + z) * sx + x;
}

__device__ static inline int in_bounds(int x, int y, int z, int sx, int sy, int sz) {
    return x >= 0 && x < sx && y >= 0 && y < sy && z >= 0 && z < sz;
}

__device__ static inline Block get_block(const Block* b, int x, int y, int z,
                                          int sx, int sy, int sz) {
    Block empty = {0,0,0,0,0,0};
    if (!in_bounds(x, y, z, sx, sy, sz)) return empty;
    return b[idx3d(x, y, z, sx, sz)];
}

__device__ static inline int is_conductor(uint8_t type) {
    return type == BLOCK_SOLID || type == BLOCK_LAMP;
}

__device__ static inline int has_quasi(uint8_t type) {
    return type == BLOCK_PISTON || type == BLOCK_STICKY_PISTON ||
           type == BLOCK_DROPPER || type == BLOCK_DISPENSER;
}

__device__ static inline int is_signal_source(uint8_t type) {
    switch (type) {
    case BLOCK_LEVER:
    case BLOCK_BUTTON:
    case BLOCK_REDSTONE_TORCH:
    case BLOCK_REPEATER:
    case BLOCK_COMPARATOR:
    case BLOCK_REDSTONE_BLOCK:
        return 1;
    default:
        return 0;
    }
}

__device__ static inline int can_survive_on(uint8_t type) {
    return is_conductor(type) || type == BLOCK_HOPPER;
}

enum {
    RS_SIDE_NONE = 0,
    RS_SIDE_SIDE = 1,
    RS_SIDE_UP   = 2
};

__device__ static inline int should_connect_to(const Block* b, int dir) {
    if (dir == DIR_NONE) {
        return b->type == BLOCK_REDSTONE_DUST;
    }
    if (b->type == BLOCK_REDSTONE_DUST) return 1;
    if (b->type == BLOCK_REPEATER) {
        int f = (int)b->facing;
        return f == dir || f == dir_opposite(dir);
    }
    if (b->type == BLOCK_OBSERVER) {
        return (int)b->facing == dir;
    }
    return is_signal_source(b->type);
}

__device__ static inline int dust_getConnectingSide(
    const Block* blocks,
    int x, int y, int z,
    int dir,
    int sx, int sy, int sz
) {
    if (dir == DIR_UP || dir == DIR_DOWN || dir == DIR_NONE) return RS_SIDE_NONE;

    Block above_self = get_block(blocks, x, y + 1, z, sx, sy, sz);
    int can_raise = !is_conductor(above_self.type);

    int nx = x + dir_dx(dir);
    int ny = y;
    int nz = z + dir_dz(dir);
    Block nb = get_block(blocks, nx, ny, nz, sx, sy, sz);

    if (can_raise) {
        int bl2 = can_survive_on(nb.type);
        if (bl2) {
            Block nb_above = get_block(blocks, nx, ny + 1, nz, sx, sy, sz);
            if (should_connect_to(&nb_above, DIR_NONE)) {
                if (is_conductor(nb.type)) return RS_SIDE_UP;
                return RS_SIDE_SIDE;
            }
        }
    }

    if (should_connect_to(&nb, dir)) {
        return RS_SIDE_SIDE;
    }
    if (!is_conductor(nb.type)) {
        Block nb_below = get_block(blocks, nx, ny - 1, nz, sx, sy, sz);
        if (should_connect_to(&nb_below, DIR_NONE)) {
            return RS_SIDE_SIDE;
        }
    }
    return RS_SIDE_NONE;
}

/* Compute dust connection state with MC 1.21 auto-cross logic.
 * Returns 1 if dust emits signal toward `query_dir`, 0 otherwise.
 *
 * MC rules (RedStoneWireBlock.getConnectionState):
 * - Compute physical connections on all 4 horizontal sides
 * - If dust was stored as dot AND derived state is still dot → stay dot (no emission)
 * - Otherwise, auto-cross: if no N/S → force E/W to SIDE; if no E/W → force N/S to SIDE
 * - getSignal(direction) returns signal if the OPPOSITE side is connected */
__device__ static inline int dust_emits_toward(
    const Block* blocks,
    int x, int y, int z,
    int query_dir,   /* the face being queried by neighbor (MC dir convention) */
    int sx, int sy, int sz
) {
    /* MC RedStoneWireBlock.getSignal (direction convention: asker→source):
     *   direction == DOWN → 0 (dust never emits to asker above it)
     *   direction == UP   → n (dust strong-powers support block below)
     * Our convention: query_dir = source→asker (opposite).
     *   query_dir == UP   = asker above dust → 0
     *   query_dir == DOWN = asker below dust → signal (support block) */
    if (query_dir == DIR_UP)   return 0;  /* asker above dust: no emission */
    if (query_dir == DIR_DOWN) return 1;  /* asker below dust: strong to support */

    /* Check physical connection on each of the 4 horizontal sides */
    int conn_n = dust_getConnectingSide(blocks, x, y, z, DIR_NORTH, sx, sy, sz) != RS_SIDE_NONE;
    int conn_s = dust_getConnectingSide(blocks, x, y, z, DIR_SOUTH, sx, sy, sz) != RS_SIDE_NONE;
    int conn_e = dust_getConnectingSide(blocks, x, y, z, DIR_EAST,  sx, sy, sz) != RS_SIDE_NONE;
    int conn_w = dust_getConnectingSide(blocks, x, y, z, DIR_WEST,  sx, sy, sz) != RS_SIDE_NONE;

    /* MC RedStoneWireBlock.getConnectionState auto-cross logic:
     * Save whether each axis had ANY connections BEFORE modifying, then
     * force both ends of the OPPOSITE axis if none. Order matters — in
     * MC both conditions are evaluated against the original flags.
     * A fully-isolated dust ends up "all SIDE" (cross mode), so it still
     * emits to all 4 horizontal directions. Previous `!has_any → 0` early
     * return treated isolated dust as dot mode, but MC keeps such dusts
     * in cross mode unless explicitly marked dot (persistent state we
     * don't currently track). "Always auto-cross" matches MC behavior for
     * the vast majority of circuits; dot-only mode would need a per-dust
     * flag imported from schem palette. */
    int had_ns = conn_n || conn_s;
    int had_ew = conn_e || conn_w;
    if (!had_ns) { conn_e = 1; conn_w = 1; }
    if (!had_ew) { conn_n = 1; conn_s = 1; }

    /* MC RedStoneWireBlock.getSignal(direction): returns n if
     *   connection_state[direction.getOpposite()].isConnected()
     * where MC's `direction` is asker->source (opposite of our query_dir).
     * So MC's `direction.getOpposite()` equals our `query_dir` directly.
     * Example: stone asker WEST of dust → MC direction=EAST → check conn
     * on EAST.opposite=WEST → conn_w. Our query_dir=WEST (source→asker)
     * → check_dir = WEST → conn_w. Match.
     * Previous code used dir_opposite(query_dir) which was inverted.
     * The inversion was masked by the auto-cross fallback: when a dust
     * had connections only on one axis, auto-cross forced both ends of
     * the other axis, and the buggy `opposite` check accidentally hit
     * the forced side. For dusts with explicit asymmetric connections
     * (e.g., dust at (2,0,1) with only W+S set), the bug surfaced. */
    int check_dir = query_dir;
    switch (check_dir) {
    case DIR_NORTH: return conn_n;
    case DIR_SOUTH: return conn_s;
    case DIR_EAST:  return conn_e;
    case DIR_WEST:  return conn_w;
    default: return 0;
    }
}

__device__ static inline uint8_t getDustSignalFrom(
    const Block* blocks,
    int x, int y, int z,
    int toward_dir,
    int sx, int sy, int sz
) {
    Block b = get_block(blocks, x, y, z, sx, sy, sz);
    if (b.signal == 0) return 0;

    /* Use auto-cross connection logic from MC.
     * toward_dir = direction from dust TO querying block.
     * In MC convention, getSignal receives the FACE direction = same as toward_dir
     * when dust emits signal through that face toward the neighbor. */
    return dust_emits_toward(blocks, x, y, z, toward_dir, sx, sy, sz) ? b.signal : 0;
}

__device__ static inline uint8_t getDirectSignal(const Block* b, int toward_dir);

__device__ static inline uint8_t getDirectSignalFrom(
    const Block* blocks,
    int x, int y, int z,
    int toward_dir,
    int sx, int sy, int sz
) {
    Block b = get_block(blocks, x, y, z, sx, sy, sz);
    if (b.type == BLOCK_REDSTONE_DUST) {
        return getDustSignalFrom(blocks, x, y, z, toward_dir, sx, sy, sz);
    }
    return getDirectSignal(&b, toward_dir);
}

/* ══════════════════════════════════════════════════════════════════════════
 * getSignal: weak power a block EMITS toward direction `toward_dir`
 *
 * This is what a non-solid neighbor reads from this block.
 * ══════════════════════════════════════════════════════════════════════════ */
__device__ static inline uint8_t getSignal(const Block* b, int toward_dir) {
    switch (b->type) {

    /* Lever: weak 15 in ALL directions when powered */
    case BLOCK_LEVER:
    case BLOCK_BUTTON:
        return (b->flags & BLOCK_FLAG_POWERED) ? 15 : 0;

    /* Redstone Block: weak 15 in all directions (NO strong power) */
    case BLOCK_REDSTONE_BLOCK:
        return 15;

    /* Torch getSignal (MC RedstoneTorchBlock + RedstoneWallTorchBlock):
     * MC rule: `FACING != direction ? 15 : 0` where `direction` in MC = from asker to source.
     * Our `toward_dir` = from source to asker = opposite(MC direction).
     * So our rule: emit 0 when toward_dir == opposite(MC FACING) = our facing
     * (since our facing = opposite(MC facing) = attached direction).
     *
     * Semantically: torch does NOT emit signal to its attached block (the direction
     * in our `facing`). It emits 15 to all other directions. This prevents the
     * torch from powering its own attached block, avoiding feedback loops. */
    case BLOCK_REDSTONE_TORCH:
        if (b->signal == 0) return 0;
        return (toward_dir == (int)b->facing) ? 0 : 15;

    /* Repeater/Comparator: signal ONLY in facing direction */
    case BLOCK_REPEATER:
    case BLOCK_COMPARATOR:
        if (b->signal == 0) return 0;
        return (toward_dir == (int)b->facing) ? b->signal : 0;

    /* Dust: same convention flip as in dust_emits_toward.
     * MC: direction==DOWN→0, direction==UP→n.
     * Our toward_dir (source→asker) is inverted: UP→0, DOWN→n. */
    case BLOCK_REDSTONE_DUST:
        if (b->signal == 0) return 0;
        if (toward_dir == DIR_UP) return 0;
        return b->signal;

    default:
        return 0;
    }
}

/* ══════════════════════════════════════════════════════════════════════════
 * getDirectSignal: strong power delivered INTO a solid block
 *
 * Only solid/conductor blocks receive this. Strong power flows THROUGH
 * the solid and comes out as weak power on all other faces.
 * ══════════════════════════════════════════════════════════════════════════ */
__device__ static inline uint8_t getDirectSignal(const Block* b, int toward_dir) {
    switch (b->type) {

    /* Lever: strong 15 ONLY toward the block it's attached to.
     * In our model, facing = attached direction, so strong toward facing. */
    case BLOCK_LEVER:
    case BLOCK_BUTTON:
        if (!(b->flags & BLOCK_FLAG_POWERED)) return 0;
        return (toward_dir == (int)b->facing) ? 15 : 0;

    /* Torch strong power: MC RedstoneTorchBlock.getDirectSignal returns
     * getSignal only when direction==DOWN (in MC conv, direction=DOWN means
     * asker is above source). Semantically: torch provides strong power
     * ONLY to the block directly ABOVE it.
     *
     * Our `toward_dir` = source→asker. Asker above source → toward_dir=UP.
     * So emit strong 15 only when toward_dir==UP. */
    case BLOCK_REDSTONE_TORCH:
        if (b->signal == 0) return 0;
        return (toward_dir == DIR_UP) ? 15 : 0;

    /* Repeater: strong in facing direction (same as weak) */
    case BLOCK_REPEATER:
    case BLOCK_COMPARATOR:
        if (b->signal == 0) return 0;
        return (toward_dir == (int)b->facing) ? b->signal : 0;

    /* Dust: strong power = same as weak (MC: getDirectSignal delegates to
     * getSignal). Convention flip: our toward_dir=UP → 0, DOWN → signal. */
    case BLOCK_REDSTONE_DUST:
        if (b->signal == 0) return 0;
        if (toward_dir == DIR_UP) return 0;
        return b->signal;

    /* Redstone Block: NO strong power (only weak) */
    case BLOCK_REDSTONE_BLOCK:
        return 0;

    default:
        return 0;
    }
}

/* ══════════════════════════════════════════════════════════════════════════
 * getSignalFrom: what signal does position (nx,ny,nz) provide toward
 * the querying block in direction `toward_dir`?
 *
 * This is the MC equivalent of SignalGetter.getSignal():
 *   If the source is a conductor → max(getSignal, strong_power_received)
 *   Otherwise → getSignal
 * ══════════════════════════════════════════════════════════════════════════ */
__device__ static inline uint8_t getSignalFrom(
    const Block* blocks,
    int nx, int ny, int nz,  /* source position */
    int toward_dir,          /* direction FROM source TO querying block */
    int sx, int sy, int sz
) {
    Block nb = get_block(blocks, nx, ny, nz, sx, sy, sz);
    uint8_t weak = nb.type == BLOCK_REDSTONE_DUST
        ? getDustSignalFrom(blocks, nx, ny, nz, toward_dir, sx, sy, sz)
        : getSignal(&nb, toward_dir);

    /* If source is a conductor, it relays strong power as weak */
    if (is_conductor(nb.type)) {
        /* Check all 6 faces for strong power INTO this conductor */
        const int offs[][3] = {{0,-1,0},{0,1,0},{0,0,-1},{0,0,1},{-1,0,0},{1,0,0}};
        for (int i = 0; i < 6; i++) {
            int sx2 = nx + offs[i][0], sy2 = ny + offs[i][1], sz2 = nz + offs[i][2];
            /* Direction from neighbor INTO this conductor */
            int into_dir = dir_from_offset(-offs[i][0], -offs[i][1], -offs[i][2]);
            uint8_t ds = getDirectSignalFrom(blocks, sx2, sy2, sz2, into_dir, sx, sy, sz);
            if (ds > weak) weak = ds;
        }
    }

    return weak;
}

/* ══════════════════════════════════════════════════════════════════════════
 * Max strong power a solid block receives from all neighbors
 * ══════════════════════════════════════════════════════════════════════════ */
__device__ static inline uint8_t getMaxStrongPower(
    const Block* blocks,
    int bx, int by, int bz,
    int sx, int sy, int sz
) {
    uint8_t max_strong = 0;
    const int offs[][3] = {{0,-1,0},{0,1,0},{0,0,-1},{0,0,1},{-1,0,0},{1,0,0}};
    for (int i = 0; i < 6; i++) {
        int nx = bx + offs[i][0], ny = by + offs[i][1], nz = bz + offs[i][2];
        int into_dir = dir_from_offset(-offs[i][0], -offs[i][1], -offs[i][2]);
        uint8_t ds = getDirectSignalFrom(blocks, nx, ny, nz, into_dir, sx, sy, sz);
        if (ds > max_strong) max_strong = ds;
    }
    return max_strong;
}

/* ══════════════════════════════════════════════════════════════════════════
 * hasNeighborSignal: does position (bx,by,bz) receive ANY signal > 0?
 * Used for lamps, pistons, etc.
 * ══════════════════════════════════════════════════════════════════════════ */
__device__ static inline int hasNeighborSignal(
    const Block* blocks,
    int bx, int by, int bz,
    int sx, int sy, int sz
) {
    const int offs[][3] = {{0,-1,0},{0,1,0},{0,0,-1},{0,0,1},{-1,0,0},{1,0,0}};
    for (int i = 0; i < 6; i++) {
        int nx = bx + offs[i][0], ny = by + offs[i][1], nz = bz + offs[i][2];
        int toward = dir_from_offset(-offs[i][0], -offs[i][1], -offs[i][2]);
        if (getSignalFrom(blocks, nx, ny, nz, toward, sx, sy, sz) > 0)
            return 1;
    }
    return 0;
}

/* ══════════════════════════════════════════════════════════════════════════
 * PER-CELL UPDATE — the whole of "block updates" for one cell.
 *
 * `src`/`dst` point at the base of ONE world's block grid; `i` is the
 * linear index of the cell within that grid. Batched callers just offset
 * the base pointers per instance — every neighbour lookup is already
 * bounds-checked against (size_x, size_y, size_z), so worlds cannot leak
 * into each other.
 * ══════════════════════════════════════════════════════════════════════════ */
__device__ static inline void signal_update_cell(
    const Block* __restrict__ src,
    Block* __restrict__ dst,
    int i,
    int size_x, int size_y, int size_z
) {
    int x = i % size_x;
    int z = (i / size_x) % size_z;
    int y = i / (size_x * size_z);

    Block block = src[i];
    dst[i] = block;

    /* ── Redstone Dust ─────────────────────────────────────────────── */
    if (block.type == BLOCK_REDSTONE_DUST) {
        uint8_t max_sig = 0;

        /* Check all 6 neighbors for non-dust signal (MC: getBestNeighborSignal)
         * MC uses shouldSignal=false to prevent dust→solid→dust loops.
         * GPU equivalent: for conductor neighbors, only count strong power
         * from NON-DUST sources (exclude dust's own getDirectSignal). */
        const int offs[][3] = {{0,-1,0},{0,1,0},{0,0,-1},{0,0,1},{-1,0,0},{1,0,0}};
        for (int j = 0; j < 6; j++) {
            int nx = x+offs[j][0], ny = y+offs[j][1], nz = z+offs[j][2];
            Block nb = get_block(src, nx, ny, nz, size_x, size_y, size_z);
            if (nb.type == BLOCK_REDSTONE_DUST) continue; /* dust handled below */
            int toward = dir_from_offset(-offs[j][0], -offs[j][1], -offs[j][2]);

            if (is_conductor(nb.type)) {
                /* For conductors: only relay strong power from non-dust sources
                 * This mimics MC's shouldSignal=false during dust recalculation */
                const int offs2[][3] = {{0,-1,0},{0,1,0},{0,0,-1},{0,0,1},{-1,0,0},{1,0,0}};
                for (int k = 0; k < 6; k++) {
                    int sx2 = nx+offs2[k][0], sy2 = ny+offs2[k][1], sz2 = nz+offs2[k][2];
                    Block src2 = get_block(src, sx2, sy2, sz2, size_x, size_y, size_z);
                    if (src2.type == BLOCK_REDSTONE_DUST) continue; /* skip dust */
                    int into = dir_from_offset(-offs2[k][0], -offs2[k][1], -offs2[k][2]);
                    uint8_t ds = getDirectSignal(&src2, into);
                    if (ds > max_sig) max_sig = ds;
                }
            } else {
                uint8_t s = getSignal(&nb, toward);
                if (s > max_sig) max_sig = s;
            }
        }

        /* Check horizontal neighbors for other dust (MC: getWireSignal) */
        const int hdx[] = { 0, 0, -1, 1 };
        const int hdz[] = { -1, 1, 0, 0 };
        const int hdir[] = { DIR_NORTH, DIR_SOUTH, DIR_WEST, DIR_EAST };
        for (int j = 0; j < 4; j++) {
            int nx = x+hdx[j], nz = z+hdz[j];
            int conn = dust_getConnectingSide(src, x, y, z, hdir[j], size_x, size_y, size_z);
            if (conn == RS_SIDE_NONE) continue;

            /* Same level dust */
            Block adj = get_block(src, nx, y, nz, size_x, size_y, size_z);
            if (conn == RS_SIDE_UP) {
                Block up = get_block(src, nx, y+1, nz, size_x, size_y, size_z);
                if (up.type == BLOCK_REDSTONE_DUST && up.signal > 1) {
                    uint8_t s = up.signal - 1;
                    if (s > max_sig) max_sig = s;
                }
                continue;
            }
            if (adj.type == BLOCK_REDSTONE_DUST && adj.signal > 0) {
                uint8_t s = adj.signal - 1;
                if (s > max_sig) max_sig = s;
            }

            /* Down stairs: dust below adjacent non-conductor */
            if (!is_conductor(adj.type) && adj.type != BLOCK_REDSTONE_DUST) {
                Block dn = get_block(src, nx, y-1, nz, size_x, size_y, size_z);
                if (dn.type == BLOCK_REDSTONE_DUST && dn.signal > 1) {
                    uint8_t s = dn.signal - 1;
                    if (s > max_sig) max_sig = s;
                }
            }
        }

        dst[i].signal = max_sig;
        dst[i].flags = max_sig > 0 ? BLOCK_FLAG_POWERED : 0;
    }
    /* ── Solid/Conductor block ─────────────────────────────────────── */
    else if (is_conductor(block.type)) {
        uint8_t strong = getMaxStrongPower(src, x, y, z, size_x, size_y, size_z);

        /* Weak power from ANY neighbor. LAMP reads via getSignalFrom (which
         * applies MC's one-hop conductor-relay) so that a lamp adjacent to
         * a strongly-powered stone (dust above, torch below, etc.) lights
         * up. SOLID reads via getSignal (no relay) to avoid spurious
         * stone-to-stone propagation that diverges from the GT transient
         * states in xor_000/004/007/008. Empirically: MC's exported GTs
         * behave as if the solid-to-lamp relay is applied but
         * solid-to-solid is not — matches the "lazy re-evaluation" model
         * where some blocks' POWERED flag is only updated on specific
         * triggers (lamp neighborChanged, etc.). */
        uint8_t weak = 0;
        const int is_lamp = (block.type == BLOCK_LAMP);
        const int offs[][3] = {{0,-1,0},{0,1,0},{0,0,-1},{0,0,1},{-1,0,0},{1,0,0}};
        for (int j = 0; j < 6; j++) {
            int nx = x+offs[j][0], ny = y+offs[j][1], nz = z+offs[j][2];
            int toward = dir_from_offset(-offs[j][0], -offs[j][1], -offs[j][2]);
            uint8_t s;
            if (is_lamp) {
                s = getSignalFrom(src, nx, ny, nz, toward, size_x, size_y, size_z);
            } else {
                Block nb = get_block(src, nx, ny, nz, size_x, size_y, size_z);
                if (nb.type == BLOCK_REDSTONE_DUST) {
                    s = getDustSignalFrom(src, nx, ny, nz, toward, size_x, size_y, size_z);
                } else {
                    s = getSignal(&nb, toward);
                }
            }
            if (s > weak) weak = s;
        }

        uint8_t total_sig = strong > weak ? strong : weak;
        dst[i].signal = total_sig;
        dst[i].flags = 0;
        if (total_sig > 0) dst[i].flags |= BLOCK_FLAG_POWERED;
        if (strong > 0) dst[i].flags |= BLOCK_FLAG_STRONG_POWER;
    }
    /* ── Torch / Repeater — state managed by component kernel ──────── */
    else if (block.type == BLOCK_REDSTONE_TORCH ||
             block.type == BLOCK_REPEATER ||
             block.type == BLOCK_COMPARATOR) {
        dst[i] = block;
    }
    /* ── Power sources (lever, redstone block) — keep state ────────── */
    else if (block.type == BLOCK_LEVER || block.type == BLOCK_BUTTON ||
             block.type == BLOCK_REDSTONE_BLOCK) {
        dst[i] = block;
    }
    /* ── Lamp ──────────────────────────────────────────────────────── */
    else if (block.type == BLOCK_LAMP) {
        int powered = hasNeighborSignal(src, x, y, z, size_x, size_y, size_z);
        dst[i].flags = powered ? BLOCK_FLAG_POWERED : 0;
    }
    /* ── Quasi-connectivity (piston, dispenser, dropper) ───────────── */
    else if (has_quasi(block.type)) {
        int powered = hasNeighborSignal(src, x, y, z, size_x, size_y, size_z);
        if (!powered)
            powered = hasNeighborSignal(src, x, y+1, z, size_x, size_y, size_z);
        dst[i].flags = block.flags & ~BLOCK_FLAG_POWERED;
        if (powered) dst[i].flags |= BLOCK_FLAG_POWERED;
    }
}

#endif /* REDSTONE_SIGNAL_DEVICE_CUH */
