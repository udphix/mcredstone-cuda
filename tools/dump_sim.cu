/* ═══════════════════════════════════════════════════════════════════════════
 * dump_sim — load a .schem, run N ticks, dump final state.
 *
 * Usage:  dump_sim <schem_path> [ticks] [--strip-floor]
 *
 *   --strip-floor : clear y=0 after import (legacy behaviour; used for the
 *                   original redstone.schem which included a build platform).
 *                   Do NOT use when validating schems whose logic lives at y=0.
 *
 * Output format (stdout, one line per non-air block):
 *   x y z type facing signal flags state
 *
 *   - x, y, z: integers
 *   - type, facing: decimal integers (enum values)
 *   - signal: 0-15
 *   - flags: hex
 *   - state: hex
 * ═══════════════════════════════════════════════════════════════════════════ */

#include "redstone/world.h"
#include "redstone/block_registry.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <schem_path> [ticks]\n", argv[0]);
        return 1;
    }

    const char* schem_path = argv[1];
    int ticks = (argc >= 3) ? atoi(argv[2]) : 500;
    int strip_floor = 0;
    for (int i = 2; i < argc; i++) {
        if (argv[i][0] == '-' && argv[i][1] == '-') {
            if (!strcmp(argv[i], "--strip-floor")) strip_floor = 1;
        }
    }

    registry_init();

    /* Use 8x8x16 world (same as main.cu) */
    World* w = world_create(8, 8, 16);
    if (!w) { fprintf(stderr, "world_create failed\n"); return 2; }

    int placed = world_import_schem(w, schem_path);
    if (placed <= 0) {
        fprintf(stderr, "Failed to import %s (placed=%d)\n", schem_path, placed);
        world_destroy(w);
        return 3;
    }

    if (strip_floor) {
        for (int fz = 0; fz < 15; fz++)
            for (int fx = 0; fx < 8; fx++)
                world_set_block(w, fx, 0, fz, BLOCK_AIR, DIR_NONE, 0);
    }

    world_tick_n(w, ticks);
    world_sync_from_gpu(w);

    /* Dump every non-air block */
    printf("# dump_sim schem=%s ticks=%d world=%dx%dx%d\n",
           schem_path, ticks, w->size_x, w->size_y, w->size_z);
    printf("# x y z type facing signal flags state\n");

    for (int y = 0; y < w->size_y; y++) {
        for (int z = 0; z < w->size_z; z++) {
            for (int x = 0; x < w->size_x; x++) {
                Block b = world_get_block(w, x, y, z);
                if (b.type == BLOCK_AIR) continue;
                printf("%d %d %d %u %u %u 0x%02x 0x%04x\n",
                       x, y, z,
                       (unsigned)b.type, (unsigned)b.facing,
                       (unsigned)b.signal,
                       (unsigned)b.flags, (unsigned)b.state);
            }
        }
    }

    world_destroy(w);
    return 0;
}
