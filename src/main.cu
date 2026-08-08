#include "redstone/world.h"
#include "redstone/block_registry.h"
#include <stdio.h>

/* Test the user's MC-built XOR gate (redstone.schem) with all 4 input combos */
int main(void) {
    printf("╔══════════════════════════════════════════╗\n");
    printf("║   MC XOR Gate Test (redstone.schem)      ║\n");
    printf("╚══════════════════════════════════════════╝\n\n");

    registry_init();

    int cases[][2] = { {0,0}, {0,1}, {1,0}, {1,1} };
    printf("  ┌─────┬─────┬────────┬──────────┐\n");
    printf("  │  A  │  B  │ A XOR B│  Result  │\n");
    printf("  ├─────┼─────┼────────┼──────────┤\n");

    int all_pass = 1;
    for (int c = 0; c < 4; c++) {
        int ia = cases[c][0], ib = cases[c][1];
        World* w = world_create(8, 8, 16);
        world_import_schem(w, "schematics/redstone.schem");

        /* Remove floor (build platform, not part of circuit) */
        for (int fz = 0; fz < 15; fz++)
            for (int fx = 0; fx < 8; fx++)
                world_set_block(w, fx, 0, fz, BLOCK_AIR, DIR_NONE, 0);

        /* Remove both levers, place only the ones we want ON.
         * redstone.schem levers are at z=11.
         * MC lever[face=floor] → our facing = DIR_DOWN */
        world_set_block(w, 1, 1, 11, BLOCK_AIR, DIR_NONE, 0);
        world_set_block(w, 3, 1, 11, BLOCK_AIR, DIR_NONE, 0);
        if (ia) world_set_block(w, 1, 1, 11, BLOCK_LEVER, DIR_DOWN, 0);
        if (ib) world_set_block(w, 3, 1, 11, BLOCK_LEVER, DIR_DOWN, 0);

        world_tick_n(w, 500);
        world_sync_from_gpu(w);

        Block lamp = world_get_block(w, 2, 1, 1);
        int result = !!(lamp.flags & BLOCK_FLAG_POWERED);
        int expected = ia ^ ib;
        int pass = (result == expected);
        if (!pass) all_pass = 0;

        printf("  │ %s │ %s │   %d    │  %s  %s │\n",
               ia ? "ON " : "OFF", ib ? "ON " : "OFF",
               expected,
               result ? "ON " : "OFF",
               pass ? "  " : "!!");

        /* Debug failing cases */
        if (!pass) {
            printf("  │ DEBUG y=1 blocks:\n");
            for (int zz = 0; zz < 15; zz++)
                for (int xx = 0; xx < 5; xx++) {
                    Block b = world_get_block(w, xx, 1, zz);
                    if (b.type != BLOCK_AIR)
                        printf("  │  (%d,1,%d) t=%d s=%d f=%d fl=%02x\n",
                               xx, zz, b.type, b.signal, b.facing, b.flags);
                }
            printf("  │ DEBUG y=2 blocks:\n");
            for (int zz = 0; zz < 15; zz++)
                for (int xx = 0; xx < 5; xx++) {
                    Block b = world_get_block(w, xx, 2, zz);
                    if (b.type != BLOCK_AIR)
                        printf("  │  (%d,2,%d) t=%d s=%d f=%d fl=%02x\n",
                               xx, zz, b.type, b.signal, b.facing, b.flags);
                }
        }

        world_destroy(w);
    }

    printf("  └─────┴─────┴────────┴──────────┘\n");
    printf("\n%s\n", all_pass ? "ALL CORRECT" : "NEEDS FIX");
    return all_pass ? 0 : 1;
}
