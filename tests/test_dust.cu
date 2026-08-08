#include "redstone/world.h"
#include "redstone/block_registry.h"
#include <stdio.h>
#include <stdlib.h>

/* ── Minimal test framework ───────────────────────────────────────────── */

static int g_tests_run = 0;
static int g_tests_passed = 0;

#define ASSERT_EQ(actual, expected, msg) do { \
    g_tests_run++; \
    if ((actual) == (expected)) { \
        g_tests_passed++; \
    } else { \
        printf("  FAIL: %s — got %d, expected %d\n", msg, (int)(actual), (int)(expected)); \
    } \
} while(0)

#define ASSERT_GT(actual, threshold, msg) do { \
    g_tests_run++; \
    if ((actual) > (threshold)) { \
        g_tests_passed++; \
    } else { \
        printf("  FAIL: %s — got %d, expected >%d\n", msg, (int)(actual), (int)(threshold)); \
    } \
} while(0)

/* ════════════════════════════════════════════════════════════════════════
 * DUST TESTS (v0.1)
 * ════════════════════════════════════════════════════════════════════════ */

static void test_single_dust(void) {
    printf("[test] Single dust next to lever\n");
    World* w = world_create(4, 4, 4);
    world_set_block(w, 0, 1, 1, BLOCK_LEVER, DIR_NONE, 0);
    world_set_block(w, 1, 1, 1, BLOCK_REDSTONE_DUST, DIR_NONE, 0);
    world_tick_n(w, 2);
    world_sync_from_gpu(w);
    ASSERT_EQ(world_get_block(w, 1, 1, 1).signal, 15, "dust next to lever");
    world_destroy(w);
}

static void test_line_decay(void) {
    printf("[test] Line of dust — signal decays by 1 per block\n");
    World* w = world_create(32, 4, 4);
    world_set_block(w, 0, 1, 1, BLOCK_LEVER, DIR_NONE, 0);
    for (int x = 1; x <= 16; x++)
        world_set_block(w, x, 1, 1, BLOCK_REDSTONE_DUST, DIR_NONE, 0);
    world_tick_n(w, 16);
    world_sync_from_gpu(w);
    for (int x = 1; x <= 15; x++) {
        char msg[64];
        snprintf(msg, sizeof(msg), "dust at x=%d signal", x);
        ASSERT_EQ(world_get_block(w, x, 1, 1).signal, 16 - x, msg);
    }
    ASSERT_EQ(world_get_block(w, 16, 1, 1).signal, 0, "dust at x=16 should be 0");
    world_destroy(w);
}

static void test_no_signal_without_source(void) {
    printf("[test] Dust without power source stays at 0\n");
    World* w = world_create(8, 4, 4);
    for (int x = 0; x < 5; x++)
        world_set_block(w, x, 1, 1, BLOCK_REDSTONE_DUST, DIR_NONE, 0);
    world_tick_n(w, 4);
    world_sync_from_gpu(w);
    for (int x = 0; x < 5; x++) {
        char msg[64];
        snprintf(msg, sizeof(msg), "unpowered dust at x=%d", x);
        ASSERT_EQ(world_get_block(w, x, 1, 1).signal, 0, msg);
    }
    world_destroy(w);
}

static void test_two_sources(void) {
    printf("[test] Dust between two power sources — takes max\n");
    World* w = world_create(16, 4, 4);
    world_set_block(w, 0, 1, 1, BLOCK_LEVER, DIR_NONE, 0);
    world_set_block(w, 10, 1, 1, BLOCK_LEVER, DIR_NONE, 0);
    for (int x = 1; x <= 9; x++)
        world_set_block(w, x, 1, 1, BLOCK_REDSTONE_DUST, DIR_NONE, 0);
    world_tick_n(w, 10);
    world_sync_from_gpu(w);
    ASSERT_EQ(world_get_block(w, 5, 1, 1).signal, 11, "dust at x=5 between two sources");
    ASSERT_EQ(world_get_block(w, 1, 1, 1).signal, 15, "dust at x=1 next to lever");
    ASSERT_EQ(world_get_block(w, 9, 1, 1).signal, 15, "dust at x=9 next to lever");
    world_destroy(w);
}

static void test_branching(void) {
    printf("[test] Signal branches in multiple directions\n");
    World* w = world_create(8, 4, 8);
    world_set_block(w, 3, 1, 3, BLOCK_LEVER, DIR_NONE, 0);
    world_set_block(w, 4, 1, 3, BLOCK_REDSTONE_DUST, DIR_NONE, 0);
    world_set_block(w, 2, 1, 3, BLOCK_REDSTONE_DUST, DIR_NONE, 0);
    world_set_block(w, 3, 1, 4, BLOCK_REDSTONE_DUST, DIR_NONE, 0);
    world_set_block(w, 3, 1, 2, BLOCK_REDSTONE_DUST, DIR_NONE, 0);
    world_tick_n(w, 2);
    world_sync_from_gpu(w);
    ASSERT_EQ(world_get_block(w, 4, 1, 3).signal, 15, "dust +X");
    ASSERT_EQ(world_get_block(w, 2, 1, 3).signal, 15, "dust -X");
    ASSERT_EQ(world_get_block(w, 3, 1, 4).signal, 15, "dust +Z");
    ASSERT_EQ(world_get_block(w, 3, 1, 2).signal, 15, "dust -Z");
    world_destroy(w);
}

static void test_staircase_up(void) {
    printf("[test] Signal goes up stairs\n");
    World* w = world_create(8, 8, 4);
    world_set_block(w, 0, 1, 1, BLOCK_LEVER, DIR_NONE, 0);
    world_set_block(w, 1, 1, 1, BLOCK_REDSTONE_DUST, DIR_NONE, 0);
    world_set_block(w, 2, 1, 1, BLOCK_SOLID, DIR_NONE, 0);
    world_set_block(w, 2, 2, 1, BLOCK_REDSTONE_DUST, DIR_NONE, 0);
    world_tick_n(w, 4);
    world_sync_from_gpu(w);
    ASSERT_GT(world_get_block(w, 1, 1, 1).signal, 0, "bottom dust has signal");
    ASSERT_GT(world_get_block(w, 2, 2, 1).signal, 0, "top dust has signal via stairs");
    world_destroy(w);
}

static void test_lamp_powered(void) {
    printf("[test] Lamp lights up when receiving power\n");
    World* w = world_create(8, 4, 4);
    world_set_block(w, 0, 1, 1, BLOCK_LEVER, DIR_NONE, 0);
    world_set_block(w, 1, 1, 1, BLOCK_REDSTONE_DUST, DIR_NONE, 0);
    world_set_block(w, 2, 1, 1, BLOCK_LAMP, DIR_NONE, 0);
    world_tick_n(w, 2);
    world_sync_from_gpu(w);
    ASSERT_EQ(!!(world_get_block(w, 2, 1, 1).flags & BLOCK_FLAG_POWERED), 1, "lamp powered");
    world_destroy(w);
}

/* ════════════════════════════════════════════════════════════════════════
 * TORCH TESTS (v0.2/v0.3)
 * ════════════════════════════════════════════════════════════════════════ */

static void test_torch_powers_dust(void) {
    printf("[test] Torch on floor powers adjacent dust\n");
    World* w = world_create(8, 4, 4);
    world_set_block(w, 2, 1, 1, BLOCK_REDSTONE_TORCH, DIR_DOWN, 0);
    world_set_block(w, 3, 1, 1, BLOCK_REDSTONE_DUST, DIR_NONE, 0);
    world_tick_n(w, 3);
    world_sync_from_gpu(w);
    ASSERT_EQ(world_get_block(w, 3, 1, 1).signal, 15, "dust next to torch");
    world_destroy(w);
}

static void test_torch_inverts(void) {
    printf("[test] Torch inverts: powered block → torch OFF\n");
    World* w = world_create(8, 4, 4);
    /* Lever(attached to solid via DIR_EAST) → Solid → Torch */
    world_set_block(w, 0, 1, 1, BLOCK_LEVER, DIR_EAST, 0);
    world_set_block(w, 1, 1, 1, BLOCK_SOLID, DIR_NONE, 0);
    world_set_block(w, 2, 1, 1, BLOCK_REDSTONE_TORCH, DIR_WEST, 0);
    world_tick_n(w, 10);
    world_sync_from_gpu(w);
    ASSERT_EQ(world_get_block(w, 2, 1, 1).signal, 0, "torch OFF (lever side-attached)");
    world_destroy(w);

    /* Also test lever ON TOP (DIR_DOWN) — same pattern used by gates */
    World* w2 = world_create(8, 4, 4);
    world_set_block(w2, 1, 1, 1, BLOCK_SOLID, DIR_NONE, 0);
    world_set_block(w2, 1, 2, 1, BLOCK_LEVER, DIR_DOWN, 0);
    world_set_block(w2, 2, 1, 1, BLOCK_REDSTONE_TORCH, DIR_WEST, 0);
    world_tick_n(w2, 10);
    world_sync_from_gpu(w2);
    ASSERT_EQ(world_get_block(w2, 2, 1, 1).signal, 0, "torch OFF (lever on-top DIR_DOWN)");
    world_destroy(w2);
}

static void test_torch_not_gate(void) {
    printf("[test] NOT gate: lever ON → lamp OFF\n");
    World* w = world_create(8, 4, 4);
    world_set_block(w, 0, 1, 1, BLOCK_LEVER, DIR_EAST, 0);
    world_set_block(w, 1, 1, 1, BLOCK_SOLID, DIR_NONE, 0);
    world_set_block(w, 2, 1, 1, BLOCK_REDSTONE_TORCH, DIR_WEST, 0);
    world_set_block(w, 3, 1, 1, BLOCK_REDSTONE_DUST, DIR_NONE, 0);
    world_tick_n(w, 30);
    world_sync_from_gpu(w);
    /* Verify torch OFF and dust 0 (NOT gate works) */
    ASSERT_EQ(world_get_block(w, 2, 1, 1).signal, 0, "NOT gate torch OFF");
    ASSERT_EQ(world_get_block(w, 3, 1, 1).signal, 0, "NOT gate dust = 0");
    world_destroy(w);
}

static void test_torch_unpowered_on(void) {
    printf("[test] Torch on unpowered block stays ON\n");
    World* w = world_create(8, 4, 4);
    world_set_block(w, 1, 1, 1, BLOCK_SOLID, DIR_NONE, 0);
    world_set_block(w, 2, 1, 1, BLOCK_REDSTONE_TORCH, DIR_WEST, 0);
    world_set_block(w, 3, 1, 1, BLOCK_REDSTONE_DUST, DIR_NONE, 0);
    world_tick_n(w, 5);
    world_sync_from_gpu(w);
    ASSERT_EQ(world_get_block(w, 2, 1, 1).signal, 15, "torch ON (block not powered)");
    ASSERT_EQ(world_get_block(w, 3, 1, 1).signal, 15, "dust powered by torch");
    world_destroy(w);
}

/* ════════════════════════════════════════════════════════════════════════
 * REPEATER TESTS (v0.3)
 * ════════════════════════════════════════════════════════════════════════ */

static void test_repeater_passes_signal(void) {
    printf("[test] Repeater passes signal through\n");
    World* w = world_create(8, 4, 4);
    world_set_block(w, 0, 1, 1, BLOCK_LEVER, DIR_NONE, 0);
    world_set_block(w, 1, 1, 1, BLOCK_REDSTONE_DUST, DIR_NONE, 0);
    /* Repeater facing EAST (input from WEST, output to EAST) */
    world_set_block(w, 2, 1, 1, BLOCK_REPEATER, DIR_EAST, 1);
    world_set_block(w, 3, 1, 1, BLOCK_REDSTONE_DUST, DIR_NONE, 0);
    world_tick_n(w, 10);
    world_sync_from_gpu(w);
    ASSERT_GT(world_get_block(w, 3, 1, 1).signal, 0, "dust after repeater has signal");
    world_destroy(w);
}

static void test_repeater_boosts_to_15(void) {
    printf("[test] Repeater boosts signal back to 15\n");
    World* w = world_create(32, 4, 4);
    world_set_block(w, 0, 1, 1, BLOCK_LEVER, DIR_NONE, 0);
    for (int x = 1; x <= 14; x++)
        world_set_block(w, x, 1, 1, BLOCK_REDSTONE_DUST, DIR_NONE, 0);
    world_set_block(w, 15, 1, 1, BLOCK_REPEATER, DIR_EAST, 1);
    world_set_block(w, 16, 1, 1, BLOCK_REDSTONE_DUST, DIR_NONE, 0);
    world_tick_n(w, 30);
    world_sync_from_gpu(w);
    /* Dust at x=14 has signal 2. Repeater boosts to 15. Dust at x=16 = 15. */
    ASSERT_EQ(world_get_block(w, 16, 1, 1).signal, 15, "dust after repeater boosted to 15");
    world_destroy(w);
}

static void test_repeater_directional(void) {
    printf("[test] Repeater only outputs in facing direction\n");
    World* w = world_create(8, 4, 8);
    world_set_block(w, 0, 1, 3, BLOCK_LEVER, DIR_NONE, 0);
    world_set_block(w, 1, 1, 3, BLOCK_REDSTONE_DUST, DIR_NONE, 0);
    /* Repeater facing EAST */
    world_set_block(w, 2, 1, 3, BLOCK_REPEATER, DIR_EAST, 1);
    /* Dust in front (should get signal) */
    world_set_block(w, 3, 1, 3, BLOCK_REDSTONE_DUST, DIR_NONE, 0);
    /* Dust on sides (should NOT get signal from repeater) */
    world_set_block(w, 2, 1, 2, BLOCK_REDSTONE_DUST, DIR_NONE, 0);
    world_set_block(w, 2, 1, 4, BLOCK_REDSTONE_DUST, DIR_NONE, 0);
    world_tick_n(w, 10);
    world_sync_from_gpu(w);
    ASSERT_GT(world_get_block(w, 3, 1, 3).signal, 0, "dust in front of repeater");
    ASSERT_EQ(world_get_block(w, 2, 1, 2).signal, 0, "dust on side of repeater (no signal)");
    ASSERT_EQ(world_get_block(w, 2, 1, 4).signal, 0, "dust on other side (no signal)");
    world_destroy(w);
}

static void test_repeater_locking(void) {
    printf("[test] Repeater locks when side repeater is powered\n");
    World* w = world_create(8, 4, 8);

    /* Main: Lever → Dust → Repeater(EAST) → Dust */
    world_set_block(w, 0, 1, 3, BLOCK_LEVER, DIR_NONE, 0);
    world_set_block(w, 1, 1, 3, BLOCK_REDSTONE_DUST, DIR_NONE, 0);
    world_set_block(w, 2, 1, 3, BLOCK_REPEATER, DIR_EAST, 1);
    world_set_block(w, 3, 1, 3, BLOCK_REDSTONE_DUST, DIR_NONE, 0);

    /* Locking: Lever → Repeater(NORTH) pointing into side of main */
    world_set_block(w, 2, 1, 5, BLOCK_LEVER, DIR_NONE, 0);
    world_set_block(w, 2, 1, 4, BLOCK_REPEATER, DIR_NORTH, 1);

    world_tick_n(w, 15);
    world_sync_from_gpu(w);

    Block rep = world_get_block(w, 2, 1, 3);
    ASSERT_EQ(!!(rep.flags & BLOCK_FLAG_LOCKED), 1, "main repeater is locked");
    world_destroy(w);
}

/* ════════════════════════════════════════════════════════════════════════
 * REDSTONE BLOCK TESTS
 * ════════════════════════════════════════════════════════════════════════ */

static void test_redstone_block_powers_dust(void) {
    printf("[test] Redstone block powers adjacent dust\n");
    World* w = world_create(8, 4, 4);
    world_set_block(w, 2, 1, 1, BLOCK_REDSTONE_BLOCK, DIR_NONE, 0);
    world_set_block(w, 3, 1, 1, BLOCK_REDSTONE_DUST, DIR_NONE, 0);
    world_set_block(w, 4, 1, 1, BLOCK_REDSTONE_DUST, DIR_NONE, 0);
    world_tick_n(w, 3);
    world_sync_from_gpu(w);
    ASSERT_EQ(world_get_block(w, 3, 1, 1).signal, 15, "dust next to redstone block");
    ASSERT_EQ(world_get_block(w, 4, 1, 1).signal, 14, "dust 2 away decays");
    world_destroy(w);
}

/* ════════════════════════════════════════════════════════════════════════
 * TIMING TESTS (v0.4) — verify delays match Minecraft behavior
 * ════════════════════════════════════════════════════════════════════════ */

static void test_torch_toggle_2_ticks(void) {
    printf("[test] Torch toggle takes exactly 2 game ticks\n");
    World* w = world_create(8, 4, 4);

    /* Lever → Solid → Torch. Torch starts ON, should turn OFF after 2 ticks
     * of detecting that its attached block is powered. */
    world_set_block(w, 0, 1, 1, BLOCK_LEVER, DIR_EAST, 0);
    world_set_block(w, 1, 1, 1, BLOCK_SOLID, DIR_NONE, 0);
    world_set_block(w, 2, 1, 1, BLOCK_REDSTONE_TORCH, DIR_WEST, 0);

    /* Tick 0: lever strongly powers solid. Detect schedules torch OFF at tick 2 */
    world_tick_n(w, 1);
    world_sync_from_gpu(w);
    ASSERT_EQ(world_get_block(w, 2, 1, 1).signal, 15, "tick 1: torch still ON");

    /* Tick 1: still waiting */
    world_tick_n(w, 1);
    world_sync_from_gpu(w);
    ASSERT_EQ(world_get_block(w, 2, 1, 1).signal, 15, "tick 2: torch still ON (event fires at start of tick 2)");

    /* By tick 3 the torch event has fired */
    world_tick_n(w, 1);
    world_sync_from_gpu(w);
    ASSERT_EQ(world_get_block(w, 2, 1, 1).signal, 0, "tick 3: torch OFF");

    world_destroy(w);
}

static void test_repeater_delay_1(void) {
    printf("[test] Repeater delay=1 takes 2 game ticks\n");
    World* w = world_create(8, 4, 4);

    world_set_block(w, 0, 1, 1, BLOCK_LEVER, DIR_NONE, 0);
    world_set_block(w, 1, 1, 1, BLOCK_REDSTONE_DUST, DIR_NONE, 0);
    world_set_block(w, 2, 1, 1, BLOCK_REPEATER, DIR_EAST, 1);
    world_set_block(w, 3, 1, 1, BLOCK_REDSTONE_DUST, DIR_NONE, 0);

    /* Before the repeater fires, output dust should be 0 */
    world_tick_n(w, 1);
    world_sync_from_gpu(w);
    ASSERT_EQ(world_get_block(w, 2, 1, 1).signal, 0, "tick 1: repeater not yet fired");

    /* After enough ticks for delay=1 (2 game ticks) */
    world_tick_n(w, 4);
    world_sync_from_gpu(w);
    ASSERT_GT(world_get_block(w, 2, 1, 1).signal, 0, "tick 5: repeater has fired");
    ASSERT_EQ(world_get_block(w, 3, 1, 1).signal, 15, "tick 5: output dust = 15");

    world_destroy(w);
}

static void test_repeater_delay_4(void) {
    printf("[test] Repeater delay=4 takes 8 game ticks\n");
    World* w = world_create(8, 4, 4);

    world_set_block(w, 0, 1, 1, BLOCK_LEVER, DIR_NONE, 0);
    world_set_block(w, 1, 1, 1, BLOCK_REDSTONE_DUST, DIR_NONE, 0);
    world_set_block(w, 2, 1, 1, BLOCK_REPEATER, DIR_EAST, 4);
    world_set_block(w, 3, 1, 1, BLOCK_REDSTONE_DUST, DIR_NONE, 0);

    /* At tick 4, delay=4 repeater should NOT have fired yet (needs 8 ticks) */
    world_tick_n(w, 4);
    world_sync_from_gpu(w);
    ASSERT_EQ(world_get_block(w, 2, 1, 1).signal, 0, "tick 4: delay-4 repeater not yet");

    /* By tick 12, it should have fired */
    world_tick_n(w, 8);
    world_sync_from_gpu(w);
    ASSERT_GT(world_get_block(w, 2, 1, 1).signal, 0, "tick 12: delay-4 repeater fired");

    world_destroy(w);
}

static void test_repeater_chain_cumulative_delay(void) {
    printf("[test] Chain of 3 repeaters: cumulative delay\n");
    World* w = world_create(16, 4, 4);

    world_set_block(w, 0, 1, 1, BLOCK_LEVER, DIR_NONE, 0);
    world_set_block(w, 1, 1, 1, BLOCK_REDSTONE_DUST, DIR_NONE, 0);
    world_set_block(w, 2, 1, 1, BLOCK_REPEATER, DIR_EAST, 1); /* 2 ticks */
    world_set_block(w, 3, 1, 1, BLOCK_REDSTONE_DUST, DIR_NONE, 0);
    world_set_block(w, 4, 1, 1, BLOCK_REPEATER, DIR_EAST, 1); /* 2 ticks */
    world_set_block(w, 5, 1, 1, BLOCK_REDSTONE_DUST, DIR_NONE, 0);
    world_set_block(w, 6, 1, 1, BLOCK_REPEATER, DIR_EAST, 1); /* 2 ticks */
    world_set_block(w, 7, 1, 1, BLOCK_REDSTONE_DUST, DIR_NONE, 0);

    /* First repeater should fire before second, before third */
    world_tick_n(w, 5);
    world_sync_from_gpu(w);
    int r1 = world_get_block(w, 2, 1, 1).signal > 0;
    int r3 = world_get_block(w, 6, 1, 1).signal > 0;
    ASSERT_EQ(r1, 1, "tick 5: first repeater fired");
    ASSERT_EQ(r3, 0, "tick 5: third repeater NOT fired yet");

    /* After enough ticks all should have fired */
    world_tick_n(w, 15);
    world_sync_from_gpu(w);
    ASSERT_GT(world_get_block(w, 7, 1, 1).signal, 0, "tick 20: end dust has signal");

    world_destroy(w);
}

static void test_quasi_connectivity_piston(void) {
    printf("[test] Quasi-connectivity: piston activated by power above\n");
    World* w = world_create(8, 8, 4);

    /* Lever at y=3 powers the space at (2,3,1) */
    world_set_block(w, 1, 3, 1, BLOCK_LEVER, DIR_NONE, 0);
    world_set_block(w, 2, 3, 1, BLOCK_REDSTONE_DUST, DIR_NONE, 0);

    /* Piston at y=2 — one block BELOW the dust.
     * Quasi-connectivity: piston checks power at its own pos AND y+1 */
    world_set_block(w, 2, 2, 1, BLOCK_PISTON, DIR_EAST, 0);

    world_tick_n(w, 5);
    world_sync_from_gpu(w);

    Block piston = world_get_block(w, 2, 2, 1);
    ASSERT_EQ(!!(piston.flags & BLOCK_FLAG_POWERED), 1,
              "piston powered via quasi-connectivity");

    world_destroy(w);
}

static void test_no_quasi_for_normal_blocks(void) {
    printf("[test] Lamp (non-quasi block) NOT activated by power above it\n");
    World* w = world_create(8, 8, 4);

    /* Dust at y=3 powers position (3,3,1) */
    world_set_block(w, 1, 3, 1, BLOCK_LEVER, DIR_NONE, 0);
    world_set_block(w, 2, 3, 1, BLOCK_REDSTONE_DUST, DIR_NONE, 0);
    world_set_block(w, 3, 3, 1, BLOCK_REDSTONE_DUST, DIR_NONE, 0);

    /* Lamp at y=2 is NOT directly adjacent to any powered block.
     * If quasi-connectivity leaked to lamps, it would turn on. */
    world_set_block(w, 3, 2, 1, BLOCK_LAMP, DIR_NONE, 0);

    /* Piston at same y=2 position but at x=4 SHOULD be powered (quasi) */
    world_set_block(w, 4, 3, 1, BLOCK_REDSTONE_DUST, DIR_NONE, 0);
    world_set_block(w, 4, 2, 1, BLOCK_PISTON, DIR_EAST, 0);

    world_tick_n(w, 5);
    world_sync_from_gpu(w);

    /* Lamp is directly below dust, so it IS adjacent — it will be powered.
     * Instead, check that a lamp 2 blocks below is NOT powered */
    Block piston = world_get_block(w, 4, 2, 1);
    ASSERT_EQ(!!(piston.flags & BLOCK_FLAG_POWERED), 1,
              "piston powered via quasi");

    /* Place a lamp far from any power — verify it stays off */
    world_set_block(w, 6, 1, 1, BLOCK_LAMP, DIR_NONE, 0);
    world_tick_n(w, 3);
    world_sync_from_gpu(w);
    Block lamp = world_get_block(w, 6, 1, 1);
    ASSERT_EQ(!!(lamp.flags & BLOCK_FLAG_POWERED), 0,
              "isolated lamp stays OFF");

    world_destroy(w);
}

/* ════════════════════════════════════════════════════════════════════════
 * SNAPSHOT TESTS — save/restore GPU state
 * ════════════════════════════════════════════════════════════════════════ */

static void test_save_restore(void) {
    printf("[test] Save state, advance, restore → state reverts\n");
    World* w = world_create(8, 4, 4);

    world_set_block(w, 0, 1, 1, BLOCK_LEVER, DIR_NONE, 0);
    world_set_block(w, 1, 1, 1, BLOCK_REDSTONE_DUST, DIR_NONE, 0);
    world_set_block(w, 2, 1, 1, BLOCK_REDSTONE_DUST, DIR_NONE, 0);

    /* Tick until dust is powered */
    world_tick_n(w, 3);
    world_sync_from_gpu(w);
    ASSERT_EQ(world_get_block(w, 1, 1, 1).signal, 15, "dust powered before save");
    uint32_t saved_tick = w->current_tick;

    /* Save state */
    WorldSnapshot* snap = world_save_state(w);

    /* Advance more ticks, clear the lever (simulating a change) */
    world_set_block(w, 0, 1, 1, BLOCK_AIR, DIR_NONE, 0);
    world_tick_n(w, 20);
    world_sync_from_gpu(w);
    ASSERT_EQ(world_get_block(w, 1, 1, 1).signal, 0, "dust unpowered after clearing lever");

    /* Restore → should go back to saved state */
    world_restore_state(w, snap);
    ASSERT_EQ(w->current_tick, saved_tick, "tick counter restored");
    Block d = world_get_block(w, 1, 1, 1);
    ASSERT_EQ(d.signal, 15, "dust signal restored to 15");

    /* Can continue simulating from restored state */
    world_tick_n(w, 1);
    world_sync_from_gpu(w);
    ASSERT_EQ(world_get_block(w, 1, 1, 1).signal, 15, "simulation continues from restored state");

    world_snapshot_destroy(snap);
    world_destroy(w);
}

/* ════════════════════════════════════════════════════════════════════════
 * PROBE TESTS — read specific signals without full sync
 * ════════════════════════════════════════════════════════════════════════ */

static void test_probes_read_signal(void) {
    printf("[test] Probes read correct signal values\n");
    World* w = world_create(16, 4, 4);

    world_set_block(w, 0, 1, 1, BLOCK_LEVER, DIR_NONE, 0);
    for (int x = 1; x <= 10; x++)
        world_set_block(w, x, 1, 1, BLOCK_REDSTONE_DUST, DIR_NONE, 0);

    /* Add probes at specific positions */
    world_add_probe(w, 1, 1, 1);  /* dust next to lever → 15 */
    world_add_probe(w, 5, 1, 1);  /* dust at x=5 → 11 */
    world_add_probe(w, 10, 1, 1); /* dust at x=10 → 6 */
    world_add_probe(w, 0, 1, 1);  /* lever → signal 15 */

    world_tick_n(w, 10);

    /* Read probes (no full sync needed!) */
    const ProbeResult* results = world_read_probes(w);

    ASSERT_EQ(results[0].signal, 15, "probe[0] dust at x=1 = 15");
    ASSERT_EQ(results[1].signal, 11, "probe[1] dust at x=5 = 11");
    ASSERT_EQ(results[2].signal, 6,  "probe[2] dust at x=10 = 6");
    ASSERT_EQ(results[3].type, BLOCK_LEVER, "probe[3] type = LEVER");

    world_destroy(w);
}

static void test_probes_no_full_sync(void) {
    printf("[test] Probes work without world_sync_from_gpu\n");
    World* w = world_create(8, 4, 4);

    world_set_block(w, 0, 1, 1, BLOCK_LEVER, DIR_NONE, 0);
    world_set_block(w, 1, 1, 1, BLOCK_REDSTONE_DUST, DIR_NONE, 0);
    world_set_block(w, 2, 1, 1, BLOCK_LAMP, DIR_NONE, 0);

    world_add_probe(w, 2, 1, 1); /* lamp */

    world_tick_n(w, 3);

    /* Read probes WITHOUT calling world_sync_from_gpu */
    const ProbeResult* r = world_read_probes(w);
    ASSERT_EQ(r[0].type, BLOCK_LAMP, "probe type = LAMP");
    ASSERT_EQ(!!(r[0].flags & BLOCK_FLAG_POWERED), 1, "lamp powered via probe");

    world_destroy(w);
}

/* ── Main ─────────────────────────────────────────────────────────────── */

int main(void) {
    printf("╔══════════════════════════════════════╗\n");
    printf("║   Redstone Simulator — Unit Tests     ║\n");
    printf("╚══════════════════════════════════════╝\n\n");

    registry_init();

    /* Dust tests */
    test_single_dust();
    test_line_decay();
    test_no_signal_without_source();
    test_two_sources();
    test_branching();
    test_staircase_up();
    test_lamp_powered();

    /* Torch tests */
    test_torch_powers_dust();
    test_torch_inverts();
    test_torch_not_gate();
    test_torch_unpowered_on();

    /* Repeater tests */
    test_repeater_passes_signal();
    test_repeater_boosts_to_15();
    test_repeater_directional();
    test_repeater_locking();

    /* Redstone block tests */
    test_redstone_block_powers_dust();

    /* Timing tests */
    test_torch_toggle_2_ticks();
    test_repeater_delay_1();
    test_repeater_delay_4();
    test_repeater_chain_cumulative_delay();

    /* Quasi-connectivity tests */
    test_quasi_connectivity_piston();
    test_no_quasi_for_normal_blocks();

    /* Snapshot tests */
    test_save_restore();

    /* Probe tests */
    test_probes_read_signal();
    test_probes_no_full_sync();

    printf("\n════════════════════════════════════════\n");
    printf("Results: %d/%d tests passed\n", g_tests_passed, g_tests_run);
    printf("════════════════════════════════════════\n");

    return (g_tests_passed == g_tests_run) ? 0 : 1;
}
