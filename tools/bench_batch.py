"""Throughput benchmark: World (one world per launch) vs WorldBatch (N per launch).

The single-world simulator is launch-latency-bound — a 16^3 world is 4096
cells, roughly 0.1% of a 3060 Ti — so its cost per tick barely depends on
world size. Batching amortizes that launch over N worlds.

Run:
    python tools/bench_batch.py
    python tools/bench_batch.py --size 16 16 16 --ticks 40
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bindings"))

from redstone_sim import (  # noqa: E402
    World, WorldBatch,
    BLOCK_SOLID, BLOCK_LEVER, BLOCK_LAMP, BLOCK_REDSTONE_DUST,
    DIR_DOWN, DIR_NONE,
)


def scene(size):
    """Floor + lever + a 13-dust run + lamp: the shape of a routing task."""
    sx, sy, sz = size
    p = [(x, 0, z, BLOCK_SOLID, DIR_NONE) for x in range(sx) for z in range(sz)]
    p.append((0, 1, 0, BLOCK_LEVER, DIR_DOWN))
    run = min(13, sx - 2)
    for x in range(1, run + 1):
        p.append((x, 1, 0, BLOCK_REDSTONE_DUST, DIR_NONE))
    p.append((run + 1, 1, 0, BLOCK_LAMP, DIR_NONE))
    return p


def bench_single(size, ticks, n_worlds):
    """Time n_worlds sequential single-world simulations (steady state)."""
    p = scene(size)
    with World(size=size) as w:
        for (x, y, z, t, f) in p:
            w.set_block(x, y, z, t, facing=f)
        w.tick(5)                                   # warm up / settle
        t0 = time.perf_counter()
        for _ in range(n_worlds):
            w.tick(ticks)
        dt = time.perf_counter() - t0
    return dt


def bench_batch(size, ticks, n_worlds, until_stable=False):
    p = scene(size)
    with WorldBatch(n_worlds=n_worlds, size=size) as b:
        for i in range(n_worlds):
            for (x, y, z, t, f) in p:
                b.set_block(i, x, y, z, t, facing=f)
        b.tick(5)
        b.refresh_done()                            # drain the launch queue
        t0 = time.perf_counter()
        if until_stable:
            used = b.tick_until_stable(max_ticks=ticks, check_every=4)
        else:
            b.tick(ticks)
            used = ticks
        # The tick path is fully async (no per-kernel sync — that is the
        # point). Force completion before stopping the clock, or we would be
        # timing kernel *enqueue* rather than execution.
        b.refresh_done()
        dt = time.perf_counter() - t0
    return dt, used


def bench_truth_table(size, ticks, n_worlds):
    """The real workload: per circuit, restore -> set levers -> settle ->
    read lamp, for every truth-table row."""
    from redstone_sim.batch_eval import TruthTableSpec, evaluate_truth_tables

    sx, sy, sz = size
    p = scene(size)
    lever = (0, 1, 0)
    lamp = next((x, y, z) for (x, y, z, t, f) in p if t == BLOCK_LAMP)
    table = [0, 1]

    # Single-world reference (what TruthTableTask.evaluate_final does today).
    n_ref = 16
    with World(size=size) as w:
        for (x, y, z, t, f) in p:
            w.set_block(x, y, z, t, facing=f)
        w.tick(5)
        t0 = time.perf_counter()
        for _ in range(n_ref):
            snap = w.save()
            for row in range(len(table)):
                w.restore(snap)
                w.set_lever_powered(*lever, bool(row & 1))
                w.tick(ticks)
                w.is_powered(*lamp)
            snap.close()
        single = (time.perf_counter() - t0) / n_ref * 1e6

    with WorldBatch(n_worlds=n_worlds, size=size) as b:
        for i in range(n_worlds):
            for (x, y, z, t, f) in p:
                b.set_block(i, x, y, z, t, facing=f)
        specs = [TruthTableSpec(levers=[lever], lamp=lamp, table=table)
                 for _ in range(n_worlds)]
        evaluate_truth_tables(b, specs, ticks_per_row=ticks)   # warm up
        t0 = time.perf_counter()
        evaluate_truth_tables(b, specs, ticks_per_row=ticks)
        batched = (time.perf_counter() - t0) / n_worlds * 1e6

    print(f"truth-table eval ({len(table)} rows, {ticks} ticks/row): "
          f"World {single:.0f} us/circuit  ->  WorldBatch[{n_worlds}] "
          f"{batched:.2f} us/circuit ({single/batched:.0f}x)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, nargs=3, default=[16, 16, 16])
    ap.add_argument("--ticks", type=int, default=40,
                    help="ticks per circuit evaluation")
    ap.add_argument("--batches", type=int, nargs="*",
                    default=[1, 8, 32, 128, 512, 2048])
    args = ap.parse_args()

    size = tuple(args.size)
    ticks = args.ticks
    print(f"world {size[0]}x{size[1]}x{size[2]}, {ticks} ticks per circuit")
    print()

    ref_n = 64
    dt = bench_single(size, ticks, ref_n)
    single_us = dt / ref_n * 1e6
    print(f"World (one per launch):   {single_us:9.1f} us per circuit "
          f"({single_us/ticks:6.2f} us/tick, {ref_n/dt:8.1f} circuits/s)")
    print()
    print(f"{'batch':>6} {'us/circuit':>12} {'us/tick':>10} {'circuits/s':>12} "
          f"{'speedup':>9}")
    print("-" * 54)

    for n in args.batches:
        dt, _ = bench_batch(size, ticks, n)
        per = dt / n * 1e6
        print(f"{n:>6} {per:>12.2f} {per/ticks:>10.3f} {n/dt:>12.0f} "
              f"{single_us/per:>8.0f}x")

    # Early-stop on top of batching: how much of the fixed tick budget is
    # actually needed once convergence is detected.
    print()
    n = args.batches[-1]
    dt_fixed, _ = bench_batch(size, ticks, n)
    dt_stable, used = bench_batch(size, ticks, n, until_stable=True)
    print(f"batch={n}: fixed {ticks} ticks = {dt_fixed*1e3:.1f} ms  ->  "
          f"tick_until_stable used {used} ticks = {dt_stable*1e3:.1f} ms "
          f"({dt_fixed/dt_stable:.1f}x)")

    # End-to-end: the loop the RL/eval pipeline actually runs — snapshot
    # restore + lever set + settle + lamp read, per truth-table row.
    print()
    bench_truth_table(size, ticks, min(512, args.batches[-1]))

    # The RL observation path, for context: it can cost more than the sim.
    with World(size=size) as w:
        w.tick(1)
        t0 = time.perf_counter()
        for _ in range(200):
            w.get_all_blocks()
        single_obs = (time.perf_counter() - t0) / 200 * 1e6
    with WorldBatch(n_worlds=n, size=size) as b:
        b.tick(1)
        t0 = time.perf_counter()
        for _ in range(20):
            b.get_all_blocks()
        batch_obs = (time.perf_counter() - t0) / 20 / n * 1e6
    print(f"get_all_blocks: World {single_obs:.1f} us/world  ->  "
          f"WorldBatch {batch_obs:.2f} us/world ({single_obs/batch_obs:.0f}x)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
