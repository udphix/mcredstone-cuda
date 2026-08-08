"""WorldBatch <-> World equivalence tests.

The batched launch shape only earns its keep if it is *exactly* the same
simulator. These tests build identical random circuits in both paths and
assert bit-identical block state (all 8 bytes: type, signal, facing, flags,
state, tick_scheduled) — not just "the lamp agrees".

Run:
    python tests/test_batch_equivalence.py
    python tests/test_batch_equivalence.py --n 256 --ticks 60   # heavier

Exit code 0 = all passed.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bindings"))

from redstone_sim import (  # noqa: E402
    World, WorldBatch,
    BLOCK_AIR, BLOCK_SOLID, BLOCK_LEVER, BLOCK_LAMP,
    BLOCK_REDSTONE_DUST, BLOCK_REDSTONE_TORCH, BLOCK_REPEATER,
    BLOCK_REDSTONE_BLOCK,
    DIR_DOWN, DIR_UP, DIR_NORTH, DIR_SOUTH, DIR_WEST, DIR_EAST, DIR_NONE,
    FLAG_POWERED,
)

SIZE = (16, 8, 16)
HORIZ = (DIR_NORTH, DIR_SOUTH, DIR_WEST, DIR_EAST)

_passed = 0
_failed = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global _passed, _failed
    if ok:
        _passed += 1
        print(f"  [ok]   {name}")
    else:
        _failed += 1
        print(f"  [FAIL] {name}{(' — ' + detail) if detail else ''}")


# ── Random circuit generation ────────────────────────────────────────────

def random_circuit(rng: random.Random, size=SIZE):
    """A random-but-plausible scene: floor, levers, dust runs on supports,
    torches on solid walls, repeaters, a lamp. Returns a placement list of
    (x, y, z, type, facing, state) plus the lever positions.

    Dust always gets a support block below it so the circuits exercise real
    propagation paths rather than degenerate floating wire.
    """
    sx, sy, sz = size
    placements = []
    occupied = set()

    def put(x, y, z, t, facing=DIR_NONE, state=0):
        if not (0 <= x < sx and 0 <= y < sy and 0 <= z < sz):
            return False
        if (x, y, z) in occupied:
            return False
        occupied.add((x, y, z))
        placements.append((x, y, z, t, facing, state))
        return True

    # Floor.
    for x in range(sx):
        for z in range(sz):
            put(x, 0, z, BLOCK_SOLID)

    levers = []
    n_levers = rng.randint(1, 3)
    for _ in range(n_levers):
        lx, lz = rng.randrange(sx), rng.randrange(sz)
        if put(lx, 1, lz, BLOCK_LEVER, DIR_DOWN):
            levers.append((lx, 1, lz))

    # Dust runs: random walks on top of the floor (and on stacked supports).
    for _ in range(rng.randint(2, 6)):
        x, z = rng.randrange(sx), rng.randrange(sz)
        y = 1
        for _ in range(rng.randint(3, 14)):
            put(x, y, z, BLOCK_REDSTONE_DUST)
            if rng.random() < 0.15 and y + 1 < sy - 1:
                # staircase step: solid + dust on top
                nx, nz = x, z
                if rng.random() < 0.5:
                    nx += rng.choice((-1, 1))
                else:
                    nz += rng.choice((-1, 1))
                if put(nx, y, nz, BLOCK_SOLID):
                    y += 1
                    x, z = nx, nz
                    continue
            if rng.random() < 0.5:
                x += rng.choice((-1, 1))
            else:
                z += rng.choice((-1, 1))
            x = max(0, min(sx - 1, x))
            z = max(0, min(sz - 1, z))

    # Torch towers: a solid pillar with a wall torch on it.
    for _ in range(rng.randint(0, 3)):
        x, z = rng.randrange(sx), rng.randrange(sz)
        h = rng.randint(1, 3)
        for y in range(1, h + 1):
            put(x, y, z, BLOCK_SOLID)
        d = rng.choice(HORIZ)
        dx = {DIR_WEST: -1, DIR_EAST: 1}.get(d, 0)
        dz = {DIR_NORTH: -1, DIR_SOUTH: 1}.get(d, 0)
        # facing = direction of the attached block (our convention)
        put(x - dx, h, z - dz, BLOCK_REDSTONE_TORCH, d)

    # Repeaters and a redstone block, for tile-tick scheduling coverage.
    for _ in range(rng.randint(0, 3)):
        put(rng.randrange(sx), 1, rng.randrange(sz), BLOCK_REPEATER,
            rng.choice(HORIZ), rng.randint(1, 4))
    if rng.random() < 0.4:
        put(rng.randrange(sx), 1, rng.randrange(sz), BLOCK_REDSTONE_BLOCK)

    # Lamps.
    for _ in range(rng.randint(1, 3)):
        put(rng.randrange(sx), 1, rng.randrange(sz), BLOCK_LAMP)

    return placements, levers


def build_world(placements, size=SIZE) -> World:
    w = World(size=size)
    for (x, y, z, t, f, s) in placements:
        w.set_block(x, y, z, t, facing=f, state=s)
    return w


def build_batch(circuits, size=SIZE) -> WorldBatch:
    b = WorldBatch(n_worlds=len(circuits), size=size)
    for i, (placements, _levers) in enumerate(circuits):
        for (x, y, z, t, f, s) in placements:
            b.set_block(i, x, y, z, t, facing=f, state=s)
    return b


# ── Tests ────────────────────────────────────────────────────────────────

def test_random_equivalence(n_worlds: int, ticks: int, seed: int) -> None:
    """Same circuits, single vs batched: bit-identical after `ticks` ticks."""
    rng = random.Random(seed)
    circuits = [random_circuit(rng) for _ in range(n_worlds)]

    with build_batch(circuits) as batch:
        batch.tick(ticks)
        batch_state = batch.get_all_blocks()

        mismatches = []
        for i, (placements, _levers) in enumerate(circuits):
            with build_world(placements) as w:
                w.tick(ticks)
                single = w.get_all_blocks()
            if not np.array_equal(single, batch_state[i]):
                diff = np.argwhere(single != batch_state[i])
                mismatches.append((i, len(diff), diff[:3].tolist()))

        check(
            f"random equivalence ({n_worlds} worlds x {ticks} ticks, full state)",
            not mismatches,
            f"{len(mismatches)} world(s) diverged: {mismatches[:3]}",
        )


def test_lever_toggle_equivalence(n_worlds: int, ticks: int, seed: int) -> None:
    """Device-side set_levers must match host-side set_lever_powered."""
    rng = random.Random(seed + 1)
    circuits = [random_circuit(rng) for _ in range(n_worlds)]

    with build_batch(circuits) as batch:
        batch.tick(ticks)
        entries = []
        for i, (_p, levers) in enumerate(circuits):
            for k, (lx, ly, lz) in enumerate(levers):
                entries.append((i, lx, ly, lz, k % 2 == 0))
        batch.set_levers(entries)
        batch.tick(ticks)
        batch_state = batch.get_all_blocks()

        mismatches = []
        for i, (placements, levers) in enumerate(circuits):
            with build_world(placements) as w:
                w.tick(ticks)
                # sync() is REQUIRED here, not cosmetic: set_lever_powered
                # edits the host mirror and dirties the world, so the next
                # tick() re-uploads that mirror wholesale. Without pulling
                # GPU state back first, the mirror is still the initial
                # scene and the upload silently discards everything the
                # previous ticks computed. WorldBatch.set_levers writes
                # device memory directly and has no such failure mode, so
                # skipping this would compare "reset + N ticks" against
                # "continue + N ticks".
                w.sync()
                for k, (lx, ly, lz) in enumerate(levers):
                    w.set_lever_powered(lx, ly, lz, k % 2 == 0)
                w.tick(ticks)
                single = w.get_all_blocks()
            if not np.array_equal(single, batch_state[i]):
                mismatches.append(i)

        check(
            f"lever-toggle equivalence ({n_worlds} worlds)",
            not mismatches,
            f"{len(mismatches)} world(s) diverged: {mismatches[:5]}",
        )


def test_instance_isolation() -> None:
    """Instances are contiguous slabs — instance i's last cell is adjacent in
    memory to instance i+1's first cell. Signal must not cross that seam."""
    sx, sy, sz = SIZE
    with WorldBatch(n_worlds=4, size=SIZE) as b:
        # Instance 1: fully powered corner at the far end of its slab.
        for x in range(sx):
            for z in range(sz):
                b.set_block(1, x, sy - 1, z, BLOCK_SOLID)
        b.set_block(1, sx - 1, sy - 1, sz - 1, BLOCK_REDSTONE_BLOCK)
        b.set_block(1, sx - 2, sy - 1, sz - 1, BLOCK_REDSTONE_DUST)
        # Instance 2: a dust carpet on a floor, expected to stay dark.
        for x in range(sx):
            for z in range(sz):
                b.set_block(2, x, 0, z, BLOCK_SOLID)
                b.set_block(2, x, 1, z, BLOCK_REDSTONE_DUST)
        b.tick(30)
        state = b.get_all_blocks()

        leaked = int((state[2]["signal"] > 0).sum())
        check("instance isolation (no signal across the slab seam)",
              leaked == 0, f"{leaked} cells lit in the neighbouring instance")

        lit = int((state[1]["signal"] > 0).sum())
        check("instance isolation (source instance still works)", lit > 0)

        empty0 = int((state[0]["type"] != BLOCK_AIR).sum())
        check("untouched instances stay empty", empty0 == 0,
              f"{empty0} non-air blocks appeared")


def test_snapshot_restore(seed: int) -> None:
    rng = random.Random(seed + 2)
    circuits = [random_circuit(rng) for _ in range(8)]
    # Toggling every lever guarantees a state change — many random circuits
    # have already settled by tick 20, so simply ticking on would prove
    # nothing about restore().
    toggles = [(i, lx, ly, lz, False)
               for i, (_p, levers) in enumerate(circuits)
               for (lx, ly, lz) in levers]

    with build_batch(circuits) as b:
        b.tick(20)
        before = b.get_all_blocks()
        snap = b.save()

        b.set_levers(toggles)
        b.tick(15)
        after = b.get_all_blocks()
        check("snapshot: state actually diverged before restore",
              not np.array_equal(before, after))

        b.restore(snap)
        restored = b.get_all_blocks()
        check("snapshot: restore() reverts the whole batch",
              np.array_equal(before, restored))

        # Per-instance restore leaves the others running.
        b.set_levers(toggles)
        b.tick(15)
        advanced = b.get_all_blocks()
        b.restore_instance(snap, 3)
        mixed = b.get_all_blocks()
        check("snapshot: restore_instance reverts only that instance",
              np.array_equal(mixed[3], before[3]) and
              np.array_equal(mixed[4], advanced[4]))
        snap.close()


def routing_circuit(rng: random.Random, size=SIZE):
    """A purely combinational lever -> dust -> lamp route. No torches, so it
    is guaranteed to settle — the case `tick_until_stable` must stop early on."""
    sx, sy, sz = size
    placements = [(x, 0, z, BLOCK_SOLID, DIR_NONE, 0)
                  for x in range(sx) for z in range(sz)]
    z = rng.randrange(sz)
    placements.append((0, 1, z, BLOCK_LEVER, DIR_DOWN, 0))
    run = rng.randint(4, 12)
    for x in range(1, run):
        placements.append((x, 1, z, BLOCK_REDSTONE_DUST, DIR_NONE, 0))
    placements.append((run, 1, z, BLOCK_LAMP, DIR_NONE, 0))
    return placements, [(0, 1, z)]


def torch_clock_circuit(size=SIZE):
    """A wall-torch clock: the torch's output climbs a staircase and returns
    as dust on top of its own attachment block, which strong-powers it down.
    Never reaches a fixed point — `done` must never be claimed for it."""
    sx, _sy, sz = size
    placements = [(x, 0, z, BLOCK_SOLID, DIR_NONE, 0)
                  for x in range(sx) for z in range(sz)]
    placements += [
        (1, 1, 1, BLOCK_SOLID,          DIR_NONE, 0),   # attachment block
        (2, 1, 1, BLOCK_REDSTONE_TORCH, DIR_WEST, 0),   # wall torch on it
        (2, 1, 2, BLOCK_REDSTONE_DUST,  DIR_NONE, 0),   # lit from the side
        (1, 1, 2, BLOCK_SOLID,          DIR_NONE, 0),   # staircase step
        (1, 2, 2, BLOCK_REDSTONE_DUST,  DIR_NONE, 0),
        (1, 2, 1, BLOCK_REDSTONE_DUST,  DIR_NONE, 0),   # strong-powers it down
    ]
    return placements, []


def test_tick_until_stable(seed: int) -> None:
    """Converging early must land on the same state as ticking a long time,
    and `done` must not be claimed while tile ticks are still pending."""
    rng = random.Random(seed + 3)

    # ── Settling circuits: must stop well before max_ticks ──────────
    routes = [routing_circuit(rng) for _ in range(32)]
    with build_batch(routes) as b1, build_batch(routes) as b2:
        used = b1.tick_until_stable(max_ticks=200, check_every=4)
        early = b1.get_all_blocks()
        b1.refresh_done()
        done = b1.done_count()
        b2.tick(200)
        late = b2.get_all_blocks()

        check(f"tick_until_stable reaches the same fixed point as 200 ticks "
              f"({used} used, {done}/{b1.n_worlds} converged)",
              np.array_equal(early, late))
        check(f"tick_until_stable stops early on settling circuits ({used} ticks)",
              used < 200 and done == b1.n_worlds)

    # ── Mixed batch: a clock must keep everything else honest ───────
    mixed = [routing_circuit(rng) for _ in range(15)] + [torch_clock_circuit()]
    with build_batch(mixed) as b:
        used = b.tick_until_stable(max_ticks=120, check_every=4)
        b.refresh_done()
        check("oscillating instance never reports converged",
              not b.instance_done(15) and b.done_count() == 15,
              f"done_count={b.done_count()}, clock done={b.instance_done(15)}")
        check("a single non-converging instance uses the full budget",
              used == 120, f"used {used}")

        # And the clock must still be *running*, not frozen at a fixed point.
        # Sample consecutive ticks rather than comparing against a single
        # later snapshot: this torch clock has a period of 4, so any probe at
        # a multiple of the period lands on the same phase and would look
        # frozen even though it is running.
        frames = []
        for _ in range(5):
            frames.append(b.get_all_blocks()[15]["signal"].copy())
            b.tick(1)
        check("oscillating instance keeps changing state",
              any(not np.array_equal(frames[0], f) for f in frames[1:]),
              f"all {len(frames)} consecutive frames identical")

    # ── `done` must never be latched while a tile tick is pending ───
    rand = [random_circuit(rng) for _ in range(32)]
    with build_batch(rand) as b:
        b.tick_until_stable(max_ticks=200, check_every=4)
        state = b.get_all_blocks()
        b.refresh_done()
        sched = (state["flags"] & 0x08) != 0   # FLAG_SCHEDULED
        stuck = [i for i in range(b.n_worlds)
                 if b.instance_done(i) and sched[i].any()]
        check("converged instances hold no pending tile ticks",
              not stuck, f"instances {stuck[:5]} are done but still scheduled")


def test_probes(seed: int) -> None:
    """Batched probes must agree with host-mirror reads."""
    rng = random.Random(seed + 4)
    circuits = [random_circuit(rng) for _ in range(16)]
    with build_batch(circuits) as b:
        b.tick(30)
        state = b.get_all_blocks()
        sx, sy, sz = SIZE
        entries = [(i, rng.randrange(sx), rng.randrange(2), rng.randrange(sz))
                   for i in range(16) for _ in range(4)]
        b.set_probes(entries)
        results = b.read_probes()

        bad = []
        for (inst, x, y, z), r in zip(entries, results):
            cell = state[inst, y, z, x]
            if (r.type != cell["type"] or r.signal != cell["signal"]
                    or r.flags != cell["flags"]):
                bad.append((inst, x, y, z))
        check("probes match host-mirror reads", not bad,
              f"{len(bad)} mismatched probe(s): {bad[:3]}")


def test_truth_table_eval(seed: int) -> None:
    """`batch_eval.evaluate_truth_tables` must reproduce the single-world
    evaluation loop (TruthTableTask.evaluate_final) exactly."""
    from redstone_sim.batch_eval import TruthTableSpec, evaluate_truth_tables

    rng = random.Random(seed + 5)
    sx, sy, sz = SIZE
    circuits, specs = [], []

    for i in range(24):
        z = rng.randrange(sz)
        run = rng.randint(3, 11)
        placements = [(x, 0, zz, BLOCK_SOLID, DIR_NONE, 0)
                      for x in range(sx) for zz in range(sz)]
        placements.append((0, 1, z, BLOCK_LEVER, DIR_DOWN, 0))
        # Every third circuit gets a deliberate gap, so the batch contains
        # both solved and unsolved cases.
        gap = (i % 3 == 0)
        for x in range(1, run):
            if gap and x == run // 2:
                continue
            placements.append((x, 1, z, BLOCK_REDSTONE_DUST, DIR_NONE, 0))
        placements.append((run, 1, z, BLOCK_LAMP, DIR_NONE, 0))
        circuits.append((placements, [(0, 1, z)]))
        specs.append(TruthTableSpec(levers=[(0, 1, z)], lamp=(run, 1, z),
                                    table=[0, 1]))

    # Reference: the single-world loop, one World per circuit.
    ref = []
    for (placements, _levers), spec in zip(circuits, specs):
        with build_world(placements) as w:
            snap = w.save()
            m = 0
            for row, want in enumerate(spec.table):
                w.restore(snap)
                lx, ly, lz = spec.levers[0]
                w.set_lever_powered(lx, ly, lz, bool(row & 1))
                w.tick(40)
                if int(w.is_powered(*spec.lamp)) == want:
                    m += 1
            snap.close()
            ref.append(m / len(spec.table))
    ref = np.array(ref, dtype=np.float32)

    with build_batch(circuits) as b:
        rates, solved = evaluate_truth_tables(b, specs, ticks_per_row=40)

    check("batched truth-table eval matches the single-world loop",
          np.array_equal(rates, ref),
          f"batch={rates.tolist()[:6]} single={ref.tolist()[:6]}")
    check("batched eval separates solved from broken circuits",
          0 < int(solved.sum()) < len(specs),
          f"{int(solved.sum())}/{len(specs)} solved")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64, help="worlds in the batch")
    ap.add_argument("--ticks", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("WorldBatch equivalence tests")
    print("=" * 60)
    test_random_equivalence(args.n, args.ticks, args.seed)
    test_lever_toggle_equivalence(min(args.n, 32), args.ticks, args.seed)
    test_instance_isolation()
    test_snapshot_restore(args.seed)
    test_tick_until_stable(args.seed)
    test_probes(args.seed)
    test_truth_table_eval(args.seed)
    print("=" * 60)
    print(f"Results: {_passed}/{_passed + _failed} tests passed")
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
