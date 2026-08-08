"""Comparator semantics: compare / subtract modes, side inputs, and the
level-preserving output that distinguishes a comparator from a repeater.

Reference behaviour (Minecraft ComparatorBlock):
    compare  : out = rear if rear >= max(sides) else 0
    subtract : out = max(0, rear - max(sides))

Sides only accept "alternate inputs" (real signal sources); a plain solid block
carrying strong power feeds the rear but never a side.

NOT covered here, because the simulator does not model it: container fullness.
A comparator behind a chest reads 0.
"""
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from redstone_sim import (
    World, WorldBatch,
    BLOCK_SOLID, BLOCK_LEVER, BLOCK_REDSTONE_DUST, BLOCK_COMPARATOR,
    DIR_DOWN, DIR_SOUTH,
)

SIZE = (16, 4, 16)
TICKS = 80

_passed = 0
_failed = 0


def check(name, got, want):
    global _passed, _failed
    if got == want:
        _passed += 1
        print(f"  [ok]   {name}: {got}")
    else:
        _failed += 1
        print(f"  [FAIL] {name}: got {got}, want {want}")


def place(setter, mode_subtract=0, side_chain=None, rear_dusts=3):
    """Lever -> rear_dusts dust -> comparator -> dust.

    `setter(x, y, z, type, facing=..., state=...)` adapts World vs WorldBatch.
    The rear level is 15 - (rear_dusts - 1); with the default 3 dusts it is 13.
    A side chain of L dusts occupies x=6..6+L-1 with its lever at 6+L, so the
    cell touching the comparator's east face ends up at 15 - (L - 1).
    """
    for x in range(SIZE[0]):
        for z in range(SIZE[2]):
            setter(x, 0, z, BLOCK_SOLID)

    setter(5, 1, 1, BLOCK_LEVER, facing=DIR_DOWN)
    for i in range(rear_dusts):
        setter(5, 1, 2 + i, BLOCK_REDSTONE_DUST)

    comp_z = 2 + rear_dusts
    setter(5, 1, comp_z, BLOCK_COMPARATOR, facing=DIR_SOUTH,
           state=(1 if mode_subtract else 0))
    setter(5, 1, comp_z + 1, BLOCK_REDSTONE_DUST)

    if side_chain:
        for i in range(side_chain):
            setter(6 + i, 1, comp_z, BLOCK_REDSTONE_DUST)
        setter(6 + side_chain, 1, comp_z, BLOCK_LEVER, facing=DIR_DOWN)
    return comp_z


def run_world(**kw):
    w = World(size=SIZE)
    try:
        comp_z = place(w.set_block, **kw)
        w.tick(TICKS)
        w.sync()
        return (w.get_signal(5, 1, comp_z - 1),      # rear
                w.get_signal(6, 1, comp_z),          # side
                w.get_signal(5, 1, comp_z),          # comparator
                w.get_signal(5, 1, comp_z + 1))      # output dust
    finally:
        w.destroy()


print("\n[comparator] compare mode")
rear, side, out, _ = run_world()
check("rear reaches the comparator at 13", rear, 13)
check("no side input -> output equals rear", out, 13)

_, side, out, _ = run_world(side_chain=4)
check("side 12 <= rear 13 -> passes rear through", (side, out), (12, 13))

_, side, out, _ = run_world(side_chain=1)
check("side 15 > rear 13 -> blocked", (side, out), (15, 0))

_, side, out, _ = run_world(side_chain=3)
check("side 13 == rear 13 -> passes (>= is inclusive)", (side, out), (13, 13))

print("\n[comparator] subtract mode")
_, _, out, _ = run_world(mode_subtract=1)
check("no side -> rear unchanged", out, 13)

_, side, out, _ = run_world(mode_subtract=1, side_chain=4)
check("13 - 12 = 1", (side, out), (12, 1))

_, side, out, _ = run_world(mode_subtract=1, side_chain=1)
check("13 - 15 clamps to 0", (side, out), (15, 0))

print("\n[comparator] level preservation")
# A repeater would re-emit 15 here; a comparator must carry the rear level.
_, _, out, dust = run_world(rear_dusts=6)
check("weaker rear is preserved, not amplified", out, 10)
check("output dust takes the comparator level without decay", dust, 10)

print("\n[comparator] batch equivalence")
b = WorldBatch(n_worlds=4, size=SIZE)
cases = [dict(), dict(side_chain=1), dict(mode_subtract=1, side_chain=4),
         dict(rear_dusts=6)]
comp_zs = []
for i, kw in enumerate(cases):
    def setter(x, y, z, t, facing=0, state=0, _i=i):
        b.set_block(_i, x, y, z, t, facing=facing, state=state)
    comp_zs.append(place(setter, **kw))
b.tick(TICKS)
b.sync()

for i, (kw, cz) in enumerate(zip(cases, comp_zs)):
    batched = b.get_signal(i, 5, 1, cz)
    single = run_world(**kw)[2]
    check(f"instance {i} matches single-world", batched, single)

print("\n" + "=" * 60)
print(f"Results: {_passed}/{_passed + _failed} tests passed")
sys.exit(1 if _failed else 0)
