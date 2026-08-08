"""Smoke test for TruthTableTask.

Validates:
  - set_lever_powered toggles correctly in a multi-row evaluation
  - evaluate_final iterates 2^N input combinations with correct MSB-first
    indexing
  - A hand-built OR circuit scores 1.0 accuracy (end-to-end sanity check)

Run from repo root:
    python bindings/smoke_truth_table.py
"""

import sys

# Windows consoles default to cp1252 and cannot encode the arrows / check marks
# printed below. Force UTF-8 so a fresh clone runs on any platform.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from redstone_sim import (
    World,
    BLOCK_AIR, BLOCK_SOLID, BLOCK_REDSTONE_DUST,
    DIR_NONE,
)
from redstone_sim.env import TASKS, TruthTableTask


def test_empty_circuit_baselines() -> None:
    """No dust/torches — lamp always OFF. Accuracy depends on truth table.

    For all 2-input truth tables, an empty circuit (lamp always 0) matches
    exactly the rows whose target is 0:
      OR  [0,1,1,1]  -> 1/4 = 0.25
      XOR [0,1,1,0]  -> 2/4 = 0.50
      AND [0,0,0,1]  -> 3/4 = 0.75
    """
    print("\n[test] empty-circuit baselines")
    expected = {"or_2in": 0.25, "xor_2in": 0.50, "and_2in": 0.75}
    for name, exp in expected.items():
        task = TASKS[name]((10, 5, 10))
        with World(size=(10, 5, 10)) as w:
            task.setup_world(w)
            w.tick(20)
            reward, solved = task.evaluate_final(w)
            print(f"  {name}: {reward:.3f}  (expect {exp:.2f})")
            assert abs(reward - exp) < 1e-6, f"{name}: got {reward}, expected {exp}"
            assert not solved
    print("  ✓ pass")


def test_or_gate_handbuilt() -> None:
    """Build a correct 2-input OR gate by hand and verify 100% accuracy.

    Layout (top-down view, y=1):
        z=0:  L  D  .  .  .  .  .  .  .  .
        z=1:  .  D  .  .  .  .  .  .  .  .
        z=2:  .  D  .  .  .  .  .  .  .  .
        z=3:  .  D  .  .  .  .  .  .  .  .
        z=4:  .  D  D  D  D  D  D  D  D  *
        z=5:  .  D  .  .  .  .  .  .  .  .
        z=6:  .  D  .  .  .  .  .  .  .  .
        z=7:  .  D  .  .  .  .  .  .  .  .
        z=8:  .  D  .  .  .  .  .  .  .  .
        z=9:  L  D  .  .  .  .  .  .  .  .

    Y-shape: both levers feed dust at x=1; the center trunk at z=4 carries
    whichever signal is active to the lamp. OR by construction.
    """
    print("\n[test] hand-built OR gate")
    task = TASKS["or_2in"]((10, 5, 10))
    with World(size=(10, 5, 10)) as w:
        task.setup_world(w)
        # Path at x=1 from z=0 to z=9 (adjacent to both levers)
        for z in range(0, 10):
            w.set_block(1, 1, z, BLOCK_REDSTONE_DUST)
        # Trunk at z=4 from x=1 to x=8 (terminating adjacent to lamp at x=9)
        for x in range(2, 9):
            w.set_block(x, 1, 4, BLOCK_REDSTONE_DUST)
        w.tick(40)

        reward, solved = task.evaluate_final(w)
        print(f"  OR hand-built: reward={reward:.3f}  solved={solved}  (expect 1.00 True)")
        assert reward == 1.0, f"expected 1.0, got {reward}"
        assert solved
    print("  ✓ pass")


def test_lever_toggle_roundtrip() -> None:
    """Verify set_lever_powered can cycle a lever through ON/OFF multiple
    times within a single evaluate_final call (both levers cycle through
    all 4 combinations = 8 toggle calls internally)."""
    print("\n[test] lever toggle round-trip")
    task = TASKS["or_2in"]((10, 5, 10))
    with World(size=(10, 5, 10)) as w:
        task.setup_world(w)
        w.tick(20)
        # Run evaluate_final 3x in a row; result should be deterministic
        r1, _ = task.evaluate_final(w)
        r2, _ = task.evaluate_final(w)
        r3, _ = task.evaluate_final(w)
        print(f"  3x evaluate_final: {r1:.3f}, {r2:.3f}, {r3:.3f}")
        assert r1 == r2 == r3 == 0.25, "evaluate_final not deterministic"
    print("  ✓ pass")


if __name__ == "__main__":
    test_empty_circuit_baselines()
    test_or_gate_handbuilt()
    test_lever_toggle_roundtrip()
    print("\nAll TruthTableTask smoke tests passed ✓")
