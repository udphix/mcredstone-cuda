"""Smoke test for the Python bindings.

Validates: DLL loading, Block struct layout, World lifecycle, set/get_block,
ticking, snapshot save/restore, schem import.

Run from repo root:
    python bindings/smoke_test.py
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
    BLOCK_AIR, BLOCK_LEVER, BLOCK_LAMP, BLOCK_REDSTONE_DUST, BLOCK_SOLID,
    DIR_DOWN, DIR_NONE,
    FLAG_POWERED,
)


def test_lever_to_lamp() -> None:
    """Simplest circuit: lever → dust line → lamp."""
    print("\n[test] lever → dust → lamp")
    with World(size=(8, 4, 8)) as w:
        w.set_block(0, 0, 0, BLOCK_SOLID)
        w.set_block(0, 1, 0, BLOCK_LEVER, facing=DIR_DOWN)

        for z in range(1, 7):
            w.set_block(0, 0, z, BLOCK_SOLID)
            w.set_block(0, 1, z, BLOCK_REDSTONE_DUST)

        w.set_block(0, 0, 7, BLOCK_SOLID)
        w.set_block(0, 1, 7, BLOCK_LAMP)

        w.tick(10)

        lamp_on = w.is_powered(0, 1, 7)
        mid_signal = w.get_signal(0, 1, 3, auto_sync=False)
        print(f"  lamp powered: {lamp_on}, mid-dust signal: {mid_signal}")
        assert lamp_on, "Lamp should be lit when lever is ON"
        assert 1 <= mid_signal <= 15, f"mid signal out of range: {mid_signal}"
    print("  ✓ pass")


def test_snapshot_restore() -> None:
    """Save state, mutate, restore — state should be back."""
    print("\n[test] snapshot save/restore")
    with World(size=(4, 4, 4)) as w:
        w.set_block(0, 0, 0, BLOCK_SOLID)
        w.set_block(0, 1, 0, BLOCK_LEVER, facing=DIR_DOWN)
        w.tick(5)

        snap = w.save()
        original = w.get_block(0, 1, 0)

        w.clear()
        w.tick(5)
        w.sync()
        assert w.get_block(0, 1, 0).type == BLOCK_AIR

        w.restore(snap)
        restored = w.get_block(0, 1, 0)
        assert restored.type == BLOCK_LEVER
        assert restored.flags == original.flags
        snap.close()
    print("  ✓ pass")


def test_xor_schem() -> None:
    """Import the MC XOR schematic and verify case (1,1) → lamp OFF."""
    print("\n[test] import MC XOR schem")
    schem = Path(__file__).parent.parent / "schematics" / "redstone.schem"
    if not schem.exists():
        print(f"  (skipped — {schem} not found)")
        return
    with World(size=(8, 8, 16)) as w:
        placed = w.import_schem(str(schem))
        assert placed > 0, "schem import placed zero blocks"

        # Remove the floor (matches main.cu behavior)
        for z in range(15):
            for x in range(8):
                w.set_block(x, 0, z, BLOCK_AIR)

        w.tick(500)
        lamp_on = w.is_powered(2, 1, 1)
        print(f"  lamp (2,1,1) powered: {lamp_on} (XOR(1,1)=0 → expect OFF)")
        assert not lamp_on, "XOR(1,1)=0, lamp should be OFF"
    print("  ✓ pass")


if __name__ == "__main__":
    test_lever_to_lamp()
    test_snapshot_restore()
    test_xor_schem()
    print("\nAll smoke tests passed ✓")
