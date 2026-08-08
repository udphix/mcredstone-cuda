"""Load each redstone_xor_*.schem, find all levers + lamp, run all 4
input combos, and check XOR truth table: [0, 1, 1, 0]."""
import sys, os, glob
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bindings"))
from redstone_sim import World, BLOCK_LEVER, BLOCK_LAMP

XOR_TT = [0, 1, 1, 0]  # (A=0,B=0), (0,1), (1,0), (1,1)


def find_levers_lamp(w):
    levers, lamp = [], None
    for y in range(w.size[1]):
        for z in range(w.size[2]):
            for x in range(w.size[0]):
                b = w.get_block(x, y, z)
                if b.type == BLOCK_LEVER:
                    levers.append((x, y, z))
                elif b.type == BLOCK_LAMP:
                    lamp = (x, y, z)
    return levers, lamp


def test_schem(path: str) -> tuple[bool, str]:
    import ctypes
    WORLD_SIZE = (16, 16, 16)
    with World(size=WORLD_SIZE) as w:
        n = w.import_schem(path)
        if n <= 0:
            return False, f"import failed ({n})"
        w.push()
        levers, lamp = find_levers_lamp(w)
        if len(levers) != 2 or lamp is None:
            return False, f"expected 2 levers + 1 lamp, got {len(levers)} + {lamp!r}"

        snap = w.save()
        got = []
        for row in range(4):
            w.restore(snap)
            for i, (lx, ly, lz) in enumerate(levers):
                bit = (row >> (1 - i)) & 1
                w.set_lever_powered(lx, ly, lz, bool(bit))
            w.tick(40)
            got.append(1 if w.is_powered(*lamp) else 0)
        snap.close()
        ok = (got == XOR_TT)
        return ok, f"got={got} exp={XOR_TT}"


def main():
    schems = sorted(glob.glob(str(ROOT / "schematics" / "golden" / "xor" / "redstone_xor_*.schem")))
    print(f"=== XOR functional test: {len(schems)} schems ===")
    passed = 0
    for s in schems:
        name = Path(s).name
        ok, msg = test_schem(s)
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {name}: {msg}")
        if ok: passed += 1
    print(f"\n=== {passed}/{len(schems)} XOR functional passed ===")
    return 0 if passed == len(schems) else 1


if __name__ == "__main__":
    sys.exit(main())
