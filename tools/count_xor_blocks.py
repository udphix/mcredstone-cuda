"""Count payload blocks (non-floor, non-lever, non-lamp, non-anchor-stone)
across all golden XOR schems, to find the minimum-K XOR."""
import sys, glob
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bindings"))

from redstone_sim import World

results = []
for path in sorted(glob.glob(str(ROOT / "schematics/golden/xor/redstone_xor_*.schem"))):
    name = Path(path).name
    with World(size=(24, 12, 24)) as w:
        w.import_schem(path)
        w.push()  # import only touches host; push so get_all_blocks sees it
        arr = w.get_all_blocks(auto_sync=False)
        counts = {}
        for t in arr["type"].flatten():
            counts[int(t)] = counts.get(int(t), 0) + 1
        counts.pop(0, None)  # air
        total = sum(counts.values())
        dims = f"{arr.shape[2]}x{arr.shape[0]}x{arr.shape[1]}"
        # Estimate scaffold: all y=0 stones + levers + lamp.
        # Everything else is payload.
        nonscaff = total - counts.get(10, 0) - counts.get(54, 0)
        results.append((name, dims, total, counts))

results.sort(key=lambda r: r[2])
print("name                    total_blocks   by_type")
for name, dims, total, counts in results:
    type_names = {1: "SOLID", 10: "LEVER", 20: "DUST", 30: "TORCH",
                  31: "REP", 13: "RBLOCK", 54: "LAMP"}
    parts = ",  ".join(f"{type_names.get(t, t)}={n}" for t, n in sorted(counts.items()))
    print(f"  {name:30s}  {total:3d}   {parts}")
