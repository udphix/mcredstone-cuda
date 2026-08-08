#!/usr/bin/env python3
"""
Compare simulator output against schem ground truth.

  python tools/compare_schem.py <schem> [ticks]

Runs  build/Release/dump_sim.exe <schem> <ticks>, then parses the schem
ground truth (power=, lit=, powered=) and compares block-by-block.
"""
import gzip, os, struct, subprocess, sys

# Windows consoles default to cp1252 and cannot encode the arrows / check marks
# printed below. Force UTF-8 so a fresh clone runs on any platform.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(__file__))
from inspect_schem import parse_nbt, flatten, parse_props, read_varint  # reuse

DUMP_SIM = os.path.join("build", "Release", "dump_sim.exe")

# Our BlockType enum values (from types.h)
BT = {
    "AIR": 0, "SOLID": 1, "LEVER": 10, "BUTTON": 11, "PRESSURE_PLATE": 12,
    "REDSTONE_BLOCK": 13, "REDSTONE_DUST": 20, "REDSTONE_TORCH": 30,
    "REPEATER": 31, "COMPARATOR": 32, "PISTON": 40, "STICKY_PISTON": 41,
    "LAMP": 54,
}
BT_NAMES = {v: k for k, v in BT.items()}
FLAG_POWERED = 1 << 0
FLAG_STRONG = 1 << 4

TORCH_NAMES = ("redstone_wall_torch", "redstone_torch")


def load_ground_truth(path):
    """Return dict (x,y,z) -> {'name': str, 'props': {...}}"""
    with gzip.open(path, "rb") as f:
        data = f.read()
    _, root = parse_nbt(data)
    root = flatten(root)
    W, H, L = root["Width"], root["Height"], root["Length"]
    palette = root.get("Palette") or root["Blocks"]["Palette"]
    bd = root.get("BlockData") or root["Blocks"]["Data"]
    idx_to_state = {v: k for k, v in palette.items()}

    gt = {}
    pos = 0
    idx = 0
    total = W * H * L
    while pos < len(bd) and idx < total:
        v, pos = read_varint(bd, pos)
        x = idx % W
        z = (idx // W) % L
        y = idx // (W * L)
        idx += 1
        state = idx_to_state.get(v)
        if not state:
            continue
        name, props = parse_props(state)
        bname = name.replace("minecraft:", "")
        if bname in ("air", "cave_air"):
            continue
        gt[(x, y, z)] = {"name": bname, "props": props}
    return gt, (W, H, L)


def load_sim_dump(schem_path, ticks):
    """Run dump_sim.exe, parse output into dict (x,y,z) -> block dict."""
    cmd = [DUMP_SIM, schem_path, str(ticks)]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    sim = {}
    for line in res.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("["):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        try:
            x, y, z = int(parts[0]), int(parts[1]), int(parts[2])
            sim[(x, y, z)] = {
                "type": int(parts[3]),
                "facing": int(parts[4]),
                "signal": int(parts[5]),
                "flags": int(parts[6], 16),
                "state": int(parts[7], 16),
            }
        except ValueError:
            continue
    return sim


def expected_from_gt(gt_entry):
    """Given a ground-truth entry, extract expected signal/powered/lit.

    Returns dict with keys 'signal', 'powered', 'lit' (when applicable)."""
    name = gt_entry["name"]
    p = gt_entry["props"]
    out = {"name": name}
    if name == "redstone_wire":
        out["signal"] = int(p.get("power", 0))
    elif name == "lever":
        out["powered"] = p.get("powered") == "true"
    elif name in TORCH_NAMES:
        out["lit"] = p.get("lit", "true") == "true"
    elif name == "redstone_lamp":
        out["lit"] = p.get("lit", "false") == "true"
    elif name == "repeater":
        out["powered"] = p.get("powered") == "true"
        out["locked"] = p.get("locked") == "true"
    return out


def actual_from_sim(sim_entry, gt_name):
    """Extract sim-side signal/powered/lit so we can compare."""
    if sim_entry is None:
        return {"signal": 0, "powered": False, "lit": False, "missing": True}
    out = {
        "type": sim_entry["type"],
        "signal": sim_entry["signal"],
        "powered": bool(sim_entry["flags"] & FLAG_POWERED),
    }
    # Torches / lamps — lit means the block is emitting light in MC.
    # For our sim: torch.lit = (signal > 0). lamp.lit = (flags & POWERED).
    if gt_name in TORCH_NAMES:
        out["lit"] = sim_entry["signal"] > 0
    elif gt_name == "redstone_lamp":
        out["lit"] = bool(sim_entry["flags"] & FLAG_POWERED)
    return out


def compare(schem_path, ticks=500):
    print(f"\n{'=' * 70}")
    print(f"COMPARE: {schem_path}  (ticks={ticks})")
    print(f"{'=' * 70}")

    gt, (W, H, L) = load_ground_truth(schem_path)
    sim = load_sim_dump(schem_path, ticks)

    # Collect diffs
    diffs = []
    # Key blocks to check explicitly
    for pos, gt_e in sorted(gt.items()):
        name = gt_e["name"]
        if name in ("light_gray_concrete_powder",):
            continue  # floor, we removed it anyway
        exp = expected_from_gt(gt_e)
        act = actual_from_sim(sim.get(pos), name)

        mismatch = False
        reasons = []
        if "signal" in exp and act.get("signal", -1) != exp["signal"]:
            mismatch = True
            reasons.append(f"signal exp={exp['signal']} got={act.get('signal')}")
        if "powered" in exp and act.get("powered") != exp["powered"]:
            mismatch = True
            reasons.append(f"powered exp={exp['powered']} got={act.get('powered')}")
        if "lit" in exp and act.get("lit") != exp["lit"]:
            mismatch = True
            reasons.append(f"lit exp={exp['lit']} got={act.get('lit')}")

        if mismatch:
            diffs.append((pos, name, reasons, exp, act))

    if not diffs:
        print("✓ All blocks match ground truth!")
        return True

    print(f"✗ {len(diffs)} divergences:\n")
    for pos, name, reasons, exp, act in diffs:
        x, y, z = pos
        print(f"  ({x},{y},{z}) {name}")
        for r in reasons:
            print(f"      {r}")
        print(f"      [exp] {exp}")
        print(f"      [sim] {act}")
    return False


def main():
    args = sys.argv[1:]
    if not args:
        args = [
            "schematics/1redstone.schem",
            "schematics/2redstone.schem",
            "schematics/3redstone.schem",
            "schematics/4redstone.schem",
        ]
    ticks = 500
    paths = []
    for a in args:
        if a.isdigit():
            ticks = int(a)
        else:
            paths.append(a)
    all_ok = True
    for p in paths:
        if os.path.exists(p):
            ok = compare(p, ticks)
            all_ok = all_ok and ok
        else:
            print(f"!! not found: {p}")
            all_ok = False
    print("\n" + ("=" * 70))
    print("OVERALL:", "PASS" if all_ok else "FAIL")


if __name__ == "__main__":
    main()
