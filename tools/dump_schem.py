#!/usr/bin/env python3
"""Dump all non-air blocks of a .schem file."""
import gzip, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from inspect_schem import parse_nbt, flatten, parse_props, read_varint

def dump(path):
    with gzip.open(path, "rb") as f:
        data = f.read()
    _, root = parse_nbt(data)
    root = flatten(root)
    W, H, L = root["Width"], root["Height"], root["Length"]
    palette = root.get("Palette") or root["Blocks"]["Palette"]
    bd = root.get("BlockData") or root["Blocks"]["Data"]
    idx_to_state = {v: k for k, v in palette.items()}

    print(f"{path}: {W}x{H}x{L}")
    pos = 0
    idx = 0
    total = W * H * L
    blocks = []
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
        blocks.append((x, y, z, bname, props))

    for y in range(H):
        print(f"\n--- y={y} ---")
        for x, yy, z, bname, props in blocks:
            if yy != y:
                continue
            short = ""
            for k in ("power","lit","powered","facing"):
                if k in props:
                    short += f" {k}={props[k]}"
            print(f"  ({x},{yy},{z}) {bname}{short}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        dump(p)
