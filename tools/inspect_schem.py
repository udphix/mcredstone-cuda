#!/usr/bin/env python3
"""
Minimal .schem (Sponge Schematic v2) palette + dimensions inspector.
Decompresses gzip, parses NBT enough to dump:
  - Dimensions
  - All palette entries (block state strings)
  - For each palette entry, decode the properties (lever powered, lamp lit, etc.)
"""
import gzip, struct, sys, os

# Windows consoles default to cp1252 and cannot encode the arrows / check marks
# printed below. Force UTF-8 so a fresh clone runs on any platform.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TAG_END = 0
TAG_BYTE = 1
TAG_SHORT = 2
TAG_INT = 3
TAG_LONG = 4
TAG_FLOAT = 5
TAG_DOUBLE = 6
TAG_BYTE_ARRAY = 7
TAG_STRING = 8
TAG_LIST = 9
TAG_COMPOUND = 10
TAG_INT_ARRAY = 11
TAG_LONG_ARRAY = 12


class Reader:
    def __init__(self, data):
        self.data = data
        self.pos = 0

    def u8(self):
        v = self.data[self.pos]
        self.pos += 1
        return v

    def i16(self):
        v = struct.unpack_from(">h", self.data, self.pos)[0]
        self.pos += 2
        return v

    def i32(self):
        v = struct.unpack_from(">i", self.data, self.pos)[0]
        self.pos += 4
        return v

    def i64(self):
        v = struct.unpack_from(">q", self.data, self.pos)[0]
        self.pos += 8
        return v

    def f32(self):
        v = struct.unpack_from(">f", self.data, self.pos)[0]
        self.pos += 4
        return v

    def f64(self):
        v = struct.unpack_from(">d", self.data, self.pos)[0]
        self.pos += 8
        return v

    def string(self):
        n = self.i16()
        s = self.data[self.pos : self.pos + n].decode("utf-8", errors="replace")
        self.pos += n
        return s

    def byte_array(self):
        n = self.i32()
        b = self.data[self.pos : self.pos + n]
        self.pos += n
        return b


def read_payload(r, t):
    if t == TAG_BYTE: return r.u8()
    if t == TAG_SHORT: return r.i16()
    if t == TAG_INT: return r.i32()
    if t == TAG_LONG: return r.i64()
    if t == TAG_FLOAT: return r.f32()
    if t == TAG_DOUBLE: return r.f64()
    if t == TAG_BYTE_ARRAY: return r.byte_array()
    if t == TAG_STRING: return r.string()
    if t == TAG_LIST:
        et = r.u8()
        n = r.i32()
        return [read_payload(r, et) for _ in range(n)]
    if t == TAG_COMPOUND:
        out = {}
        while True:
            ct = r.u8()
            if ct == TAG_END:
                break
            cn = r.string()
            out[cn] = read_payload(r, ct)
        return out
    if t == TAG_INT_ARRAY:
        n = r.i32()
        out = [r.i32() for _ in range(n)]
        return out
    if t == TAG_LONG_ARRAY:
        n = r.i32()
        out = [r.i64() for _ in range(n)]
        return out
    raise ValueError(f"Unknown tag type {t}")


def parse_nbt(data):
    r = Reader(data)
    root_t = r.u8()
    root_name = r.string()
    root = read_payload(r, root_t)
    return root_name, root


def flatten(root):
    """Unwrap Sponge v3 'Schematic' wrapper if present."""
    if isinstance(root, dict) and "Schematic" in root and isinstance(root["Schematic"], dict):
        return root["Schematic"]
    return root


def parse_props(block_state):
    """'minecraft:lever[face=floor,facing=north,powered=true]' → (name, {props})"""
    if "[" not in block_state:
        return block_state, {}
    name = block_state[: block_state.index("[")]
    props_str = block_state[block_state.index("[") + 1 : -1]
    props = {}
    for kv in props_str.split(","):
        if "=" in kv:
            k, v = kv.split("=", 1)
            props[k.strip()] = v.strip()
    return name, props


def read_varint(data, pos):
    val = 0
    shift = 0
    while pos < len(data):
        b = data[pos]
        pos += 1
        val |= (b & 0x7F) << shift
        if not (b & 0x80):
            break
        shift += 7
    return val, pos


def inspect(path):
    print(f"\n{'=' * 60}")
    print(f"FILE: {path}")
    print(f"{'=' * 60}")

    with gzip.open(path, "rb") as f:
        data = f.read()

    root_name, root = parse_nbt(data)
    root = flatten(root)

    W = root.get("Width", 0)
    H = root.get("Height", 0)
    L = root.get("Length", 0)
    print(f"Dimensions: {W}x{H}x{L}  (root_name={root_name!r})")

    # Palette may be under 'Palette' or 'Blocks.Palette'
    palette = root.get("Palette")
    block_data = root.get("BlockData")
    if palette is None and "Blocks" in root and isinstance(root["Blocks"], dict):
        palette = root["Blocks"].get("Palette")
        block_data = root["Blocks"].get("Data")

    if not palette:
        print("!! No palette found")
        return

    # Invert: index → state string
    idx_to_state = {v: k for k, v in palette.items()}
    print(f"Palette ({len(palette)} entries):")

    # Print palette with highlighting of key props
    interesting = ["lever", "redstone_lamp", "redstone_torch", "redstone_wall_torch",
                   "redstone_wire", "repeater"]
    for i in sorted(idx_to_state.keys()):
        state = idx_to_state[i]
        mark = ""
        for key in interesting:
            if key in state:
                mark = "  <<<"
                break
        print(f"  [{i:3d}] {state}{mark}")

    # Decode block data and find interesting block positions
    if block_data is None:
        print("!! No BlockData")
        return

    # block_data is bytes (from BYTE_ARRAY); decode varints
    positions = {}  # name → list of (x,y,z,state)
    pos = 0
    idx = 0
    total = W * H * L
    while pos < len(block_data) and idx < total:
        v, pos = read_varint(block_data, pos)
        x = idx % W
        z = (idx // W) % L
        y = idx // (W * L)
        idx += 1
        state = idx_to_state.get(v)
        if state is None:
            continue
        name, props = parse_props(state)
        bname = name.replace("minecraft:", "")
        if bname in ("air", "cave_air"):
            continue
        if bname in ("lever", "redstone_lamp", "redstone_torch",
                     "redstone_wall_torch", "redstone_wire", "repeater"):
            positions.setdefault(bname, []).append((x, y, z, state, props))

    print("\nKey blocks and their states:")
    for bname in ["lever", "redstone_lamp", "redstone_torch", "redstone_wall_torch",
                  "redstone_wire", "repeater"]:
        if bname in positions:
            print(f"  {bname}:")
            for x, y, z, state, props in positions[bname]:
                key_props = {}
                for k in ("powered", "lit", "power", "locked", "delay", "facing", "face",
                          "north", "south", "east", "west"):
                    if k in props:
                        key_props[k] = props[k]
                print(f"    ({x},{y},{z}) {state}")

    # Summary: is the lamp lit? Are levers powered?
    print("\nSUMMARY:")
    levers_on = 0
    levers_off = 0
    for x, y, z, state, props in positions.get("lever", []):
        if props.get("powered") == "true":
            levers_on += 1
            print(f"  LEVER ON  at ({x},{y},{z})")
        else:
            levers_off += 1
            print(f"  lever off at ({x},{y},{z})")
    for x, y, z, state, props in positions.get("redstone_lamp", []):
        lit = props.get("lit", "false")
        print(f"  LAMP {'LIT ' if lit == 'true' else 'off '}at ({x},{y},{z})  (lit={lit})")
    print(f"\n  → {levers_on} levers ON, {levers_off} levers OFF")


if __name__ == "__main__":
    paths = sys.argv[1:] or [
        "schematics/1redstone.schem",
        "schematics/2redstone.schem",
        "schematics/3redstone.schem",
        "schematics/4redstone.schem",
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                inspect(p)
            except Exception as e:
                print(f"[{p}] error: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"[{p}] not found")
