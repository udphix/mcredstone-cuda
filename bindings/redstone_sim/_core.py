"""ctypes wrapper over the native simulator DLL.

Loads `build/Release/redstone_sim.dll` (or Debug fallback) and exposes a
pythonic `World` class plus constants that mirror `include/redstone/types.h`.

This is intentionally a thin wrapper — performance-sensitive inner loops
(simulation, signal propagation) stay on the GPU inside the native library.
Python only manages lifetimes, shapes, and RL glue.
"""

from __future__ import annotations

import ctypes as C
import os
from pathlib import Path
from typing import Optional, Tuple

# ── Locate and load the DLL ──────────────────────────────────────────────

_REPO_ROOT = Path(__file__).resolve().parents[2]

_DLL_CANDIDATES = [
    _REPO_ROOT / "build" / "Release" / "redstone_sim.dll",
    _REPO_ROOT / "build" / "Debug" / "redstone_sim.dll",
    _REPO_ROOT / "build" / "redstone_sim.dll",
    _REPO_ROOT / "build" / "redstone_sim.so",          # linux
    _REPO_ROOT / "build" / "libredstone_sim.so",       # linux
]


def _find_dll() -> str:
    # Allow override via env var for unusual setups
    env = os.environ.get("REDSTONE_SIM_DLL")
    if env and Path(env).exists():
        return env
    for p in _DLL_CANDIDATES:
        if p.exists():
            return str(p)
    raise FileNotFoundError(
        "redstone_sim shared library not found. Build it first:\n"
        "  cd build && cmake --build . --target redstone_sim --config Release\n"
        f"Searched: {[str(p) for p in _DLL_CANDIDATES]}"
    )


_lib = C.CDLL(_find_dll())


# ── Block struct (8 bytes, matches include/redstone/types.h #pragma pack) ──

class Block(C.Structure):
    """Mirrors the 8-byte Block struct on the C side."""
    _pack_ = 1
    _fields_ = [
        ("type",           C.c_uint8),
        ("signal",         C.c_uint8),
        ("facing",         C.c_uint8),
        ("flags",          C.c_uint8),
        ("state",          C.c_uint16),
        ("tick_scheduled", C.c_uint16),
    ]

    def __repr__(self) -> str:
        return (f"Block(type={self.type}, signal={self.signal}, "
                f"facing={self.facing}, flags=0x{self.flags:02x}, "
                f"state=0x{self.state:04x})")


assert C.sizeof(Block) == 8, "Block must be 8 bytes (matches native struct)"


# ── Function signatures ──────────────────────────────────────────────────

# Lifecycle
_lib.registry_init.argtypes = []
_lib.registry_init.restype = None

_lib.world_create.argtypes = [C.c_int, C.c_int, C.c_int]
_lib.world_create.restype = C.c_void_p

_lib.world_destroy.argtypes = [C.c_void_p]
_lib.world_destroy.restype = None

# Block access
_lib.world_set_block.argtypes = [
    C.c_void_p, C.c_int, C.c_int, C.c_int,  # world, x, y, z
    C.c_int, C.c_int, C.c_uint16,           # type, facing, state
]
_lib.world_set_block.restype = None

_lib.world_get_block.argtypes = [C.c_void_p, C.c_int, C.c_int, C.c_int]
_lib.world_get_block.restype = Block

_lib.world_set_lever_powered.argtypes = [
    C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_int,
]
_lib.world_set_lever_powered.restype = None  # struct by value (8 bytes, fits RAX)

# Simulation
_lib.world_tick.argtypes = [C.c_void_p]
_lib.world_tick.restype = None

_lib.world_tick_n.argtypes = [C.c_void_p, C.c_int]
_lib.world_tick_n.restype = None

# Buffer sync
_lib.world_sync_to_gpu.argtypes = [C.c_void_p]
_lib.world_sync_to_gpu.restype = None

_lib.world_sync_from_gpu.argtypes = [C.c_void_p]
_lib.world_sync_from_gpu.restype = None

# Schematic I/O
_lib.world_import_schem.argtypes = [C.c_void_p, C.c_char_p]
_lib.world_import_schem.restype = C.c_int

_lib.world_export_schem.argtypes = [C.c_void_p, C.c_char_p]
_lib.world_export_schem.restype = C.c_int

# Snapshots (for RL reset/restore)
_lib.world_save_state.argtypes = [C.c_void_p]
_lib.world_save_state.restype = C.c_void_p

_lib.world_restore_state.argtypes = [C.c_void_p, C.c_void_p]
_lib.world_restore_state.restype = None

_lib.world_snapshot_destroy.argtypes = [C.c_void_p]
_lib.world_snapshot_destroy.restype = None

# Utility
_lib.world_clear.argtypes = [C.c_void_p]
_lib.world_clear.restype = None

_lib.world_copy_host_buffer.argtypes = [C.c_void_p, C.POINTER(Block)]
_lib.world_copy_host_buffer.restype = None


# ── WorldBatch: N independent worlds per kernel launch ───────────────────

class BatchProbeResult(C.Structure):
    """Mirrors BatchProbeResult in include/redstone/batch.h."""
    _fields_ = [
        ("inst",   C.c_int32),
        ("x",      C.c_int32),
        ("y",      C.c_int32),
        ("z",      C.c_int32),
        ("type",   C.c_uint8),
        ("signal", C.c_uint8),
        ("flags",  C.c_uint8),
        ("_pad",   C.c_uint8),
    ]


_lib.world_batch_create.argtypes = [C.c_int, C.c_int, C.c_int, C.c_int]
_lib.world_batch_create.restype = C.c_void_p

_lib.world_batch_destroy.argtypes = [C.c_void_p]
_lib.world_batch_destroy.restype = None

_lib.world_batch_set_block.argtypes = [
    C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_int,  # batch, inst, x, y, z
    C.c_int, C.c_int, C.c_uint16,                    # type, facing, state
]
_lib.world_batch_set_block.restype = None

_lib.world_batch_get_block.argtypes = [
    C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_int,
]
_lib.world_batch_get_block.restype = Block

_lib.world_batch_clear.argtypes = [C.c_void_p]
_lib.world_batch_clear.restype = None

_lib.world_batch_clear_instance.argtypes = [C.c_void_p, C.c_int]
_lib.world_batch_clear_instance.restype = None

_lib.world_batch_copy_instance.argtypes = [C.c_void_p, C.c_int, C.c_int]
_lib.world_batch_copy_instance.restype = None

_lib.world_batch_sync_to_gpu.argtypes = [C.c_void_p]
_lib.world_batch_sync_to_gpu.restype = None

_lib.world_batch_sync_from_gpu.argtypes = [C.c_void_p]
_lib.world_batch_sync_from_gpu.restype = None

_lib.world_batch_copy_host_buffer.argtypes = [C.c_void_p, C.POINTER(Block)]
_lib.world_batch_copy_host_buffer.restype = None

_lib.world_batch_tick.argtypes = [C.c_void_p, C.c_int]
_lib.world_batch_tick.restype = None

_lib.world_batch_tick_until_stable.argtypes = [C.c_void_p, C.c_int, C.c_int]
_lib.world_batch_tick_until_stable.restype = C.c_int

_lib.world_batch_refresh_done.argtypes = [C.c_void_p]
_lib.world_batch_refresh_done.restype = None

_lib.world_batch_done_count.argtypes = [C.c_void_p]
_lib.world_batch_done_count.restype = C.c_int

_lib.world_batch_instance_done.argtypes = [C.c_void_p, C.c_int]
_lib.world_batch_instance_done.restype = C.c_int

_lib.world_batch_instance_tick.argtypes = [C.c_void_p, C.c_int]
_lib.world_batch_instance_tick.restype = C.c_uint32

_lib.world_batch_set_levers.argtypes = [
    C.c_void_p, C.POINTER(C.c_int32), C.POINTER(C.c_uint8), C.c_int,
]
_lib.world_batch_set_levers.restype = None

_lib.world_batch_set_probes.argtypes = [C.c_void_p, C.POINTER(C.c_int32), C.c_int]
_lib.world_batch_set_probes.restype = None

_lib.world_batch_read_probes.argtypes = [C.c_void_p]
_lib.world_batch_read_probes.restype = C.POINTER(BatchProbeResult)

_lib.world_batch_save.argtypes = [C.c_void_p]
_lib.world_batch_save.restype = C.c_void_p

_lib.world_batch_restore.argtypes = [C.c_void_p, C.c_void_p]
_lib.world_batch_restore.restype = None

_lib.world_batch_restore_instance.argtypes = [C.c_void_p, C.c_void_p, C.c_int]
_lib.world_batch_restore_instance.restype = None

_lib.world_batch_snapshot_destroy.argtypes = [C.c_void_p]
_lib.world_batch_snapshot_destroy.restype = None


# ── Constants (mirror include/redstone/types.h) ──────────────────────────

# BlockType
BLOCK_AIR              = 0
BLOCK_SOLID            = 1
BLOCK_LEVER            = 10
BLOCK_BUTTON           = 11
BLOCK_PRESSURE_PLATE   = 12
BLOCK_REDSTONE_BLOCK   = 13
BLOCK_REDSTONE_DUST    = 20
BLOCK_REDSTONE_TORCH   = 30
BLOCK_REPEATER         = 31
BLOCK_COMPARATOR       = 32
BLOCK_PISTON           = 40
BLOCK_STICKY_PISTON    = 41
BLOCK_LAMP             = 54

# Direction
DIR_DOWN  = 0
DIR_UP    = 1
DIR_NORTH = 2
DIR_SOUTH = 3
DIR_WEST  = 4
DIR_EAST  = 5
DIR_NONE  = 6

# Block flags
FLAG_POWERED      = 1 << 0
FLAG_LOCKED       = 1 << 1
FLAG_EXTENDED     = 1 << 2
FLAG_SCHEDULED    = 1 << 3
FLAG_STRONG_POWER = 1 << 4
FLAG_UPDATED      = 1 << 5


# ── Init the block registry once at import time ──────────────────────────

_lib.registry_init()


# ── Snapshot handle ──────────────────────────────────────────────────────

class Snapshot:
    """Opaque handle to a saved GPU state. Destroy via `close()`."""

    def __init__(self, handle: int):
        self._h: Optional[int] = handle

    def close(self) -> None:
        if self._h:
            _lib.world_snapshot_destroy(self._h)
            self._h = None

    def __del__(self):
        self.close()


# ── World class ──────────────────────────────────────────────────────────

class World:
    """A 3D redstone simulation world backed by GPU memory.

    Example:
        from redstone_sim import World, BLOCK_LEVER, BLOCK_LAMP, DIR_DOWN

        w = World(size=(8, 4, 8))
        w.set_block(0, 1, 0, BLOCK_LEVER, facing=DIR_DOWN)
        w.set_block(7, 1, 0, BLOCK_LAMP)
        w.tick(50)
        w.sync()                               # pull GPU state back to host
        print(w.get_block(7, 1, 0).signal)
    """

    def __init__(self, size: Tuple[int, int, int]):
        sx, sy, sz = size
        handle = _lib.world_create(sx, sy, sz)
        if not handle:
            raise RuntimeError(f"world_create({sx},{sy},{sz}) failed")
        self._h: Optional[int] = handle
        self.size: Tuple[int, int, int] = (sx, sy, sz)

    # ── Lifecycle ───────────────────────────────────────────────────

    def destroy(self) -> None:
        if self._h:
            _lib.world_destroy(self._h)
            self._h = None

    def __del__(self):
        self.destroy()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.destroy()

    def _check_alive(self) -> None:
        if not self._h:
            raise RuntimeError("World has been destroyed")

    # ── Block access ────────────────────────────────────────────────

    def set_block(
        self,
        x: int, y: int, z: int,
        type_: int,
        facing: int = DIR_NONE,
        state: int = 0,
    ) -> None:
        """Place a block at (x,y,z). Call `sync()` after mutations if you
        want them visible on GPU before the next tick (tick() auto-syncs)."""
        self._check_alive()
        _lib.world_set_block(self._h, x, y, z, type_, facing, state)

    def get_block(self, x: int, y: int, z: int) -> Block:
        """Read a block from the host mirror. After `tick()` you must call
        `sync()` first to pull the latest GPU state back to the host."""
        self._check_alive()
        return _lib.world_get_block(self._h, x, y, z)

    def clear(self) -> None:
        """Reset all blocks to AIR."""
        self._check_alive()
        _lib.world_clear(self._h)

    def get_all_blocks(self, auto_sync: bool = True):
        """Bulk-read the full grid into a numpy structured array of shape
        (size_y, size_z, size_x). Fast path for building RL observations.

        Returns a numpy array with fields: type, signal, facing, flags,
        state, tick_scheduled. Treat the buffer as read-only.
        """
        try:
            import numpy as np
        except ImportError as e:
            raise RuntimeError(
                "numpy is required for get_all_blocks(); pip install numpy"
            ) from e
        self._check_alive()
        if auto_sync:
            self.sync()
        sx, sy, sz = self.size
        n = sx * sy * sz
        buf = (Block * n)()
        _lib.world_copy_host_buffer(self._h, buf)
        dtype = np.dtype([
            ("type",           "u1"),
            ("signal",         "u1"),
            ("facing",         "u1"),
            ("flags",          "u1"),
            ("state",          "u2"),
            ("tick_scheduled", "u2"),
        ])
        return np.frombuffer(buf, dtype=dtype).reshape(sy, sz, sx).copy()

    # ── Simulation ──────────────────────────────────────────────────

    def tick(self, n: int = 1) -> None:
        """Advance simulation by `n` game ticks."""
        self._check_alive()
        if n == 1:
            _lib.world_tick(self._h)
        else:
            _lib.world_tick_n(self._h, n)

    # ── Buffer sync ─────────────────────────────────────────────────

    def sync(self) -> None:
        """Pull GPU state back to host so `get_block()` sees latest values."""
        self._check_alive()
        _lib.world_sync_from_gpu(self._h)

    def push(self) -> None:
        """Push host state to GPU. Normally unnecessary — `tick()` handles it."""
        self._check_alive()
        _lib.world_sync_to_gpu(self._h)

    # ── Schematics ──────────────────────────────────────────────────

    def import_schem(self, path: str) -> int:
        """Load a Sponge Schematic v2 (.schem) file. Returns number of
        non-air blocks placed (or negative on error)."""
        self._check_alive()
        return _lib.world_import_schem(self._h, str(path).encode("utf-8"))

    def export_schem(self, path: str) -> int:
        """Write the current world out as a .schem file."""
        self._check_alive()
        return _lib.world_export_schem(self._h, str(path).encode("utf-8"))

    # ── Snapshots (RL reset/restore) ────────────────────────────────

    def save(self) -> Snapshot:
        """Save the current GPU state as a snapshot. Use for RL reset."""
        self._check_alive()
        h = _lib.world_save_state(self._h)
        if not h:
            raise RuntimeError("world_save_state failed")
        return Snapshot(h)

    def restore(self, snap: Snapshot) -> None:
        """Restore a previously saved snapshot."""
        self._check_alive()
        if snap._h is None:
            raise RuntimeError("Snapshot has been closed")
        _lib.world_restore_state(self._h, snap._h)

    # ── Convenience helpers ─────────────────────────────────────────

    def get_signal(self, x: int, y: int, z: int, auto_sync: bool = True) -> int:
        """Returns signal (0-15) of block at (x,y,z). Auto-syncs by default."""
        if auto_sync:
            self.sync()
        return self.get_block(x, y, z).signal

    def is_powered(self, x: int, y: int, z: int, auto_sync: bool = True) -> bool:
        """Returns True if block has BLOCK_FLAG_POWERED set."""
        if auto_sync:
            self.sync()
        return bool(self.get_block(x, y, z).flags & FLAG_POWERED)

    def set_lever_powered(self, x: int, y: int, z: int, on: bool) -> None:
        """Toggle a lever's POWERED/signal state in place.

        No-op if the block at (x,y,z) is not a lever. The world is marked
        dirty so the next `tick()` syncs the new state to the GPU.

        Used by RL environments that evaluate a circuit across multiple
        input combinations (e.g., truth-table tasks) without destroying
        the rest of the layout.
        """
        self._check_alive()
        _lib.world_set_lever_powered(self._h, x, y, z, 1 if on else 0)

    def __repr__(self) -> str:
        return f"World(size={self.size}, handle={self._h})"


# ── Batch snapshot handle ────────────────────────────────────────────────

class BatchSnapshot:
    """Opaque handle to a saved batch GPU state. Destroy via `close()`."""

    def __init__(self, handle: int):
        self._h: Optional[int] = handle

    def close(self) -> None:
        if self._h:
            _lib.world_batch_snapshot_destroy(self._h)
            self._h = None

    def __del__(self):
        self.close()


# ── WorldBatch class ─────────────────────────────────────────────────────

_BLOCK_DTYPE_FIELDS = [
    ("type",           "u1"),
    ("signal",         "u1"),
    ("facing",         "u1"),
    ("flags",          "u1"),
    ("state",          "u2"),
    ("tick_scheduled", "u2"),
]


class WorldBatch:
    """N independent redstone worlds simulated in one kernel launch.

    A single 16^3 `World` is 4096 cells — far too little work to fill a GPU,
    so `World.tick()` costs ~80 us no matter how big the world is (it is pure
    launch latency). `WorldBatch` runs N worlds per launch, so that same ~80 us
    covers all of them.

    Semantics are identical to `World` — the batched kernels call the same
    per-cell device code (`src/kernels/*_device.cuh`), and
    `tests/test_batch_equivalence.py` asserts bit-identical state.

    Example:
        b = WorldBatch(n_worlds=256, size=(16, 16, 16))
        for i in range(256):
            b.set_block(i, 0, 0, 0, BLOCK_LEVER, facing=DIR_DOWN)
            ...
        b.tick_until_stable(max_ticks=64)
        b.set_probes([(i, 5, 1, 5) for i in range(256)])
        lit = [bool(p.flags & FLAG_POWERED) for p in b.read_probes()]

    Host<->GPU contract (differs from `World` — see include/redstone/batch.h):
      * `set_block` edits the host mirror and marks the batch dirty; the next
        tick uploads the WHOLE batch. Use it to build scenes.
      * `set_levers` writes GPU memory directly (no host round-trip). Use it
        inside evaluation loops — a full re-upload per truth-table row would
        dominate the runtime.
      * `restore*` touch GPU memory only and leave the host mirror stale.
        Call `sync()` before host-side block reads, or use probes.
      * Do NOT `set_block` after `set_levers`/`restore` without `sync()`
        first: the dirty-upload would clobber device state.
    """

    def __init__(self, n_worlds: int, size: Tuple[int, int, int]):
        sx, sy, sz = size
        handle = _lib.world_batch_create(n_worlds, sx, sy, sz)
        if not handle:
            raise RuntimeError(
                f"world_batch_create({n_worlds},{sx},{sy},{sz}) failed"
            )
        self._h: Optional[int] = handle
        self.n_worlds = n_worlds
        self.size: Tuple[int, int, int] = (sx, sy, sz)
        self.cells = sx * sy * sz

    # ── Lifecycle ───────────────────────────────────────────────────

    def destroy(self) -> None:
        if self._h:
            _lib.world_batch_destroy(self._h)
            self._h = None

    def __del__(self):
        self.destroy()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.destroy()

    def _check_alive(self) -> None:
        if not self._h:
            raise RuntimeError("WorldBatch has been destroyed")

    def _check_inst(self, inst: int) -> None:
        if not (0 <= inst < self.n_worlds):
            raise IndexError(
                f"instance {inst} out of range [0, {self.n_worlds})"
            )

    # ── Block access ────────────────────────────────────────────────

    def set_block(self, inst: int, x: int, y: int, z: int, type_: int,
                  facing: int = DIR_NONE, state: int = 0) -> None:
        """Place a block in instance `inst`. Marks the batch dirty."""
        self._check_alive()
        self._check_inst(inst)
        _lib.world_batch_set_block(self._h, inst, x, y, z, type_, facing, state)

    def get_block(self, inst: int, x: int, y: int, z: int) -> Block:
        """Read from the host mirror — call `sync()` first after ticking."""
        self._check_alive()
        self._check_inst(inst)
        return _lib.world_batch_get_block(self._h, inst, x, y, z)

    def clear(self) -> None:
        """Reset every instance to AIR."""
        self._check_alive()
        _lib.world_batch_clear(self._h)

    def clear_instance(self, inst: int) -> None:
        self._check_alive()
        self._check_inst(inst)
        _lib.world_batch_clear_instance(self._h, inst)

    def copy_instance(self, src_inst: int, dst_inst: int) -> None:
        """Duplicate a built scene (host side). Useful for RL groups that
        sample K circuits against one prompt."""
        self._check_alive()
        self._check_inst(src_inst)
        self._check_inst(dst_inst)
        _lib.world_batch_copy_instance(self._h, src_inst, dst_inst)

    def get_all_blocks(self, auto_sync: bool = True):
        """Bulk-read every instance as a numpy structured array of shape
        (n_worlds, size_y, size_z, size_x) — ONE device->host copy for the
        whole batch, not one per world.
        """
        try:
            import numpy as np
        except ImportError as e:
            raise RuntimeError(
                "numpy is required for get_all_blocks(); pip install numpy"
            ) from e
        self._check_alive()
        if auto_sync:
            self.sync()
        sx, sy, sz = self.size
        arr = np.empty(self.n_worlds * self.cells, dtype=np.dtype(_BLOCK_DTYPE_FIELDS))
        _lib.world_batch_copy_host_buffer(
            self._h, arr.ctypes.data_as(C.POINTER(Block))
        )
        return arr.reshape(self.n_worlds, sy, sz, sx)

    # ── Simulation ──────────────────────────────────────────────────

    def tick(self, n: int = 1) -> None:
        """Advance every instance by `n` game ticks (converged instances are
        skipped — they are already at a fixed point)."""
        self._check_alive()
        _lib.world_batch_tick(self._h, n)

    def tick_until_stable(self, max_ticks: int = 64, check_every: int = 4) -> int:
        """Tick until every instance reaches a fixed point, or `max_ticks`.

        An instance is converged when a whole tick changed no block AND no
        block holds a pending tile tick. Returns the number of ticks run —
        compare against `max_ticks` to detect circuits that never settle
        (clocks, oscillating torch loops).
        """
        self._check_alive()
        return _lib.world_batch_tick_until_stable(self._h, max_ticks, check_every)

    def refresh_done(self) -> None:
        """Pull the per-instance convergence flags from the GPU."""
        self._check_alive()
        _lib.world_batch_refresh_done(self._h)

    def done_count(self) -> int:
        """How many instances have converged (per the last refresh)."""
        self._check_alive()
        return _lib.world_batch_done_count(self._h)

    def instance_done(self, inst: int) -> bool:
        self._check_alive()
        self._check_inst(inst)
        return bool(_lib.world_batch_instance_done(self._h, inst))

    def instance_tick(self, inst: int) -> int:
        """That instance's game tick (clocks are per-instance)."""
        self._check_alive()
        self._check_inst(inst)
        return int(_lib.world_batch_instance_tick(self._h, inst))

    # ── Buffer sync ─────────────────────────────────────────────────

    def sync(self) -> None:
        """Pull GPU state back to the host mirror (whole batch)."""
        self._check_alive()
        _lib.world_batch_sync_from_gpu(self._h)

    def push(self) -> None:
        """Push the host mirror to GPU. Normally unnecessary — tick() does it."""
        self._check_alive()
        _lib.world_batch_sync_to_gpu(self._h)

    # ── Device-side lever writes ────────────────────────────────────

    def set_levers(self, entries) -> None:
        """Toggle many levers across many instances in one GPU write.

        `entries` is an iterable of (inst, x, y, z, on). Non-lever targets
        are ignored, matching `World.set_lever_powered`. This does NOT dirty
        the host mirror, so it is safe (and fast) inside evaluation loops.
        """
        self._check_alive()
        entries = list(entries)
        if not entries:
            return
        n = len(entries)
        pos = (C.c_int32 * (n * 4))()
        vals = (C.c_uint8 * n)()
        for k, (inst, x, y, z, on) in enumerate(entries):
            self._check_inst(inst)
            pos[k * 4 + 0] = inst
            pos[k * 4 + 1] = x
            pos[k * 4 + 2] = y
            pos[k * 4 + 3] = z
            vals[k] = 1 if on else 0
        _lib.world_batch_set_levers(self._h, pos, vals, n)

    # ── Probes ──────────────────────────────────────────────────────

    def set_probes(self, entries) -> None:
        """Register cells to read. `entries` = iterable of (inst, x, y, z)."""
        self._check_alive()
        entries = list(entries)
        self._probe_count = len(entries)
        if not entries:
            _lib.world_batch_set_probes(self._h, None, 0)
            return
        pos = (C.c_int32 * (len(entries) * 4))()
        for k, (inst, x, y, z) in enumerate(entries):
            self._check_inst(inst)
            pos[k * 4 + 0] = inst
            pos[k * 4 + 1] = x
            pos[k * 4 + 2] = y
            pos[k * 4 + 3] = z
        _lib.world_batch_set_probes(self._h, pos, len(entries))

    def read_probes(self):
        """Read all registered probes in one device->host copy.

        Returns a list of BatchProbeResult (inst, x, y, z, type, signal, flags).
        """
        self._check_alive()
        count = getattr(self, "_probe_count", 0)
        if count == 0:
            return []
        ptr = _lib.world_batch_read_probes(self._h)
        return [ptr[i] for i in range(count)]

    def probes_powered(self):
        """Convenience: read probes and return a list of bools (POWERED)."""
        return [bool(p.flags & FLAG_POWERED) for p in self.read_probes()]

    # ── Snapshots ───────────────────────────────────────────────────

    def save(self) -> BatchSnapshot:
        """Snapshot every instance (blocks + per-instance clocks + flags)."""
        self._check_alive()
        h = _lib.world_batch_save(self._h)
        if not h:
            raise RuntimeError("world_batch_save failed")
        return BatchSnapshot(h)

    def restore(self, snap: BatchSnapshot) -> None:
        """Restore all instances. Leaves the host mirror stale — see class doc."""
        self._check_alive()
        if snap._h is None:
            raise RuntimeError("BatchSnapshot has been closed")
        _lib.world_batch_restore(self._h, snap._h)

    def restore_instance(self, snap: BatchSnapshot, inst: int) -> None:
        """Restore one instance (blocks + its own clock), leaving others running."""
        self._check_alive()
        self._check_inst(inst)
        if snap._h is None:
            raise RuntimeError("BatchSnapshot has been closed")
        _lib.world_batch_restore_instance(self._h, snap._h, inst)

    # ── Convenience helpers (host-mirror reads) ─────────────────────

    def get_signal(self, inst: int, x: int, y: int, z: int,
                   auto_sync: bool = True) -> int:
        if auto_sync:
            self.sync()
        return self.get_block(inst, x, y, z).signal

    def is_powered(self, inst: int, x: int, y: int, z: int,
                   auto_sync: bool = True) -> bool:
        if auto_sync:
            self.sync()
        return bool(self.get_block(inst, x, y, z).flags & FLAG_POWERED)

    def __repr__(self) -> str:
        return (f"WorldBatch(n_worlds={self.n_worlds}, size={self.size}, "
                f"handle={self._h})")
