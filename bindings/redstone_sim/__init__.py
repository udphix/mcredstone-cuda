"""Python bindings for the redstone simulator GPU engine.

Example:
    from redstone_sim import World, BLOCK_LEVER, BLOCK_LAMP, BLOCK_REDSTONE_DUST, DIR_DOWN

    w = World(size=(8, 4, 8))
    w.set_block(0, 1, 0, BLOCK_LEVER, facing=DIR_DOWN)
    for z in range(1, 7):
        w.set_block(0, 1, z, BLOCK_REDSTONE_DUST)
    w.set_block(0, 1, 7, BLOCK_LAMP)
    w.tick(50)
    print("lamp lit:", w.is_powered(0, 1, 7))

Requires the native library built via CMake:
    cmake --build build --target redstone_sim --config Release
"""

__version__ = "0.1.0"

from ._core import (
    # Main class
    World,
    Block,
    Snapshot,

    # Batched simulation (N independent worlds per kernel launch)
    WorldBatch,
    BatchSnapshot,
    BatchProbeResult,

    # BlockType constants
    BLOCK_AIR,
    BLOCK_SOLID,
    BLOCK_LEVER,
    BLOCK_BUTTON,
    BLOCK_PRESSURE_PLATE,
    BLOCK_REDSTONE_BLOCK,
    BLOCK_REDSTONE_DUST,
    BLOCK_REDSTONE_TORCH,
    BLOCK_REPEATER,
    BLOCK_COMPARATOR,
    BLOCK_PISTON,
    BLOCK_STICKY_PISTON,
    BLOCK_LAMP,

    # Direction constants
    DIR_DOWN,
    DIR_UP,
    DIR_NORTH,
    DIR_SOUTH,
    DIR_WEST,
    DIR_EAST,
    DIR_NONE,

    # Block flags
    FLAG_POWERED,
    FLAG_LOCKED,
    FLAG_EXTENDED,
    FLAG_SCHEDULED,
    FLAG_STRONG_POWER,
    FLAG_UPDATED,
)

# Gymnasium env is in .env — import explicitly:
#   from redstone_sim.env import RedstoneEnv
# (kept out of the top-level imports to avoid forcing numpy/gymnasium deps
#  on users who only want the core simulator bindings)


__all__ = [
    "World", "Block", "Snapshot",
    "WorldBatch", "BatchSnapshot", "BatchProbeResult",
    "BLOCK_AIR", "BLOCK_SOLID", "BLOCK_LEVER", "BLOCK_BUTTON",
    "BLOCK_PRESSURE_PLATE", "BLOCK_REDSTONE_BLOCK", "BLOCK_REDSTONE_DUST",
    "BLOCK_REDSTONE_TORCH", "BLOCK_REPEATER", "BLOCK_COMPARATOR",
    "BLOCK_PISTON", "BLOCK_STICKY_PISTON", "BLOCK_LAMP",
    "DIR_DOWN", "DIR_UP", "DIR_NORTH", "DIR_SOUTH", "DIR_WEST", "DIR_EAST", "DIR_NONE",
    "FLAG_POWERED", "FLAG_LOCKED", "FLAG_EXTENDED", "FLAG_SCHEDULED",
    "FLAG_STRONG_POWER", "FLAG_UPDATED",
]
