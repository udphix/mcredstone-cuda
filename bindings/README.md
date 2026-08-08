# Python Bindings

Thin Python layer over the C/CUDA simulator using ctypes.

## Layout

- `redstone_sim/__init__.py` — package entry point, re-exports public API
- `redstone_sim/_core.py` — ctypes wrapper around `redstone_sim.dll`
- `redstone_sim/env.py` — (TODO) Gymnasium-compatible environment
- `smoke_test.py` — validates DLL loading, World lifecycle, snapshots, schem import

## Setup

1. Build the native shared library (from the repo root):

   ```bash
   cmake -B build -DCMAKE_CUDA_ARCHITECTURES=86
   cmake --build build --target redstone_sim --config Release
   ```

   This produces `build/Release/redstone_sim.dll`.

2. Install this package as editable:

   ```bash
   pip install -e bindings          # pip install -e "bindings[env]" for Gymnasium
   ```

   It must be **editable**. The package does not vendor the compiled library:
   `_core.py` resolves `build/Release/redstone_sim.dll` relative to the
   repository root, which only works while the package still lives inside the
   repo. A regular install would need `$REDSTONE_SIM_DLL` set by hand.

3. Run the smoke test:

   ```bash
   python bindings/smoke_test.py
   ```

## Usage

```python
from redstone_sim import (
    World, BLOCK_LEVER, BLOCK_LAMP, BLOCK_REDSTONE_DUST, BLOCK_SOLID, DIR_DOWN,
)

with World(size=(8, 4, 8)) as w:
    # Build a simple "lever → dust → lamp" circuit
    w.set_block(0, 0, 0, BLOCK_SOLID)
    w.set_block(0, 1, 0, BLOCK_LEVER, facing=DIR_DOWN)
    for z in range(1, 7):
        w.set_block(0, 0, z, BLOCK_SOLID)
        w.set_block(0, 1, z, BLOCK_REDSTONE_DUST)
    w.set_block(0, 0, 7, BLOCK_SOLID)
    w.set_block(0, 1, 7, BLOCK_LAMP)

    w.tick(10)
    print("lamp lit:", w.is_powered(0, 1, 7))
```

## API summary

See `_core.py` for the full surface. Main entry points:

- `World(size)` / `w.destroy()` — lifecycle (context manager supported)
- `w.set_block(x, y, z, type, facing, state)` — place a block
- `w.get_block(x, y, z)` → `Block` — read (call `w.sync()` first after tick)
- `w.tick(n)` — advance simulation
- `w.import_schem(path)` / `w.export_schem(path)` — MC schematic I/O
- `w.save()` → `Snapshot`, `w.restore(snap)` — RL reset/restore
- `w.get_signal(x, y, z)`, `w.is_powered(x, y, z)` — convenience accessors

## DLL location

`_core.py` searches these paths (in order):
1. `$REDSTONE_SIM_DLL` environment variable (if set and exists)
2. `build/Release/redstone_sim.dll`
3. `build/Debug/redstone_sim.dll`
4. `build/redstone_sim.dll`
5. `build/redstone_sim.so` / `build/libredstone_sim.so` (Linux)

## Gymnasium environment

`redstone_sim.env.RedstoneEnv` wraps the simulator as an RL env:

```python
from redstone_sim.env import RedstoneEnv

env = RedstoneEnv(task="connect", world_size=(8, 4, 8))
obs, info = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
```

- **Observation**: `Box(NUM_CHANNELS, X, Y, Z)` float32 in `[0, 1]`
  (one-hot block types + normalized signal + powered flag)
- **Action**: `MultiDiscrete([X*Y*Z, NUM_PLACEABLE, NUM_FACINGS])`
  (flat cell index, block type index, facing index)
- **Reward** (task-dependent): shaped partial credit + terminal bonus
- **Reset**: uses snapshots after the first episode → sub-millisecond

Available tasks (see `env.py::TASKS`):
- `connect` — route signal from lever at (0,1,0) to lamp at (X-1,1,0)

Requires `pip install numpy gymnasium`.

## Smoke tests

```bash
python bindings/smoke_test.py         # core bindings
python bindings/smoke_env_test.py     # Gymnasium env
```

## Next steps

- More tasks: `not`, `and`, `or`, `xor`, `adder`, `latch` — extend `TASKS`
- Curriculum wrapper: sample tasks with weighted distribution to avoid
  catastrophic forgetting
- Dedicated `rs_set_lever_powered()` C helper for clean input toggling
  (needed for tasks with variable inputs like NOT/AND/XOR)
- Zero-copy observation extraction (numpy/torch tensor over GPU memory)
  once the simulator is batched (v0.5)
- PPO training script in `ai/scripts/train.py`
