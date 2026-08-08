# mcredstone-cuda

A GPU-accelerated Minecraft redstone simulator, written in CUDA. It reproduces
Minecraft 1.21 signal propagation — strong/weak power, conductor relay, torch
inversion, repeater delay and locking, dust auto-cross — and can simulate
**thousands of independent circuits in a single kernel launch**.

That last part is the point. A 16³ world is 4096 cells: about 0.1% of an
RTX 3060 Ti. Simulating one world at a time is pure launch latency — measured
at 92.7 µs per tick, essentially independent of world size. Batching N worlds
per launch brings that down to **0.32 µs per tick per world**:

| | µs/tick/world | circuits/s | speedup |
| --- | --- | --- | --- |
| `World` (one per launch) | 92.68 | 270 | 1× |
| `WorldBatch`, batch 128 | 0.416 | 60,043 | 223× |
| `WorldBatch`, batch 512 | 0.326 | 76,782 | 285× |
| `WorldBatch`, batch 2048 | 0.320 | 78,124 | **290×** |

End-to-end truth-table evaluation — set the levers, run to convergence, read the
lamp, for every input row — comes out at **319×** (`World` 7547 µs/circuit vs
`WorldBatch[512]` 23.7 µs/circuit). Reproduce with `python tools/bench_batch.py`;
the numbers above are from an RTX 3060 Ti.

If you want to search a large space of redstone circuits — brute force,
reinforcement learning, genetic search, automated verification — that ratio is
the difference between a feasible experiment and an infeasible one.

## Build

Requires a CUDA Toolkit, CMake 3.18+, and a C/C++ toolchain (Visual Studio 2022
on Windows).

```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=86   # 86 = RTX 30-series
cmake --build build --config Release
```

Set `CMAKE_CUDA_ARCHITECTURES` to your GPU's compute capability (`nvidia-smi
--query-gpu=compute_cap --format=csv` will tell you; drop the dot, so 8.9 → 89).
It defaults to 86 if you omit it.

CMake's `native` is deliberately not the default: on CMake 4.3 with CUDA 13.2 it
resolved to 75 on an sm_86 card. The build still runs — the embedded PTX gets
JIT-compiled — but it quietly leaves performance behind, which is worse than a
default you set on purpose.

> **Windows note**: build somewhere with a short path. MSBuild writes `.tlog`
> files deep inside the build tree and will fail with `MSB6003` /
> `DirectoryNotFoundException` once the total path exceeds `MAX_PATH` (260).

This produces, under `build/Release/`:

| Target | What it does |
| --- | --- |
| `redstone-sim` | Loads a XOR gate built in Minecraft, runs all 4 input combinations. Should print `ALL CORRECT`. |
| `redstone-test` | 72 unit tests covering dust, torches, repeaters, lamps, probes, snapshots. |
| `dump_sim` | Loads a `.schem`, runs N ticks, dumps final block state as text (for diffing). |
| `redstone_sim` | Shared library used by the Python bindings. |

## Quick start (Python)

```bash
pip install numpy
```

```python
import sys; sys.path.insert(0, "bindings")
from redstone_sim import (
    World, BLOCK_LEVER, BLOCK_LAMP, BLOCK_REDSTONE_DUST, BLOCK_SOLID, DIR_DOWN,
)

with World(size=(8, 4, 8)) as w:
    w.set_block(0, 0, 0, BLOCK_SOLID)
    w.set_block(0, 1, 0, BLOCK_LEVER, facing=DIR_DOWN)
    for z in range(1, 7):
        w.set_block(0, 0, z, BLOCK_SOLID)          # floor to carry the wire
        w.set_block(0, 1, z, BLOCK_REDSTONE_DUST)
    w.set_block(0, 0, 7, BLOCK_SOLID)
    w.set_block(0, 1, 7, BLOCK_LAMP)

    w.tick(10)
    print("lamp lit:", w.is_powered(0, 1, 7))
```

### Batched simulation

For anything beyond a handful of circuits, use `WorldBatch`. Semantics are
identical to `World` — same device code, same tick order — but N worlds advance
per launch:

```python
from redstone_sim import WorldBatch

b = WorldBatch(n_worlds=2048, size=(16, 16, 16))
for i, circuit in enumerate(circuits):
    for (x, y, z, block_type, facing) in circuit:
        b.set_block(i, x, y, z, block_type, facing=facing)

used = b.tick_until_stable(max_ticks=40)
```

`tick_until_stable` returns the ticks actually consumed. An instance is
considered *done* when a whole tick changed nothing **and** no block holds a
pending tile tick — a genuine fixed point, so it is frozen and skipped by
subsequent launches. If `used == max_ticks`, some circuit never settled, which
is how you detect clocks and torch loops.

See `bindings/README.md` for the full API, including snapshots, probes and the
truth-table evaluator.

## How it works

**Block representation.** Every cell is an 8-byte `Block`: type, signal level,
facing, flags, a type-specific 16-bit state word, and the game tick at which its
next scheduled event fires. A `static_assert` pins the size. Scheduled events
live *inside* the block rather than in a global queue, which means "are there
pending tile ticks?" is a reduction over the grid instead of shared mutable
state — a much better fit for a GPU.

**The tick.** Each tick runs Minecraft's four phases in order:

1. **Tile ticks** — scheduled events fire (torch toggles, repeater output).
2. **Block updates** — signal propagation, double-buffered (read A, write B).
3. **Block events** — piston movement (not yet implemented).
4. **Scheduling** — detect component input changes, schedule future tile ticks.

**Signal rules.** The core mirrors Minecraft's two signal queries: `getSignal`
(weak power emitted toward a direction) and `getDirectSignal` (strong power
delivered into a solid block). The rule that makes everything else work: when a
conductor is queried, it returns the max of its own weak output and the strongest
power it receives on any of its six faces — that is how solid blocks relay strong
power outward as weak power.

**One copy of the rules, two launch shapes.** All per-cell logic lives in
`src/kernels/signal_device.cuh` and `src/kernels/components_device.cuh`. Both the
single-world kernels and the batched kernels call the same device functions;
neither reimplements a rule. `tests/test_batch_equivalence.py` asserts the two
paths agree bit-for-bit.

## Fidelity, honestly

This is faithful enough to run circuits built in real Minecraft — the test suite
includes XOR gates exported from the game — but it is not a bit-perfect
reimplementation. Known differences:

- **Wire propagation is iterative, not instantaneous.** One tick is one
  relaxation pass over the grid, so a signal advances one dust cell per tick.
  Minecraft propagates dust instantly within a tick. Final states and
  component delays (torch 2 ticks, repeater 1–4) are faithful; wire latency is
  not. In practice you run a circuit to its fixed point and read the result,
  which is what `tick_until_stable` is for.
- **Lamps and solid blocks read power differently.** Lamps apply the
  conductor relay; plain solid blocks don't. This is not derived from Minecraft's
  code — it is calibrated against transient states in the exported reference
  circuits. See the comment in `signal_device.cuh`.
- **Dust "dot" mode is not modelled.** Isolated dust always auto-crosses.
  Tracking real dot mode needs a per-dust flag imported from the schematic
  palette.
- **Comparators are declared but behave as repeaters.** No compare/subtract
  modes, no container reading.
- **Pistons, observers, rails and note blocks are stubs.** The block IDs and
  quasi-connectivity plumbing exist; the behaviour does not.

## Validation

- `redstone-test` — 72 unit tests on individual components and rules.
- `tests/test_batch_equivalence.py` — `WorldBatch` vs `World`, bit-exact.
- `tests/golden_runner.py` — replays circuits exported from Minecraft under
  `schematics/golden/` and fails on any divergence.
- `tests/validate_xor_truth.py` — runs every golden XOR through all four input
  combinations and checks the truth table.
- `tools/compare_schem.py` — diffs simulator output against a schematic
  block-by-block. When a circuit misbehaves, the first diverging block usually
  localizes the bug.

```bash
build/Release/redstone-test
python tests/test_batch_equivalence.py
python tests/golden_runner.py
python tools/bench_batch.py          # World vs WorldBatch throughput
```

## Layout

```
include/redstone/     Public C API — types, world, batch, block registry, kernels
src/
  world.cu            World: GPU double buffering, snapshots, probes, schem I/O
  block_registry.c    Data-driven block property tables
  schem_import.c      Sponge Schematic v2 reader (gzip + NBT)
  schem_export.c      Sponge Schematic v2 writer
  kernels/
    signal_device.cuh      Shared per-cell signal logic
    components_device.cuh  Shared per-cell torch/repeater logic
    signal.cu              Single-world propagation kernel
    components.cu          Single-world component kernels
    batch.cu               WorldBatch: batched kernels + host implementation
    movement.cu            Piston movement (stub)
bindings/             Python package (ctypes, numpy only)
tests/                Unit tests, equivalence checks, golden replays
tools/                Debug and benchmarking utilities
schematics/golden/    Reference circuits exported from Minecraft
third_party/miniz.*   Single-file gzip library (public domain)
```

## Design notes

- **Direction convention**: `facing` is the direction of the *attached* block,
  which is the opposite of Minecraft's `FACING` property for torches and levers.
  The schematic importer flips accordingly.
- **`toward_dir` in kernels**: direction *from source to asker* — the opposite of
  Minecraft's `getSignal(direction)` convention. Getting this backwards is the
  single most likely source of bugs when editing the rules; the header comments
  spell out each translation.
- **Host↔GPU sync**: `set_block` writes only the host mirror and sets a dirty
  flag; the next `tick()` uploads. Any API that reads GPU state must honour that
  flag. `WorldBatch` differs — `set_levers` writes device memory directly via a
  kernel precisely so evaluation loops don't trigger a full re-upload per row.
  Read `include/redstone/batch.h` before mixing them.

## License

MIT. Bundles miniz, which is public domain.
