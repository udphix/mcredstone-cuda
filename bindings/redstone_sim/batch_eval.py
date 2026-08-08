"""Batched truth-table evaluation on top of `WorldBatch`.

The single-world evaluation loop that `TruthTableTask.evaluate_final` and
`tools/functional_eval.py` run is:

    for row in range(2**n):        # per circuit
        restore(snapshot); set levers; tick(40); read lamp

which costs one launch-bound `World` per circuit. Batched, the same loop
runs once for ALL circuits at the same time:

    for row in range(2**n):        # for the whole batch
        batch.restore(snap); batch.set_levers(...); batch.tick(...); probes

Everything inside the row loop is a single GPU operation regardless of how
many circuits are in flight, so a group of K sampled circuits costs about
the same as one.

Example:
    from redstone_sim import WorldBatch
    from redstone_sim.batch_eval import TruthTableSpec, evaluate_truth_tables

    batch = WorldBatch(n_worlds=len(circuits), size=(16, 16, 16))
    ...build each instance...
    specs = [TruthTableSpec(levers=[(0,1,0)], lamp=(9,1,4), table=[0,1])
             for _ in circuits]
    rates, solved = evaluate_truth_tables(batch, specs)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

try:
    import numpy as np
except ImportError as e:  # pragma: no cover
    raise ImportError("redstone_sim.batch_eval requires numpy") from e

from ._core import WorldBatch, FLAG_POWERED


@dataclass
class TruthTableSpec:
    """One circuit's evaluation contract.

    levers: input lever positions, MSB first (matches TruthTableTask's
            row indexing: input i occupies bit (len(levers) - 1 - i)).
    lamp:   the output cell to read.
    table:  expected lamp output per row, length 2**len(levers).
    """
    levers: Sequence[Tuple[int, int, int]]
    lamp: Tuple[int, int, int]
    table: Sequence[int]

    def __post_init__(self):
        expected = 1 << len(self.levers)
        if len(self.table) != expected:
            raise ValueError(
                f"truth table length {len(self.table)} != 2^{len(self.levers)}"
            )


def evaluate_truth_tables(
    batch: WorldBatch,
    specs: List[TruthTableSpec],
    ticks_per_row: int = 40,
    until_stable: bool = True,
    check_every: int = 4,
):
    """Evaluate every instance against its truth table.

    Returns (match_rates, solved):
      match_rates: float array (n_worlds,) — fraction of rows that matched.
      solved:      bool array (n_worlds,)  — every row matched.

    `until_stable=True` uses convergence detection and stops as soon as all
    instances settle, falling back to `ticks_per_row` as the cap. Circuits
    that never settle (clocks, torch loops) consume the full budget and are
    read at that point, matching the fixed-tick behaviour.

    NOTE: this mutates lever states and leaves the batch on the last row's
    state. It snapshots internally and restores per row, so the circuit
    layout itself is preserved.
    """
    if len(specs) != batch.n_worlds:
        raise ValueError(
            f"got {len(specs)} specs for {batch.n_worlds} instances"
        )

    n_worlds = batch.n_worlds
    max_rows = max(1 << len(s.levers) for s in specs)
    matches = np.zeros(n_worlds, dtype=np.int32)
    n_rows = np.array([1 << len(s.levers) for s in specs], dtype=np.int32)

    # Lamps never move — register the probes once.
    batch.set_probes([(i, *s.lamp) for i, s in enumerate(specs)])

    snap = batch.save()
    try:
        for row in range(max_rows):
            batch.restore(snap)

            entries = []
            for i, s in enumerate(specs):
                n = len(s.levers)
                if row >= (1 << n):
                    continue          # this circuit has fewer input rows
                for k, (lx, ly, lz) in enumerate(s.levers):
                    bit = (row >> (n - 1 - k)) & 1
                    entries.append((i, lx, ly, lz, bool(bit)))
            if entries:
                batch.set_levers(entries)

            if until_stable:
                batch.tick_until_stable(max_ticks=ticks_per_row,
                                        check_every=check_every)
            else:
                batch.tick(ticks_per_row)

            lit = np.fromiter(
                ((p.flags & FLAG_POWERED) != 0 for p in batch.read_probes()),
                dtype=bool, count=n_worlds,
            )
            for i, s in enumerate(specs):
                if row < (1 << len(s.levers)) and int(lit[i]) == s.table[row]:
                    matches[i] += 1
    finally:
        snap.close()

    rates = matches.astype(np.float32) / n_rows
    return rates, matches == n_rows
