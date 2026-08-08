"""Gymnasium-compatible RL environment for redstone circuit synthesis.

Single-env (non-batched) implementation. Good enough to prototype the RL loop
and solve tiny tasks (connect, NOT). For serious training the simulator needs
to be batched on GPU (v0.5).

Usage:
    import gymnasium as gym
    from redstone_sim.env import RedstoneEnv

    env = RedstoneEnv(task="connect", world_size=(8, 4, 8))
    obs, info = env.reset()
    for _ in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
"""

from __future__ import annotations

from functools import partial
from typing import Callable, Dict, Tuple, Any, Optional

try:
    import numpy as np
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "redstone_sim.env requires numpy and gymnasium.\n"
        "Install with: pip install numpy gymnasium"
    ) from e

from ._core import (
    World, Snapshot,
    BLOCK_AIR, BLOCK_SOLID, BLOCK_LEVER, BLOCK_LAMP,
    BLOCK_REDSTONE_DUST, BLOCK_REDSTONE_TORCH, BLOCK_REPEATER,
    DIR_DOWN, DIR_UP, DIR_NORTH, DIR_SOUTH, DIR_WEST, DIR_EAST, DIR_NONE,
    FLAG_POWERED,
)


# ── Action space: what the agent can place ───────────────────────────────

PLACEABLE_TYPES = [
    BLOCK_AIR,             # 0: remove / nothing
    BLOCK_SOLID,           # 1: support block
    BLOCK_REDSTONE_DUST,   # 2: wire
    BLOCK_REDSTONE_TORCH,  # 3: inverter
    BLOCK_REPEATER,        # 4: delay/boost
]
NUM_PLACEABLE = len(PLACEABLE_TYPES)

ACTION_FACINGS = [DIR_DOWN, DIR_UP, DIR_NORTH, DIR_SOUTH, DIR_WEST, DIR_EAST, DIR_NONE]
NUM_FACINGS = len(ACTION_FACINGS)


# ── Observation channels ─────────────────────────────────────────────────
#
# One-hot encode the block types the agent might see (including non-placeable
# ones like LEVER, LAMP that are placed by the task setup). Plus scalar
# channels for signal level and flags.

OBS_TYPES = [
    BLOCK_AIR, BLOCK_SOLID, BLOCK_LEVER, BLOCK_LAMP,
    BLOCK_REDSTONE_DUST, BLOCK_REDSTONE_TORCH, BLOCK_REPEATER,
]
OBS_TYPE_TO_CHANNEL: Dict[int, int] = {t: i for i, t in enumerate(OBS_TYPES)}
NUM_OBS_TYPES = len(OBS_TYPES)

# Channels: one-hot types + signal/15 + powered flag
NUM_CHANNELS = NUM_OBS_TYPES + 1 + 1
CH_SIGNAL = NUM_OBS_TYPES
CH_POWERED = NUM_OBS_TYPES + 1


def build_obs(world: "World", world_size: Tuple[int, int, int]) -> np.ndarray:
    """Build the observation tensor for a World in the standard format.

    Shape: (NUM_CHANNELS, X, Y, Z) float32 in [0, 1].
    Channels:
      0..NUM_OBS_TYPES-1  one-hot block type
      CH_SIGNAL           normalized signal level (signal / 15)
      CH_POWERED          POWERED flag as 0.0 or 1.0

    Shared by RedstoneEnv (PPO) and AZEnvAdapter (AlphaZero) so both see
    exactly the same obs layout.
    """
    arr = world.get_all_blocks()  # (Y, Z, X) structured
    sx, sy, sz = world_size
    obs = np.zeros((NUM_CHANNELS, sx, sy, sz), dtype=np.float32)

    types = arr["type"]
    signal = arr["signal"]
    flags = arr["flags"]

    for t, ch in OBS_TYPE_TO_CHANNEL.items():
        mask = (types == t)
        obs[ch] = mask.transpose(2, 0, 1).astype(np.float32)

    obs[CH_SIGNAL] = (signal.astype(np.float32) / 15.0).transpose(2, 0, 1)
    obs[CH_POWERED] = ((flags & FLAG_POWERED) > 0).astype(np.float32).transpose(2, 0, 1)

    return obs


# ── Task base class and registry ─────────────────────────────────────────

class Task:
    """A task defines the setup, validity rules, and reward for an episode."""

    def setup_world(self, world: World) -> None:
        raise NotImplementedError

    def on_episode_start(self, world: Optional[World] = None) -> None:
        """Reset any per-episode bookkeeping (e.g. shaped-reward baselines).

        Called by the env after snapshot restore and initial tick. Tasks
        that use delta-based reward should compute initial progress here
        (from `world`) so the first agent action isn't credited with the
        scaffold's pre-existing progress.
        """
        pass

    def is_valid_placement(self, world: World, x: int, y: int, z: int,
                            block_type: int, facing: int) -> bool:
        """Return True if the agent is allowed to perform this placement.
        Override to enforce per-task rules (protected cells, MC-style
        support requirements, etc.). The default allows everything."""
        return True

    def evaluate(self, world: World, ticks: int) -> Tuple[float, bool]:
        """Run `ticks` game ticks, return (reward, solved)."""
        raise NotImplementedError


class ReachLampTask(Task):
    """Base class for 'route a signal from lever to lamp' tasks.

    Subclasses differ only in WHERE the lever/lamp are placed and which
    block types are allowed. The shared pieces (world setup, placement
    validation, BFS-based shaped reward, delta smoothing) all live here.

    Reward:
      - +1.0 if the lamp has FLAG_POWERED (episode terminates)
      - delta-based shaped: change in (target_dist - closest_dust_to_lamp)
        / target_dist between consecutive steps, where `closest_dust` is
        computed by BFS from the lever through lit dust cells only. This
        excludes isolated signal sources and forces true connectivity.
      - -0.02 per step (step_penalty, applied by the env, not here)
      - -0.05 extra if the action was invalid (applied by env)
    """

    # Default: everything a route-style task might legitimately use.
    # Subclasses can narrow this to constrain the action space.
    ALLOWED_TYPES = (BLOCK_AIR, BLOCK_SOLID, BLOCK_REDSTONE_DUST)

    # What counts as "solid support" for dust placement (mirrors MC:
    # any full opaque block, or a lamp, counts as support).
    SUPPORT_TYPES = (BLOCK_SOLID, BLOCK_LAMP)

    def __init__(
        self,
        world_size: Tuple[int, int, int],
        lever_pos: Tuple[int, int, int],
        lamp_pos: Tuple[int, int, int],
    ):
        sx, sy, sz = world_size
        self.world_size = world_size
        self.lever_pos = lever_pos
        self.lamp_pos = lamp_pos

        lx, ly, lz = lever_pos
        mx, my, mz = lamp_pos
        if not (0 <= lx < sx and 0 <= ly < sy and 0 <= lz < sz):
            raise ValueError(f"lever_pos {lever_pos} out of bounds for {world_size}")
        if not (0 <= mx < sx and 0 <= my < sy and 0 <= mz < sz):
            raise ValueError(f"lamp_pos {lamp_pos} out of bounds for {world_size}")

        # Manhattan distance drives the shaped-reward normalization.
        self.target_dist = abs(mx - lx) + abs(my - ly) + abs(mz - lz)

        # Protect the entire y=0 floor plus lever and lamp cells.
        self.protected = {lever_pos, lamp_pos}
        for x in range(sx):
            for z in range(sz):
                self.protected.add((x, 0, z))

        self._prev_progress = 0.0

    def on_episode_start(self, world: Optional[World] = None) -> None:
        """Compute the initial progress from the current world state so
        that pre-placed scaffold blocks don't leak their progress into
        the agent's first-step reward."""
        if world is None:
            self._prev_progress = 0.0
        else:
            self._prev_progress = self._compute_progress(world)

    def _compute_progress(self, world: World) -> float:
        """Shaped progress measure. Returns a value in [0, 1].

        Two components combined:
          a) BFS from lever through lit dust → closest dust to lamp.
             This is the main reward signal (weight 1.0).
          b) "Staircase potential": solids placed adjacent to lit dust
             whose top is AIR (= could be a next staircase step).
             Scored by how close the top-of-solid cell is to the lamp.
             Given a small weight (0.3) so it nudges exploration toward
             solid placements without dominating the main reward.

        The staircase-potential term is the key to making SOLID
        placements earn a positive delta during training: without it,
        PPO never discovers that a solid on its own (giving 0 lit-dust
        progress) is a useful preparatory action.
        """
        arr = world.get_all_blocks()
        mx, my, mz = self.lamp_pos
        sx, sy, sz = self.world_size
        lx, ly, lz = self.lever_pos

        is_dust = (arr["type"] == BLOCK_REDSTONE_DUST)
        is_solid = (arr["type"] == BLOCK_SOLID)
        is_air = (arr["type"] == BLOCK_AIR)
        lit_dust = is_dust & (arr["signal"] > 0)

        # ── a) BFS through lit dust ───────────────────────────────
        visited = np.zeros_like(lit_dust)
        frontier: list = []
        NEIGHBORS = [(-1, 0, 0), (1, 0, 0), (0, -1, 0),
                     (0, 1, 0), (0, 0, -1), (0, 0, 1)]
        for dx, dy, dz in NEIGHBORS:
            nx, ny, nz = lx + dx, ly + dy, lz + dz
            if 0 <= nx < sx and 0 <= ny < sy and 0 <= nz < sz:
                if lit_dust[ny, nz, nx] and not visited[ny, nz, nx]:
                    visited[ny, nz, nx] = True
                    frontier.append((nx, ny, nz))
        while frontier:
            x, y, z = frontier.pop(0)
            for dx, dy, dz in NEIGHBORS:
                nx, ny, nz = x + dx, y + dy, z + dz
                if not (0 <= nx < sx and 0 <= ny < sy and 0 <= nz < sz):
                    continue
                if lit_dust[ny, nz, nx] and not visited[ny, nz, nx]:
                    visited[ny, nz, nx] = True
                    frontier.append((nx, ny, nz))

        if visited.any():
            ys, zs, xs = np.nonzero(visited)
            dists = np.abs(xs - mx) + np.abs(ys - my) + np.abs(zs - mz)
            closest_lit = int(dists.min())
            main = max(0.0, (self.target_dist - closest_lit) / self.target_dist)
        else:
            closest_lit = self.target_dist + 1  # "no lit dust yet"
            main = 0.0

        # ── b) staircase potential ────────────────────────────────
        # Reward solids adjacent to lit dust whose top is AIR AND
        # strictly closer to the lamp than any currently-lit dust.
        # Strictness prevents reward hacking via spam-placing solids
        # that don't actually advance the frontier; weight 0.55
        # ensures a "proper" step-placement beats the alternative of
        # extending the dust line one more cell horizontally.
        potential = 0.0
        if visited.any():
            ys, zs, xs = np.nonzero(visited)
            for vi in range(len(xs)):
                x, y, z = int(xs[vi]), int(ys[vi]), int(zs[vi])
                for dx, dz in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nx, nz = x + dx, z + dz
                    if not (0 <= nx < sx and 0 <= nz < sz):
                        continue
                    if not is_solid[y, nz, nx]:
                        continue
                    if y + 1 >= sy:
                        continue
                    if not is_air[y + 1, nz, nx]:
                        continue
                    dist_top = (abs(nx - mx) + abs((y + 1) - my)
                                + abs(nz - mz))
                    if dist_top >= closest_lit:
                        continue  # not an actual frontier advance
                    score = max(0.0, (self.target_dist - dist_top)
                                       / self.target_dist)
                    if score > potential:
                        potential = score

        return main + 0.55 * potential

    def setup_world(self, world: World) -> None:
        world.clear()
        sx, sy, sz = self.world_size
        for x in range(sx):
            for z in range(sz):
                world.set_block(x, 0, z, BLOCK_SOLID)
        lx, ly, lz = self.lever_pos
        mx, my, mz = self.lamp_pos
        world.set_block(lx, ly, lz, BLOCK_LEVER, DIR_DOWN)
        world.set_block(mx, my, mz, BLOCK_LAMP)

    def is_valid_placement(self, world, x, y, z, block_type, facing):
        if (x, y, z) in self.protected:
            return False
        if block_type not in self.ALLOWED_TYPES:
            return False
        # MC rule: redstone_wire (dust) requires a full-block support
        # directly below. No floating dust, no vertical stacking.
        if block_type == BLOCK_REDSTONE_DUST:
            if y == 0:
                return False
            below = world.get_block(x, y - 1, z)
            if below.type not in self.SUPPORT_TYPES:
                return False
        return True

    def evaluate(self, world: World, ticks: int) -> Tuple[float, bool]:
        world.tick(ticks)
        arr = world.get_all_blocks()
        mx, my, mz = self.lamp_pos
        lamp_flags = int(arr[my, mz, mx]["flags"])
        if lamp_flags & FLAG_POWERED:
            return 1.0, True

        # Delta-based shaped reward: reward the *change* in connected-
        # dust-closeness-to-lamp. _prev_progress is initialized from the
        # actual initial world state in on_episode_start(), so scaffolds
        # don't contaminate the first step's delta.
        progress = self._compute_progress(world)
        delta = progress - self._prev_progress
        self._prev_progress = progress
        return delta, False


class ConnectTask(ReachLampTask):
    """Horizontal route on flat floor. Lever at (0,1,0), lamp at (N,1,0).

    SOLID is not allowed — on a flat pre-filled floor there's no reason
    to stack supports, and allowing SOLID just enables useless towers.
    Optimal solution: N - 1 dust blocks in a straight line.
    """

    ALLOWED_TYPES = (BLOCK_AIR, BLOCK_REDSTONE_DUST)

    def __init__(
        self,
        world_size: Tuple[int, int, int],
        distance: Optional[int] = None,
    ):
        sx, _, _ = world_size
        if distance is None:
            distance = sx - 1
        if not (1 <= distance <= sx - 1):
            raise ValueError(
                f"distance={distance} out of range [1, {sx - 1}] for {world_size}"
            )
        super().__init__(world_size, (0, 1, 0), (distance, 1, 0))


class StairTask(ReachLampTask):
    """Forced staircase: a gap in the floor directly under the lamp
    prevents the 'dust carpet' shortcut, so the agent must actually
    learn the diagonal staircase pattern.

    Layout for height H (= vertical distance from lever to lamp):
        lever at (0, 1, 0)
        lamp  at (H + 2, H + 1, 0)
        floor: SOLID everywhere except (H + 2, 0, 0) which is AIR

    Optimal solution (2H + 1 placements):
        dust at (1, 1, 0), solid at (2, 1, 0), dust at (2, 2, 0),
        ...alternating solid + dust climbing diagonally...
        last dust adjacent to the lamp.

    Scaffold: `pre_placed` lets us pre-insert some of the solution so
    curriculum training can start from a task the agent can almost
    solve already and progressively remove hints. Pre-placed cells are
    automatically added to `protected` so the agent can't remove them.
    """

    ALLOWED_TYPES = (BLOCK_AIR, BLOCK_SOLID, BLOCK_REDSTONE_DUST)

    def __init__(
        self,
        world_size: Tuple[int, int, int],
        height: int = 1,
        pre_placed=None,      # list of (x, y, z, block_type, facing)
    ):
        sx, sy, sz = world_size
        if height < 1:
            raise ValueError(f"height must be >= 1, got {height}")
        lamp = (height + 2, height + 1, 0)
        if lamp[1] >= sy or lamp[0] >= sx:
            raise ValueError(
                f"height={height} too large for world {world_size}"
            )
        super().__init__(world_size, (0, 1, 0), lamp)
        self.height = height
        self._floor_gap = (lamp[0], 0, lamp[2])
        self.pre_placed = list(pre_placed or [])
        # Anything pre-placed is sacred — the agent can't overwrite it.
        for (x, y, z, _t, _f) in self.pre_placed:
            self.protected.add((x, y, z))

    def setup_world(self, world: World) -> None:
        world.clear()
        sx, sy, sz = self.world_size
        for x in range(sx):
            for z in range(sz):
                if (x, 0, z) == self._floor_gap:
                    continue
                world.set_block(x, 0, z, BLOCK_SOLID)
        lx, ly, lz = self.lever_pos
        mx, my, mz = self.lamp_pos
        world.set_block(lx, ly, lz, BLOCK_LEVER, DIR_DOWN)
        world.set_block(mx, my, mz, BLOCK_LAMP)
        # Pre-placed scaffold goes in AFTER lever/lamp
        for (x, y, z, t, f) in self.pre_placed:
            world.set_block(x, y, z, t, f)


class ClimbTask(ReachLampTask):
    """Diagonal staircase up. Lever at (0,1,0), lamp at (H+2, H+1, 0).

    The agent must build a redstone staircase using SOLID "step" blocks
    and DUST on top of each step. In MC, dust climbs diagonally: dust
    at (x, y, z) connects up to dust at (x+1, y+1, z) when (x+1, y, z)
    is solid and (x, y+1, z) is air.

    Optimal solution for height H:
        dust at (1, 1, 0)           -- adjacent to lever
        solid at (2, 1, 0)           -- step
        dust at (2, 2, 0)            -- on step, connects up to first dust
        solid at (3, 2, 0)           -- next step  (only if H >= 2)
        dust at (3, 3, 0)            -- on next step
        ...
    i.e. 2*H + 1 placements total; final dust adjacent to lamp at
    (H+2, H+1, 0).

    Height constraint: lamp at y = H+1 requires world_size.y > H+1.
    In an 8x4x8 world (y: 0..3), max is H = 2.
    """

    ALLOWED_TYPES = (BLOCK_AIR, BLOCK_SOLID, BLOCK_REDSTONE_DUST)

    def __init__(
        self,
        world_size: Tuple[int, int, int],
        height: int = 1,
    ):
        sx, sy, sz = world_size
        if height < 1:
            raise ValueError(f"height must be >= 1, got {height}")
        lamp = (height + 2, height + 1, 0)
        if lamp[1] >= sy or lamp[0] >= sx:
            raise ValueError(
                f"height={height} too large for world {world_size} "
                f"(need y >= {height + 2} and x >= {height + 3})"
            )
        super().__init__(world_size, (0, 1, 0), lamp)
        self.height = height


class TruthTableTask(Task):
    """Multi-lever combinational circuit synthesis task.

    The agent builds a circuit with N input levers and 1 output lamp.
    Evaluation cycles through all 2^N input combinations, toggling
    levers and reading the lamp. Reward = fraction of rows that match
    the target truth table.

    Unlike ReachLampTask, this task has NO shaped reward. It returns
    reward only at terminal (end of episode) — perfect for AlphaZero
    which handles sparse terminal reward well. `evaluate()` returns
    (0.0, False) during normal steps; call `evaluate_final(world)`
    explicitly to get the terminal reward.

    Args:
        world_size:       (X, Y, Z) of the world grid.
        lever_positions:  list of (x, y, z) — the input levers. Each must
                          be y >= 1 with a solid below (the floor provides it).
        lamp_pos:         (x, y, z) where the output lamp goes.
        truth_table:      list of 0/1 with length 2^len(lever_positions).
                          Indexed MSB-first: table[a*2^(N-1) + b*2^(N-2) + ...]
                          is the expected lamp output for inputs (a, b, ...).
        lever_facings:    optional list of Direction per lever (defaults DIR_DOWN).
        ticks_per_row:    ticks to simulate per input combination.
    """

    ALLOWED_TYPES = (
        BLOCK_AIR, BLOCK_SOLID, BLOCK_REDSTONE_DUST,
        BLOCK_REDSTONE_TORCH, BLOCK_REPEATER,
    )
    SUPPORT_TYPES = (BLOCK_SOLID, BLOCK_LAMP)

    def __init__(
        self,
        world_size: Tuple[int, int, int],
        lever_positions: list,
        lamp_pos: Tuple[int, int, int],
        truth_table: list,
        lever_facings: Optional[list] = None,
        ticks_per_row: int = 40,
    ):
        sx, sy, sz = world_size
        self.world_size = world_size
        self.lever_positions = [tuple(p) for p in lever_positions]
        self.lamp_pos = tuple(lamp_pos)
        self.truth_table = list(truth_table)
        self.ticks_per_row = ticks_per_row

        n_inputs = len(self.lever_positions)
        if len(self.truth_table) != (1 << n_inputs):
            raise ValueError(
                f"truth_table length {len(self.truth_table)} != 2^{n_inputs}"
            )

        if lever_facings is None:
            self.lever_facings = [DIR_DOWN] * n_inputs
        else:
            if len(lever_facings) != n_inputs:
                raise ValueError("lever_facings length must match lever_positions")
            self.lever_facings = list(lever_facings)

        # Validate bounds
        for p in self.lever_positions + [self.lamp_pos]:
            x, y, z = p
            if not (0 <= x < sx and 0 <= y < sy and 0 <= z < sz):
                raise ValueError(f"position {p} out of bounds for {world_size}")

        # Protect floor + levers + lamp
        self.protected = set(self.lever_positions) | {self.lamp_pos}
        for x in range(sx):
            for z in range(sz):
                self.protected.add((x, 0, z))

        # For compatibility with base Task — target_dist is only used by
        # some progress metrics which we don't compute; set to a nominal
        # manhattan distance from first lever to lamp.
        lx, ly, lz = self.lever_positions[0]
        mx, my, mz = self.lamp_pos
        self.target_dist = abs(mx - lx) + abs(my - ly) + abs(mz - lz)

    def setup_world(self, world: World) -> None:
        world.clear()
        sx, sy, sz = self.world_size
        # Fill floor with SOLID (support for dust everywhere)
        for x in range(sx):
            for z in range(sz):
                world.set_block(x, 0, z, BLOCK_SOLID)
        # Place lamp
        mx, my, mz = self.lamp_pos
        world.set_block(mx, my, mz, BLOCK_LAMP)
        # Place all levers OFF (canonical snapshot has no input active)
        for (lx, ly, lz), facing in zip(self.lever_positions, self.lever_facings):
            world.set_block(lx, ly, lz, BLOCK_LEVER, facing)
            world.set_lever_powered(lx, ly, lz, False)

    def is_valid_placement(self, world, x, y, z, block_type, facing):
        if (x, y, z) in self.protected:
            return False
        if block_type not in self.ALLOWED_TYPES:
            return False
        # Dust needs solid support below
        if block_type == BLOCK_REDSTONE_DUST:
            if y == 0:
                return False
            below = world.get_block(x, y - 1, z)
            if below.type not in self.SUPPORT_TYPES:
                return False
        # Repeater likewise needs solid below
        if block_type == BLOCK_REPEATER:
            if y == 0:
                return False
            below = world.get_block(x, y - 1, z)
            if below.type not in self.SUPPORT_TYPES:
                return False
        return True

    def evaluate(self, world: World, ticks: int) -> Tuple[float, bool]:
        """Per-step evaluation. Returns zero reward — AZ uses terminal only.

        Advances the simulation one step so the env's observation reflects
        the agent's latest placement, but returns 0 reward. The real reward
        is computed by `evaluate_final` at the end of the episode.
        """
        world.tick(ticks)
        return 0.0, False

    def evaluate_final(self, world: World) -> Tuple[float, bool]:
        """Terminal evaluation: cycle all 2^N input combinations, check lamp.

        IMPORTANT: This mutates the world's lever states. The caller must
        snapshot/restore if further simulation is needed afterwards.

        Returns (reward in [0, 1], solved=True if all rows correct).
        """
        n_inputs = len(self.lever_positions)
        matches = 0
        # Snapshot the circuit so we can reset lever states between rows
        circuit_snap = world.save()
        for row in range(1 << n_inputs):
            world.restore(circuit_snap)
            for i, (lx, ly, lz) in enumerate(self.lever_positions):
                # MSB-first indexing: input i occupies bit (n_inputs - 1 - i)
                bit = (row >> (n_inputs - 1 - i)) & 1
                world.set_lever_powered(lx, ly, lz, bool(bit))
            world.tick(self.ticks_per_row)
            mx, my, mz = self.lamp_pos
            actual = 1 if world.is_powered(mx, my, mz) else 0
            if actual == self.truth_table[row]:
                matches += 1
        circuit_snap.close()

        reward = matches / float(1 << n_inputs)
        solved = (matches == (1 << n_inputs))
        return reward, solved


TASKS: Dict[str, Callable[[Tuple[int, int, int]], Task]] = {
    # --- Horizontal routing (flat floor, dust only) ---
    # Lever at (0,1,0); lamp at (N, 1, 0). Optimal: N-1 dust in a line.
    "connect_3": partial(ConnectTask, distance=3),
    "connect_4": partial(ConnectTask, distance=4),
    "connect_5": partial(ConnectTask, distance=5),
    "connect_6": partial(ConnectTask, distance=6),
    "connect_7": partial(ConnectTask, distance=7),
    "connect":   ConnectTask,  # alias → lamp at far corner

    # --- Vertical staircase — easy variant (carpet shortcut works) ---
    # Height H: lamp at (H+2, H+1, 0). Optimal: 2H+1 placements.
    "climb_1":   partial(ClimbTask, height=1),   # 3 placements
    "climb_2":   partial(ClimbTask, height=2),   # 5 placements

    # --- Vertical staircase — forced (no carpet shortcut) ---
    # Same geometry as climb_N but floor cell under lamp is AIR, so the
    # "dust beneath lamp" trick fails and the agent must staircase.
    "stair_1":   partial(StairTask, height=1),   # optimal: 3 placements
    "stair_2":   partial(StairTask, height=2),   # optimal: 5 placements

    # --- Scaffolded variants of stair_1 (curriculum) ---
    # Geometry identical to stair_1 (lamp at (3,2,0), floor gap at (3,0,0)),
    # but with increasingly more of the optimal solution pre-placed. The
    # agent only needs to provide the remaining placements.
    #
    # stair_1_finish — pre-place dust(1,1) + solid(2,1); agent places dust(2,2).
    #                  Optimal: 1 placement. Teaches "dust on elevated solid".
    "stair_1_finish": partial(
        StairTask, height=1,
        pre_placed=[
            (1, 1, 0, BLOCK_REDSTONE_DUST, DIR_NONE),
            (2, 1, 0, BLOCK_SOLID,         DIR_NONE),
        ],
    ),
    # stair_1_half — pre-place dust(1,1); agent places solid(2,1) + dust(2,2).
    #                Optimal: 2 placements. Teaches "place step + dust on top".
    "stair_1_half": partial(
        StairTask, height=1,
        pre_placed=[
            (1, 1, 0, BLOCK_REDSTONE_DUST, DIR_NONE),
        ],
    ),

    # --- Truth-table circuits (for AlphaZero-style training) ---
    # All share the same geometry: 2 levers at opposite corners of the
    # floor, 1 lamp on the far side. The agent must build a circuit whose
    # output lamp matches the target truth table across all 2^N input
    # combinations. Reward is terminal only.
    #
    # 10x5x10 world gives enough room for torch-based XOR/AND circuits.
    "or_2in":  partial(
        TruthTableTask,
        lever_positions=[(0, 1, 0), (0, 1, 9)],
        lamp_pos=(9, 1, 4),
        truth_table=[0, 1, 1, 1],
    ),
    "and_2in": partial(
        TruthTableTask,
        lever_positions=[(0, 1, 0), (0, 1, 9)],
        lamp_pos=(9, 1, 4),
        truth_table=[0, 0, 0, 1],
    ),
    "xor_2in": partial(
        TruthTableTask,
        lever_positions=[(0, 1, 0), (0, 1, 9)],
        lamp_pos=(9, 1, 4),
        truth_table=[0, 1, 1, 0],
    ),
    "not_1in": partial(
        TruthTableTask,
        lever_positions=[(0, 1, 0)],
        lamp_pos=(5, 1, 0),
        truth_table=[1, 0],
    ),
}


# ── The environment ──────────────────────────────────────────────────────

class RedstoneEnv(gym.Env):
    """Single-env Gymnasium wrapper over the redstone simulator.

    Observation: Box((NUM_CHANNELS, X, Y, Z), float32 in [0, 1])
    Action:      MultiDiscrete([X*Y*Z, NUM_PLACEABLE, NUM_FACINGS])
                 (flat cell index, block type index, facing index)

    Reward: task-defined per-step reward + optional step penalty.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        task: str = "connect",
        world_size: Tuple[int, int, int] = (8, 4, 8),
        max_steps: int = 20,
        ticks_per_eval: int = 20,
        step_penalty: float = 0.02,
        invalid_penalty: float = 0.05,
    ):
        super().__init__()
        if task not in TASKS:
            raise ValueError(f"Unknown task {task!r}. Choose from {list(TASKS)}")

        self.world_size = world_size
        self.max_steps = max_steps
        self.ticks_per_eval = ticks_per_eval
        self.step_penalty = step_penalty
        self.invalid_penalty = invalid_penalty

        self.task: Task = TASKS[task](world_size)

        sx, sy, sz = world_size
        self.n_cells = sx * sy * sz

        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(NUM_CHANNELS, sx, sy, sz),
            dtype=np.float32,
        )
        self.action_space = spaces.MultiDiscrete([
            self.n_cells,
            NUM_PLACEABLE,
            NUM_FACINGS,
        ])

        self.world = World(size=world_size)
        self._initial_snap: Optional[Snapshot] = None
        self._step_count: int = 0

    # ── lifecycle helpers ─────────────────────────────────────────

    def close(self):
        if self._initial_snap is not None:
            self._initial_snap.close()
            self._initial_snap = None
        self.world.destroy()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # ── coord <-> flat index ───────────────────────────────────────

    def _unflatten(self, idx: int) -> Tuple[int, int, int]:
        sx, sy, sz = self.world_size
        x = idx % sx
        z = (idx // sx) % sz
        y = idx // (sx * sz)
        return x, y, z

    # ── observation builder ────────────────────────────────────────

    def _build_obs(self) -> np.ndarray:
        return build_obs(self.world, self.world_size)

    # ── Gymnasium API ──────────────────────────────────────────────

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        if self._initial_snap is None:
            # First reset: build the world from scratch, snapshot it
            self.task.setup_world(self.world)
            self.world.tick(self.ticks_per_eval)  # converge initial state
            self._initial_snap = self.world.save()
        else:
            # Fast reset via snapshot
            self.world.restore(self._initial_snap)

        self.task.on_episode_start(self.world)
        self._step_count = 0
        return self._build_obs(), {}

    def step(self, action):
        idx, type_choice, facing_choice = int(action[0]), int(action[1]), int(action[2])
        x, y, z = self._unflatten(idx)
        block_type = PLACEABLE_TYPES[type_choice]
        facing = ACTION_FACINGS[facing_choice]

        info: Dict[str, Any] = {}
        # Sync before validity check so the task sees the latest GPU state
        # (support checks read the block below the proposed placement).
        self.world.sync()
        invalid = not self.task.is_valid_placement(
            self.world, x, y, z, block_type, facing
        )

        if not invalid:
            self.world.set_block(x, y, z, block_type, facing)

        self._step_count += 1

        # Evaluate (runs ticks internally)
        task_reward, solved = self.task.evaluate(self.world, self.ticks_per_eval)

        reward = task_reward - self.step_penalty
        if invalid:
            reward -= self.invalid_penalty

        terminated = solved
        truncated = self._step_count >= self.max_steps

        info["valid"] = not invalid
        info["solved"] = solved
        info["step"] = self._step_count
        info["action_pos"] = (x, y, z)

        return self._build_obs(), float(reward), terminated, truncated, info


# ── MultiTaskEnv: sample a different task each episode ───────────────────


class MultiTaskEnv(RedstoneEnv):
    """Gymnasium env that samples a task at reset-time from a weighted
    distribution. Prevents catastrophic forgetting when training on
    multiple skills in one network.

    All tasks must share the same `world_size` (so obs/action spaces
    match). Per-task initial snapshots are cached on first use, so
    subsequent resets are as fast as single-task RedstoneEnv.

    Usage:
        env = MultiTaskEnv(
            task_mix=[("connect_3", 0.2), ("connect_5", 0.3),
                      ("climb_1", 0.25),  ("climb_2", 0.25)],
            world_size=(8, 4, 8),
        )
    """

    def __init__(
        self,
        task_mix,                                # list[(task_name, weight)]
        world_size: Tuple[int, int, int] = (8, 4, 8),
        max_steps: int = 20,
        ticks_per_eval: int = 20,
        step_penalty: float = 0.02,
        invalid_penalty: float = 0.05,
    ):
        # Bypass RedstoneEnv.__init__ because we don't want a single
        # self.task — we'll set it per-episode in reset().
        gym.Env.__init__(self)

        if not task_mix:
            raise ValueError("task_mix must be non-empty")

        self.world_size = world_size
        self.max_steps = max_steps
        self.ticks_per_eval = ticks_per_eval
        self.step_penalty = step_penalty
        self.invalid_penalty = invalid_penalty

        # Instantiate all tasks up front (each validates its own config)
        self._task_names = []
        self._task_objs = []
        self._weights = []
        for name, w in task_mix:
            if name not in TASKS:
                raise ValueError(
                    f"Unknown task {name!r}. Choose from {list(TASKS)}"
                )
            self._task_names.append(name)
            self._task_objs.append(TASKS[name](world_size))
            self._weights.append(float(w))
        total = sum(self._weights)
        self._probs = np.array(self._weights, dtype=np.float64) / total

        sx, sy, sz = world_size
        self.n_cells = sx * sy * sz
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(NUM_CHANNELS, sx, sy, sz),
            dtype=np.float32,
        )
        self.action_space = spaces.MultiDiscrete([
            self.n_cells,
            NUM_PLACEABLE,
            NUM_FACINGS,
        ])

        self.world = World(size=world_size)

        # Lazy per-task initial snapshots. Populated on first reset to
        # each task so all later resets restore in ~0.2 ms.
        self._snaps: Dict[str, Snapshot] = {}

        self._initial_snap = None       # placeholder so close() doesn't fail
        self._step_count = 0
        self.task: Optional[Task] = None
        self.current_task_name: Optional[str] = None

    # ── Gymnasium API ──────────────────────────────────────────────

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        # Seed Gym's internal RNG (we use numpy.random separately).
        gym.Env.reset(self, seed=seed)

        # Sample a task for this episode
        idx = int(np.random.choice(len(self._task_objs), p=self._probs))
        self.task = self._task_objs[idx]
        self.current_task_name = self._task_names[idx]

        # Lazy-init this task's initial snapshot
        if self.current_task_name not in self._snaps:
            self.task.setup_world(self.world)
            self.world.tick(self.ticks_per_eval)
            self._snaps[self.current_task_name] = self.world.save()
        else:
            self.world.restore(self._snaps[self.current_task_name])

        self.task.on_episode_start(self.world)
        self._step_count = 0
        return self._build_obs(), {"task": self.current_task_name}

    def close(self):
        for snap in self._snaps.values():
            snap.close()
        self._snaps.clear()
        self.world.destroy()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def parse_task_mix(spec: str) -> list:
    """Parse a CLI-friendly task-mix spec like:

        "connect_3=0.2,connect_5=0.3,climb_1=0.25,climb_2=0.25"

    into a list of (name, weight) tuples suitable for MultiTaskEnv.
    Weights need not sum to 1 — they get normalized internally.
    """
    mix = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" in chunk:
            name, w = chunk.split("=", 1)
            mix.append((name.strip(), float(w)))
        else:
            mix.append((chunk, 1.0))
    if not mix:
        raise ValueError(f"Could not parse task mix: {spec!r}")
    return mix
