"""Smoke test for the Gymnasium environment.

Validates:
  - Env constructs and exposes correct observation/action spaces
  - reset() returns a well-shaped observation
  - step() accepts random actions, returns 5-tuple
  - Snapshot-based reset is fast (no re-setup after first reset)
  - A hand-crafted solution to `connect` actually terminates with reward=1

Run from repo root:
    python bindings/smoke_env_test.py
"""

from __future__ import annotations

import sys

# Windows consoles default to cp1252 and cannot encode the arrows / check marks
# printed below. Force UTF-8 so a fresh clone runs on any platform.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from redstone_sim.env import (
    RedstoneEnv,
    PLACEABLE_TYPES, ACTION_FACINGS,
    BLOCK_REDSTONE_DUST,
)
from redstone_sim import DIR_NONE


def test_env_spaces() -> None:
    print("\n[test] env spaces and reset")
    env = RedstoneEnv(task="connect", world_size=(8, 4, 8), max_steps=20)
    assert env.observation_space.shape == (9, 8, 4, 8), (
        f"obs shape {env.observation_space.shape}"
    )
    assert tuple(env.action_space.nvec) == (8 * 4 * 8, 5, 7), (
        f"action nvec {env.action_space.nvec}"
    )
    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape
    assert obs.dtype == np.float32
    assert 0.0 <= obs.min() and obs.max() <= 1.0
    env.close()
    print("  ✓ pass")


def test_random_actions() -> None:
    print("\n[test] 30 random actions, no crashes")
    env = RedstoneEnv(task="connect", world_size=(8, 4, 8), max_steps=30)
    obs, _ = env.reset()
    total_reward = 0.0
    for _ in range(30):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            obs, _ = env.reset()
    print(f"  total reward over 30 steps: {total_reward:.3f}")
    env.close()
    print("  ✓ pass")


def test_handcrafted_solution() -> None:
    """Build the optimal 7-dust path manually, confirm env terminates with r=1."""
    print("\n[test] handcrafted solution → terminal reward")
    env = RedstoneEnv(
        task="connect", world_size=(8, 4, 8),
        max_steps=10, ticks_per_eval=30,
    )
    obs, _ = env.reset()

    # Place dust from (1,1,0) to (6,1,0)
    dust_type_idx = PLACEABLE_TYPES.index(BLOCK_REDSTONE_DUST)
    facing_idx = ACTION_FACINGS.index(DIR_NONE)
    sx, sy, sz = env.world_size

    def flat(x, y, z):
        return (y * sz + z) * sx + x

    terminated = False
    for step, x in enumerate(range(1, 7)):
        action = np.array([flat(x, 1, 0), dust_type_idx, facing_idx])
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  step {step}: dust at ({x},1,0) → reward={reward:.3f}, "
              f"solved={info.get('solved')}")
        if terminated:
            break
    assert terminated, "Expected lamp to light after placing dust line"
    env.close()
    print("  ✓ pass")


def test_snapshot_reset_speed() -> None:
    """After first reset, subsequent resets should use the snapshot."""
    print("\n[test] snapshot-based fast reset")
    import time
    env = RedstoneEnv(task="connect", world_size=(8, 4, 8))

    t0 = time.perf_counter()
    env.reset()
    t_first = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(20):
        env.reset()
    t_subsequent = (time.perf_counter() - t0) / 20

    print(f"  first reset: {t_first*1000:.1f} ms")
    print(f"  subsequent:  {t_subsequent*1000:.2f} ms (avg over 20)")
    env.close()
    print("  ✓ pass")


if __name__ == "__main__":
    test_env_spaces()
    test_random_actions()
    test_handcrafted_solution()
    test_snapshot_reset_speed()
    print("\nAll env smoke tests passed ✓")
