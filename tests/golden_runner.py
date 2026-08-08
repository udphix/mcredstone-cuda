"""Golden-set runner: compare the simulator against MC-validated .schem
circuits. Exits non-zero if any divergence.

Usage:
    python tests/golden_runner.py                 # all schems under golden set
    python tests/golden_runner.py xor             # only files matching 'xor'
    python tests/golden_runner.py xor --ticks 80
    python tests/golden_runner.py --dir schematics/golden/xor
"""
import argparse
import glob
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_one(schem_path: Path, ticks: int) -> tuple[bool, str]:
    """Return (pass, summary) for a single schem."""
    out = subprocess.run(
        [sys.executable, str(ROOT / "tools/compare_schem.py"),
         str(schem_path), str(ticks)],
        capture_output=True, text=True,
        env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"},
    )
    if out.returncode != 0 and not out.stdout:
        return False, f"script error: {out.stderr.strip()[:200]}"

    # Parse divergences count from stdout.
    last_line = ""
    divergences = 0
    for line in out.stdout.splitlines():
        if "divergences" in line:
            # e.g. "✗ 6 divergences:"
            parts = line.split()
            for p in parts:
                if p.isdigit():
                    divergences = int(p)
                    break
        if line.startswith("OVERALL:"):
            last_line = line.split(":", 1)[1].strip()
    return (last_line == "PASS"), f"{last_line} ({divergences} diffs)" if last_line else "no-result"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("filter", nargs="?", default="",
                    help="substring filter, e.g. 'xor'")
    ap.add_argument("--ticks", type=int, default=40)
    ap.add_argument("--dir", default="schematics/golden")
    args = ap.parse_args()

    schem_dir = ROOT / args.dir
    all_schems = sorted(glob.glob(str(schem_dir / "**/*.schem"), recursive=True))
    filtered = [s for s in all_schems if args.filter in Path(s).name]
    if not filtered:
        print(f"No schems matching '{args.filter}' in {schem_dir}")
        return 2

    print(f"=== Golden runner: {len(filtered)} schem(s), ticks={args.ticks} ===")
    passed = 0
    failed: list[str] = []
    for s in filtered:
        name = Path(s).name
        ok, summary = run_one(Path(s), args.ticks)
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {name}: {summary}")
        if ok:
            passed += 1
        else:
            failed.append(name)

    total = len(filtered)
    print(f"\n=== {passed}/{total} passed ===")
    if failed:
        print("Failures:")
        for f in failed:
            print(f"  - {f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
