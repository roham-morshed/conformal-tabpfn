"""Merge the per-cell result parquets into one file for fast analysis.

The per-cell files under ``results/raw/`` are the resume ledger and stay exactly
where they are -- this only writes an additional consolidated copy so that
downstream analysis does not have to reopen ~15k small files every time.

    python scripts/consolidate_results.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RAW = ROOT / "results" / "raw"
OUT = ROOT / "results" / "all_results.parquet"


def main() -> int:
    files = sorted(RAW.glob("*.parquet"))
    if not files:
        print(f"no result files in {RAW}")
        return 1
    t0 = time.time()
    frames = []
    for i, f in enumerate(files, 1):
        frames.append(pd.read_parquet(f))
        if i % 2000 == 0:
            print(f"  read {i}/{len(files)} ...", flush=True)
    df = pd.concat(frames, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(
        f"consolidated {len(files)} cells -> {len(df)} rows into "
        f"{OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e6:.1f} MB) "
        f"in {time.time() - t0:.0f}s"
    )
    print("\nrows per study:")
    print(df.groupby("study").agg(cells=("cell_hash", "nunique"), rows=("coverage", "size")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
