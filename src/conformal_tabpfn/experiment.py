"""Grid enumeration and the resumable, failure-tolerant cell driver.

Guarantees implemented here:
  * every cell writes its own parquet, so a crash loses at most one cell;
  * a completed cell is skipped on restart (deterministic hash of the cell key);
  * a failing cell records exception + traceback under ``results/failures/`` and
    the run continues, with a loud summary at the end;
  * nothing is ever written as NaN without a recorded reason.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import pandas as pd

from .config import Cell, ExperimentConfig, parse_cp_method
from .data import Dataset, load_dataset
from .runners import run_classification_cell
from .utils import atomic_write_parquet, cell_hash, package_versions, set_all_seeds, write_failure

__all__ = [
    "RunSummary",
    "enumerate_cells",
    "execute_cell",
    "load_all_results",
    "run_cells",
]


def enumerate_cells(
    cfg: ExperimentConfig,
    dataset_sizes: dict[str, int],
) -> list[Cell]:
    """All main-grid cells, with jackknife+ gated on ``n_train <= jackknife_max_n``."""
    cells: list[Cell] = []
    for ds in cfg.datasets:
        n_avail = dataset_sizes.get(ds)
        if n_avail is None:
            continue
        # The training pool is what is left after the test split is carved out;
        # jackknife+ must be gated on THAT, not on the full dataset size.
        pool = n_avail - min(round(cfg.test_frac * n_avail), cfg.test_cap)
        n_train = pool if cfg.n_train is None else min(cfg.n_train, pool)
        for model in cfg.models:
            for cp in cfg.cp_methods:
                method = parse_cp_method(cp)
                if method.family == "jackknife_plus" and n_train > cfg.jackknife_max_n:
                    continue
                for seed in cfg.seeds:
                    cells.append(
                        Cell(
                            dataset=ds, model=model, cp_method=cp,
                            alpha=float(cfg.alphas[0]), seed=int(seed), n_train=int(n_train),
                        )
                    )
    return cells


def _result_path(cfg: ExperimentConfig, cell: Cell) -> Path:
    return Path(cfg.raw_dir) / f"{cell_hash(cell.key())}.parquet"


def execute_cell(
    cell: Cell,
    dataset: Dataset,
    cfg: ExperimentConfig,
    device: str | None = None,
) -> list[dict[str, Any]]:
    """Run one cell with full seeding.  Raises on failure; the caller records it."""
    set_all_seeds(cell.seed)
    rows = run_classification_cell(
        dataset=dataset,
        model_name=cell.model,
        cp_method=cell.cp_method,
        seed=cell.seed,
        alphas=list(cfg.alphas),
        scores=list(cfg.scores),
        n_train=cell.n_train,
        test_frac=cfg.test_frac,
        test_cap=cfg.test_cap,
        chunk_size=cfg.chunk_size,
        device=device,
        quantile_method=cfg.quantile_method,
        raps_lambda=cfg.raps_lambda,
        raps_k_reg=cfg.raps_k_reg,
        imbalance=cell.imbalance,
        study=cell.study,
        minority_frac=cell.minority_frac,
    )
    versions = package_versions()
    for row in rows:
        row["cell_hash"] = cell_hash(cell.key())
        row["run_timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        for pkg, ver in versions.items():
            row[f"ver_{pkg.replace('-', '_')}"] = ver
    return rows


class RunSummary:
    """Counters plus the loud end-of-run report the spec requires."""

    def __init__(self) -> None:
        self.done = 0
        self.skipped = 0
        self.failed = 0
        self.failures: list[tuple[Cell, str]] = []
        self.started = time.time()
        self.cell_seconds: list[float] = []

    @property
    def elapsed(self) -> float:
        return time.time() - self.started

    def eta(self, remaining: int) -> float:
        if not self.cell_seconds:
            return float("nan")
        return float(sum(self.cell_seconds) / len(self.cell_seconds) * remaining)

    def report(self) -> str:
        lines = [
            "",
            "=" * 78,
            f"RUN SUMMARY   done={self.done}  skipped={self.skipped}  failed={self.failed}",
            f"wall clock: {self.elapsed / 60:.1f} min",
        ]
        if self.cell_seconds:
            avg = sum(self.cell_seconds) / len(self.cell_seconds)
            lines.append(f"mean seconds per executed cell: {avg:.2f}")
        if self.failures:
            lines.append("")
            lines.append(f"!!! {len(self.failures)} FAILED CELLS -- see results/failures/ !!!")
            for cell, err in self.failures[:20]:
                lines.append(
                    f"  {cell.dataset}/{cell.model}/{cell.cp_method}/seed{cell.seed}: {err[:90]}"
                )
            if len(self.failures) > 20:
                lines.append(f"  ... and {len(self.failures) - 20} more")
        else:
            lines.append("no failures")
        lines.append("=" * 78)
        return "\n".join(lines)


def run_cells(
    cells: Iterable[Cell],
    cfg: ExperimentConfig,
    device: str | None = None,
    dry_run: bool = False,
    limit: int | None = None,
    progress_every: int = 1,
) -> RunSummary:
    """Execute cells sequentially, skipping completed ones and recording failures."""
    cells = list(cells)
    if limit is not None:
        cells = cells[:limit]
    Path(cfg.raw_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.failures_dir).mkdir(parents=True, exist_ok=True)

    summary = RunSummary()
    cache: dict[str, Dataset] = {}
    total = len(cells)

    for i, cell in enumerate(cells, start=1):
        path = _result_path(cfg, cell)
        if path.exists():
            summary.skipped += 1
            continue
        if dry_run:
            print(f"[dry-run] would run {cell.dataset}/{cell.model}/{cell.cp_method}/s{cell.seed}")
            summary.done += 1
            continue

        if cell.dataset not in cache:
            cache[cell.dataset] = load_dataset(cell.dataset)
        t0 = time.perf_counter()
        try:
            rows = execute_cell(cell, cache[cell.dataset], cfg, device=device)
            atomic_write_parquet(pd.DataFrame(rows), path)
            summary.done += 1
            summary.cell_seconds.append(time.perf_counter() - t0)
        except Exception as exc:  # by design: never lose a whole run to one cell
            write_failure(cfg.failures_dir, cell.key(), exc)
            summary.failed += 1
            summary.failures.append((cell, f"{type(exc).__name__}: {exc}"))
            print(
                f"  FAILED {cell.dataset}/{cell.model}/{cell.cp_method}/s{cell.seed}: "
                f"{type(exc).__name__}: {str(exc)[:120]}",
                file=sys.stderr, flush=True,
            )

        if progress_every and i % progress_every == 0:
            remaining = total - i
            eta = summary.eta(remaining)
            eta_txt = f"{eta / 60:6.1f} min" if eta == eta else "   ? min"
            print(
                f"[{i:>5}/{total}] done={summary.done} skip={summary.skipped} "
                f"fail={summary.failed}  eta {eta_txt}   "
                f"{cell.dataset}/{cell.model}/{cell.cp_method}/s{cell.seed}",
                flush=True,
            )
    return summary


def load_all_results(raw_dir: str | Path) -> pd.DataFrame:
    """Concatenate every result parquet into one DataFrame."""
    files = sorted(Path(raw_dir).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No results in {raw_dir}. Run `python scripts/run_experiments.py` first."
        )
    frames = [pd.read_parquet(f) for f in files]
    return pd.concat(frames, ignore_index=True)


def iter_dataset_sizes(keys: Iterable[str]) -> Iterator[tuple[str, int]]:
    for key in keys:
        try:
            yield key, load_dataset(key).n
        except FileNotFoundError:
            continue
