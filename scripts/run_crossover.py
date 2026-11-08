"""RQ6 -- at what training-set size n does Split-CP catch up with CV+?

Sweeps n_train on a handful of representative datasets for Split vs CV+ (K=10),
so the paper can state a concrete crossover point rather than an impression.

    python scripts/run_crossover.py --n-trains 100 200 400 800 1600
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conformal_tabpfn.config import Cell, load_experiment_config  # noqa: E402
from conformal_tabpfn.experiment import run_cells  # noqa: E402
from conformal_tabpfn.models import warmup  # noqa: E402
from run_experiments import add_common_args, apply_overrides, resolve_datasets  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--n-trains", nargs="*", type=int, default=None)
    args = ap.parse_args()

    cfg = load_experiment_config(args.config)
    apply_overrides(cfg, args)
    n_trains = args.n_trains or cfg.crossover_n_trains or [100, 200, 400, 800, 1600]
    methods = args.cp_methods or cfg.crossover_methods or ["split_0.3", "cvplus_K10"]
    datasets = args.datasets or cfg.crossover_datasets or cfg.datasets
    models = args.models or ["tabpfn"]

    cfg.datasets = datasets
    sizes = resolve_datasets(cfg, datasets)

    cells: list[Cell] = []
    skipped: list[tuple[str, int]] = []
    for ds, n_avail in sizes.items():
        # Available training pool after the test split is carved out.
        pool = n_avail - min(round(cfg.test_frac * n_avail), cfg.test_cap)
        for n_train in n_trains:
            if n_train > pool:
                # Capped at available data, as the spec requires -- and recorded
                # rather than silently substituted.
                skipped.append((ds, n_train))
                continue
            for model in models:
                for cp in methods:
                    for seed in cfg.seeds:
                        cells.append(
                            Cell(
                                dataset=ds, model=model, cp_method=cp,
                                alpha=float(cfg.alphas[0]), seed=int(seed),
                                n_train=int(n_train), study="crossover",
                            )
                        )
    print(f"RQ6 crossover: {len(sizes)} datasets x {len(n_trains)} n_train x "
          f"{len(methods)} methods x {len(models)} models x {len(cfg.seeds)} seeds")
    print(f"total cells: {len(cells)}")
    if skipped:
        print(f"skipped (n_train exceeds the available pool): {skipped}")

    if args.dry_run:
        run_cells(cells, cfg, device=args.device, dry_run=True, limit=args.limit)
        return 0

    for model in models:
        warmup(model, device=args.device)
    summary = run_cells(cells, cfg, device=args.device, limit=args.limit)
    print(summary.report())
    return 1 if summary.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
