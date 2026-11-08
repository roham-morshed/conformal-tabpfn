"""RQ3 -- the context/calibration budget trade-off for Split-CP.

For an in-context learner every calibration point is a point *removed from the
context*, so the usual "more calibration is better" intuition may not hold.  We
sweep the calibration fraction rho and look for rho* (the efficiency optimum) and
whether it sits in a different place for TabPFN than for a gradient-trained model.

    python scripts/run_budget_sweep.py --models tabpfn xgboost
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
    ap.add_argument("--cal-fracs", nargs="*", type=float, default=None)
    args = ap.parse_args()

    cfg = load_experiment_config(args.config)
    apply_overrides(cfg, args)
    fracs = args.cal_fracs or cfg.budget_cal_fracs or [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    models = args.models or cfg.budget_models or ["tabpfn", "xgboost"]

    sizes = resolve_datasets(cfg, cfg.datasets)
    cells: list[Cell] = []
    for ds, n_avail in sizes.items():
        n_train = n_avail if cfg.n_train is None else min(cfg.n_train, n_avail)
        for model in models:
            for rho in fracs:
                for seed in cfg.seeds:
                    cells.append(
                        Cell(
                            dataset=ds, model=model, cp_method=f"split_{rho}",
                            alpha=float(cfg.alphas[0]), seed=int(seed),
                            n_train=int(n_train), study="budget",
                        )
                    )
    print(f"RQ3 budget sweep: {len(sizes)} datasets x {len(models)} models x "
          f"{len(fracs)} rho x {len(cfg.seeds)} seeds = {len(cells)} cells")
    print(f"rho grid: {fracs}")

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
