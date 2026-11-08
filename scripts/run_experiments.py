"""Main experiment driver (spec section 6).  Resumable, failure-tolerant, offline.

    python scripts/run_experiments.py --dry-run
    python scripts/run_experiments.py --calibrate 20      # measure, then estimate
    python scripts/run_experiments.py --models tabpfn --datasets wdbc breast-w
    python scripts/run_experiments.py                     # the full grid
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conformal_tabpfn.config import load_experiment_config, parse_cp_method  # noqa: E402
from conformal_tabpfn.data import DatasetMissingError, load_dataset  # noqa: E402
from conformal_tabpfn.experiment import enumerate_cells, run_cells  # noqa: E402
from conformal_tabpfn.models import gpu_available, warmup  # noqa: E402
from conformal_tabpfn.utils import cell_hash, environment_info  # noqa: E402


def add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--config", default=None)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--cp-methods", nargs="*", default=None)
    ap.add_argument("--seeds", type=int, default=None, help="use seeds 0..N-1")
    ap.add_argument("--n-train", type=int, default=None, help="cap the training pool")
    ap.add_argument("--test-cap", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--device", default=None, help="force cpu / cuda")


def resolve_datasets(cfg, requested: list[str] | None) -> dict[str, int]:
    """Keep only datasets that are actually cached; report the ones that are not."""
    keys = requested if requested else cfg.datasets
    sizes: dict[str, int] = {}
    missing: list[str] = []
    for key in keys:
        try:
            sizes[key] = load_dataset(key).n
        except (DatasetMissingError, FileNotFoundError):
            missing.append(key)
    if missing:
        print(f"!! not cached, skipping: {missing}")
        print("   run `python scripts/download_assets.py` while connected to add them.")
    return sizes


def apply_overrides(cfg, args) -> None:
    if args.datasets:
        cfg.datasets = args.datasets
    if args.models:
        cfg.models = args.models
    if args.cp_methods:
        cfg.cp_methods = args.cp_methods
    if args.seeds is not None:
        cfg.seeds = list(range(args.seeds))
    if args.n_train is not None:
        cfg.n_train = args.n_train
    if args.test_cap is not None:
        cfg.test_cap = args.test_cap
    if getattr(args, "jackknife_max_n", None) is not None:
        cfg.jackknife_max_n = args.jackknife_max_n


def estimate_from_calibration(
    summary, remaining_cells, per_family: dict[tuple[str, str], list[float]]
) -> None:
    """Extrapolate the full-grid runtime from measured cells, by cp-family.

    A flat mean would be badly wrong: a K=20 cell costs ~20x a split cell, so we
    scale each remaining cell by the family it belongs to and fall back to the
    global mean only for families we have not timed yet.
    """
    if not summary.cell_seconds:
        print("no cells were executed; cannot estimate")
        return
    overall = sum(summary.cell_seconds) / len(summary.cell_seconds)
    print("\nmeasured seconds per cell, by cp_method:")
    means: dict[str, float] = {}
    for name, vals in sorted(per_family.items()):
        means[name] = sum(vals) / len(vals)
        print(f"  {name:<22} {means[name]:>8.2f} s   (n={len(vals)})")

    total = 0.0
    unknown = 0
    for cell in remaining_cells:
        if cell.cp_method in means:
            total += means[cell.cp_method]
        else:
            total += overall
            unknown += 1
    print(f"\nremaining cells: {len(remaining_cells)}  ({unknown} using the global mean)")
    print(f"ESTIMATED REMAINING RUNTIME: {total / 3600:.1f} hours ({total / 60:.0f} min)")


def enumerate_imbalance_cells(cfg, sizes: dict[str, int]) -> list:
    """RQ5 cells: class-imbalanced variants of a few datasets.

    These must NOT collide with the main grid, so every cell carries
    ``study="imbalance"`` and its own ``minority_frac``; both are part of the
    cell key, hence part of the result hash.  ``runners`` does the actual
    subsampling via ``data.make_imbalanced`` and records the *realised* class
    counts, because the requested ratio is not evidence of what happened.

    Only the four methods the fairness comparison needs are run: each marginal
    method paired with its Mondrian counterpart.
    """
    from conformal_tabpfn.config import Cell

    datasets = cfg.imbalance_datasets or cfg.datasets[:5]
    fracs = cfg.imbalance_ratios or [0.05, 0.1, 0.2]
    methods = ["split_0.3", "mondrian_split_0.3", "cvplus_K10", "mondrian_cvplus_K10"]

    cells = []
    for ds in datasets:
        n_avail = sizes.get(ds)
        if n_avail is None:
            continue
        pool = n_avail - min(round(cfg.test_frac * n_avail), cfg.test_cap)
        n_train = pool if cfg.n_train is None else min(cfg.n_train, pool)
        for frac in fracs:
            for model in cfg.models:
                for cp in methods:
                    for seed in cfg.seeds:
                        cells.append(
                            Cell(
                                dataset=ds, model=model, cp_method=cp,
                                alpha=float(cfg.alphas[0]), seed=int(seed),
                                n_train=int(n_train),
                                imbalance=f"minority_{frac}",
                                study="imbalance",
                                minority_frac=float(frac),
                            )
                        )
    return cells


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--jackknife-max-n", type=int, default=None,
                    help="override jackknife_max_n; needed to run the jackknife+ arm, "
                         "which the config disables by default because K = n_train")
    ap.add_argument("--calibrate", type=int, default=None,
                    help="run N cells, then print a runtime estimate and stop")
    ap.add_argument("--imbalance", action="store_true",
                    help="RQ5: rerun the grid on class-imbalanced variants")
    args = ap.parse_args()

    cfg = load_experiment_config(args.config)
    apply_overrides(cfg, args)

    env = environment_info()
    print("=" * 78)
    print(f"device: {'cuda' if gpu_available() else 'CPU'}  gpu={env['gpu_name'] or 'none'}")
    print(f"packages: {env['packages']}")
    print(f"quantile_method: {cfg.quantile_method}")
    print("=" * 78)

    sizes = resolve_datasets(cfg, cfg.datasets)
    cfg.datasets = [d for d in cfg.datasets if d in sizes]

    if args.imbalance:
        cells = enumerate_imbalance_cells(cfg, sizes)
    else:
        cells = enumerate_cells(cfg, sizes)
    print(f"\ngrid: {len(cfg.datasets)} datasets x {len(cfg.models)} models x "
          f"{len(cfg.cp_methods)} cp_methods x {len(cfg.seeds)} seeds")
    print(f"total cells: {len(cells)}  "
          f"(each emits {len(cfg.scores)} scores x {len(cfg.alphas)} alphas = "
          f"{len(cfg.scores) * len(cfg.alphas)} rows)")
    done = sum(1 for c in cells if (Path(cfg.raw_dir) / f"{cell_hash(c.key())}.parquet").exists())
    print(f"already complete: {done}   remaining: {len(cells) - done}")

    by_family: dict[str, int] = {}
    for c in cells:
        by_family[c.cp_method] = by_family.get(c.cp_method, 0) + 1
    for name, count in sorted(by_family.items()):
        print(f"  {name:<22} {count:>6} cells   (family {parse_cp_method(name).family})")

    if args.dry_run:
        run_cells(cells, cfg, device=args.device, dry_run=True, limit=args.limit)
        return 0

    if not args.calibrate:
        print("\nwarming up models (one-off init excluded from all timings) ...")
        for model in cfg.models:
            try:
                warmup(model, device=args.device)
                print(f"  {model} ready")
            except Exception as exc:  # a broken model must not kill the run
                print(f"  {model} WARM-UP FAILED: {type(exc).__name__}: {exc}")

    todo = list(cells)
    if args.calibrate:
        print(f"\n--- CALIBRATION: running {args.calibrate} representative cells ---")
        # Spread the sample across cp_methods so the estimate is not dominated by
        # whichever family happens to come first in the grid order.
        combos = {(c.model, c.cp_method) for c in todo}
        per_combo = max(1, args.calibrate // max(1, len(combos)))
        picked, seen = [], {}
        for c in todo:
            key = (c.model, c.cp_method)
            if seen.get(key, 0) < per_combo:
                picked.append(c)
                seen[key] = seen.get(key, 0) + 1
        todo = picked[: args.calibrate]
        for model in {c.model for c in todo}:
            warmup(model, device=args.device)

    summary = run_cells(todo, cfg, device=args.device, limit=args.limit)
    print(summary.report())

    if args.calibrate:
        per_family: dict[tuple[str, str], list[float]] = {}
        for c, secs in zip(todo, summary.cell_seconds):
            per_family.setdefault((c.model, c.cp_method), []).append(secs)
        remaining = [c for c in cells if c not in todo]
        estimate_from_calibration(summary, remaining, per_family)
    return 1 if summary.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
