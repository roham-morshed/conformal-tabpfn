"""60-second end-to-end sanity check.  This is what you run first.

One small dataset x TabPFN x {split, cvplus_K5} x 2 seeds, printed as a readable
table.  It exercises the whole path -- cache load, per-fold preprocessing, model
fit, score matrices, conformal sets, metrics -- so a green smoke test means the
plumbing works even before the full grid runs.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conformal_tabpfn.data import available_datasets, load_dataset  # noqa: E402
from conformal_tabpfn.models import gpu_available, tabpfn_available, warmup  # noqa: E402
from conformal_tabpfn.runners import run_classification_cell  # noqa: E402
from conformal_tabpfn.utils import environment_info, set_all_seeds  # noqa: E402

HEADER = (
    f"{'dataset':<14}{'model':<9}{'cp_method':<16}{'score':<7}{'alpha':>6}{'seed':>5}"
    f"{'cov':>8}{'size':>7}{'ccgap':>7}{'fit_s':>8}{'pred_s':>8}{'dev':>6}"
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="wdbc")
    ap.add_argument("--model", default="tabpfn")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--n-train", type=int, default=200)
    ap.add_argument("--test-cap", type=int, default=150)
    args = ap.parse_args()

    env = environment_info()
    print(f"python {env['python']}  |  cuda_available={env['cuda_available']}  "
          f"gpu={env['gpu_name'] or 'none'}")
    print(f"tabpfn={env['packages'].get('tabpfn')}  torch={env['packages'].get('torch')}  "
          f"sklearn={env['packages'].get('scikit-learn')}")

    cached = available_datasets()
    if args.dataset not in cached:
        print(f"\nERROR: {args.dataset!r} is not cached. Available: {cached}")
        print("Run `python scripts/download_assets.py` while connected.")
        return 1
    model = args.model
    if model == "tabpfn" and not tabpfn_available():
        print("\nTabPFN unavailable; falling back to rf for the smoke test.")
        model = "rf"

    ds = load_dataset(args.dataset)
    print(f"\ndataset: {ds.describe()}")
    print(f"device for this run: {'cuda' if gpu_available() else 'cpu'}\n")

    print("warming up the model (one-off init excluded from all timings) ...")
    warmup(model)

    print("\n" + HEADER)
    print("-" * len(HEADER))
    t0 = time.perf_counter()
    all_rows = []
    for cp_method in ("split_0.3", "cvplus_K5"):
        for seed in range(args.seeds):
            set_all_seeds(seed)
            rows = run_classification_cell(
                dataset=ds, model_name=model, cp_method=cp_method, seed=seed,
                alphas=[0.1], scores=["lac", "aps"], n_train=args.n_train,
                test_cap=args.test_cap,
            )
            all_rows.extend(rows)
            for r in rows:
                print(
                    f"{r['dataset']:<14}{r['model']:<9}{r['cp_method']:<16}{r['score']:<7}"
                    f"{r['alpha']:>6.2f}{r['seed']:>5}{r['coverage']:>8.3f}"
                    f"{r['avg_set_size']:>7.2f}{r['cc_coverage_gap']:>7.3f}"
                    f"{r['fit_time']:>8.3f}{r['predict_time']:>8.2f}"
                    f"{str(r['device_used'])[:5]:>6}"
                )
    elapsed = time.perf_counter() - t0

    print("-" * len(HEADER))
    print(f"{len(all_rows)} rows in {elapsed:.1f}s\n")

    # Aggregate the headline contrast the paper is about.
    print(f"{'cp_method':<16}{'mean cov':>10}{'cov sd':>9}{'mean size':>11}"
          f"{'fit_s':>9}{'predict_s':>11}{'n_fits':>8}")
    print("-" * 74)
    for cp_method in ("split_0.3", "cvplus_K5"):
        sub = [r for r in all_rows if r["cp_method"] == cp_method]
        cov = np.array([r["coverage"] for r in sub])
        print(
            f"{cp_method:<16}{cov.mean():>10.4f}{cov.std(ddof=1):>9.4f}"
            f"{np.mean([r['avg_set_size'] for r in sub]):>11.3f}"
            f"{np.mean([r['fit_time'] for r in sub]):>9.4f}"
            f"{np.mean([r['predict_time'] for r in sub]):>11.3f}"
            f"{int(np.mean([r['n_fits'] for r in sub])):>8}"
        )

    problems = []
    for r in all_rows:
        if not (0.0 <= r["coverage"] <= 1.0):
            problems.append(f"coverage out of range: {r}")
        if r["avg_set_size"] <= 0:
            problems.append(f"degenerate empty sets: {r['cp_method']}/{r['score']}")
        if r["coverage"] < 1 - r["alpha"] - 0.15:
            problems.append(
                f"suspiciously low coverage {r['coverage']:.3f} for "
                f"{r['cp_method']}/{r['score']} (nominal {1 - r['alpha']:.2f})"
            )
    if problems:
        print("\n!! SMOKE TEST WARNINGS !!")
        for p in problems:
            print("  " + p)
        return 1
    print("\nsmoke test OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
