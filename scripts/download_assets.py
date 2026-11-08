"""THE ONLY NETWORKED SCRIPT IN THIS REPOSITORY.

Fetches every OpenML dataset into ``data/cache/`` as parquet + meta.json, and
warms the TabPFN weight cache by running one tiny fit/predict per estimator.
After this script has succeeded once, the whole experiment pipeline runs with
the network fully disabled.

Usage::

    python scripts/download_assets.py                 # everything
    python scripts/download_assets.py --skip-models   # datasets only
    python scripts/download_assets.py --only breast-w diabetes
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CACHE = ROOT / "data" / "cache"
CONFIGS = ROOT / "configs"

MAX_RETRIES = 3
BACKOFF_BASE = 4.0  # seconds; 4, 8, 16 -- OpenML here is flaky, not fast

# sklearn's fetch_openml exposes no timeout and will block forever on a stalled
# socket, which turns "flaky network" into "hung script".  A process-wide default
# converts that into an ordinary exception the retry loop can handle.
SOCKET_TIMEOUT = float(os.environ.get("OPENML_SOCKET_TIMEOUT", "45"))
socket.setdefaulttimeout(SOCKET_TIMEOUT)


def _retry(fn, what: str, retries: int = MAX_RETRIES):
    """Call ``fn`` with exponential backoff.  Returns (result, error_or_None)."""
    last: BaseException | None = None
    for attempt in range(1, retries + 1):
        try:
            return fn(), None
        except Exception as exc:
            last = exc
            wait = BACKOFF_BASE * (2 ** (attempt - 1))
            print(
                f"    attempt {attempt}/{retries} for {what} failed: "
                f"{type(exc).__name__}: {str(exc)[:120]}",
                flush=True,
            )
            if attempt < retries:
                print(f"    backing off {wait:.0f}s ...", flush=True)
                time.sleep(wait)
    return None, last


def _fetch_openml(data_id: int | None, name: str | None):
    from sklearn.datasets import fetch_openml

    kwargs: dict[str, Any] = {"as_frame": True, "parser": "auto"}
    if data_id is not None:
        return fetch_openml(data_id=int(data_id), **kwargs)
    return fetch_openml(name=name, version=1, **kwargs)


def _sha256_frame(frame: pd.DataFrame) -> str:
    """Content hash of the cached table, for the reproducibility record."""
    return hashlib.sha256(
        pd.util.hash_pandas_object(frame, index=True).values.tobytes()
    ).hexdigest()


def download_classification(entry: dict[str, Any]) -> dict[str, Any]:
    """Fetch, encode and cache one classification dataset.  Returns a report row."""
    from sklearn.preprocessing import LabelEncoder

    key = entry["key"]
    primary = entry.get("openml_id")
    print(f"  {key} (openml id={primary})", flush=True)

    bunch, err = _retry(lambda: _fetch_openml(primary, None), f"{key} id={primary}")
    substituted_from = None
    if bunch is None and entry.get("fallback_name"):
        print(f"    id {primary} unusable; trying name={entry['fallback_name']!r}", flush=True)
        bunch, err = _retry(
            lambda: _fetch_openml(None, entry["fallback_name"]),
            f"{key} name={entry['fallback_name']}",
        )
        if bunch is not None:
            substituted_from = primary
    if bunch is None:
        return {
            "key": key,
            "status": "FAILED",
            "error": f"{type(err).__name__}: {err}",
        }

    X = bunch.data.copy()
    y_raw = bunch.target

    # Drop columns that are entirely missing: they carry no signal and break
    # some estimators.  Recorded so the paper can state it.
    all_nan = [c for c in X.columns if X[c].isna().all()]
    if all_nan:
        X = X.drop(columns=all_nan)

    categorical = [
        c
        for c in X.columns
        if str(X[c].dtype) in ("category", "object", "bool") or X[c].dtype == object
    ]
    # Categoricals are stored as plain strings; ordinal encoding happens
    # per-fold inside the pipeline, never here (that would leak).
    for c in categorical:
        X[c] = X[c].astype("string").fillna("__missing__").astype(str)
    for c in X.columns:
        if c not in categorical:
            X[c] = pd.to_numeric(X[c], errors="coerce").astype(np.float64)

    encoder = LabelEncoder()
    y = encoder.fit_transform(np.asarray(y_raw).astype(str))
    class_names = [str(c) for c in encoder.classes_]

    frame = X.copy()
    target_col = "__target__"
    frame[target_col] = y

    CACHE.mkdir(parents=True, exist_ok=True)
    parquet_path = CACHE / f"{key}.parquet"
    frame.to_parquet(parquet_path, index=False)

    counts = {int(v): int(c) for v, c in zip(*np.unique(y, return_counts=True))}
    expected = entry.get("expected", {})
    mismatches = []
    if expected.get("n") and int(X.shape[0]) != int(expected["n"]):
        mismatches.append(f"n {X.shape[0]} != expected {expected['n']}")
    if expected.get("d") and int(X.shape[1]) != int(expected["d"]):
        mismatches.append(f"d {X.shape[1]} != expected {expected['d']}")
    if expected.get("classes") and len(class_names) != int(expected["classes"]):
        mismatches.append(f"classes {len(class_names)} != expected {expected['classes']}")

    meta = {
        "key": key,
        "task": "classification",
        "openml_id": primary,
        "substituted_from_openml_id": substituted_from,
        "source_url": f"https://www.openml.org/d/{primary}",
        "n": int(X.shape[0]),
        "d": int(X.shape[1]),
        "n_classes": len(class_names),
        "class_names": class_names,
        "class_counts": counts,
        "class_balance": {k: round(v / len(y), 6) for k, v in counts.items()},
        "imbalance_ratio": round(max(counts.values()) / min(counts.values()), 4),
        "categorical_columns": categorical,
        "numeric_columns": [c for c in X.columns if c not in categorical],
        "dropped_all_nan_columns": all_nan,
        "target_column": target_col,
        "n_missing_cells": int(X.isna().sum().sum()),
        "sha256": _sha256_frame(frame),
        "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "expected": expected,
        "shape_mismatches": mismatches,
    }
    (CACHE / f"{key}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return {
        "key": key,
        "status": "OK",
        "n": meta["n"],
        "d": meta["d"],
        "classes": meta["n_classes"],
        "imbalance": meta["imbalance_ratio"],
        "missing": meta["n_missing_cells"],
        "bytes": parquet_path.stat().st_size,
        "substituted_from": substituted_from,
        "mismatches": mismatches,
    }


def download_regression(entry: dict[str, Any]) -> dict[str, Any]:
    key = entry["key"]
    print(f"  {key} ({entry.get('source')})", flush=True)

    if entry.get("source") == "sklearn":
        from sklearn import datasets as skd

        bunch = getattr(skd, entry["loader"])(as_frame=True)
        X, y_raw = bunch.data.copy(), np.asarray(bunch.target, dtype=np.float64)
        substituted_from = None
        source_url = f"sklearn.datasets.{entry['loader']}"
    else:
        primary = entry.get("openml_id")
        bunch, err = _retry(lambda: _fetch_openml(primary, None), f"{key} id={primary}")
        substituted_from = None
        if bunch is None and entry.get("fallback_name"):
            print(f"    id {primary} unusable; trying name={entry['fallback_name']!r}", flush=True)
            bunch, err = _retry(
                lambda: _fetch_openml(None, entry["fallback_name"]),
                f"{key} name={entry['fallback_name']}",
            )
            if bunch is not None:
                substituted_from = primary
        if bunch is None:
            return {"key": key, "status": "FAILED", "error": f"{type(err).__name__}: {err}"}
        X = bunch.data.copy()
        target = bunch.target
        if target is None:
            # Some OpenML regression sets declare no default target; the
            # convention there is that the last column is the response.
            target = X.iloc[:, -1]
            X = X.iloc[:, :-1]
            print(f"    no default target; using last column {target.name!r}", flush=True)
        if isinstance(target, pd.DataFrame):  # some sets return a 1-col frame
            target = target.iloc[:, 0]
        y_raw = pd.to_numeric(pd.Series(np.asarray(target).ravel()), errors="coerce").to_numpy(
            dtype=np.float64
        )
        source_url = f"https://www.openml.org/d/{primary}"

    all_nan = [c for c in X.columns if X[c].isna().all()]
    if all_nan:
        X = X.drop(columns=all_nan)
    categorical = [c for c in X.columns if str(X[c].dtype) in ("category", "object", "bool")]
    for c in categorical:
        X[c] = X[c].astype("string").fillna("__missing__").astype(str)
    for c in X.columns:
        if c not in categorical:
            X[c] = pd.to_numeric(X[c], errors="coerce").astype(np.float64)

    keep = np.isfinite(y_raw)
    X, y = X.loc[keep].reset_index(drop=True), y_raw[keep]

    frame = X.copy()
    target_col = "__target__"
    frame[target_col] = y
    CACHE.mkdir(parents=True, exist_ok=True)
    parquet_path = CACHE / f"{key}.parquet"
    frame.to_parquet(parquet_path, index=False)

    meta = {
        "key": key,
        "task": "regression",
        "openml_id": entry.get("openml_id"),
        "substituted_from_openml_id": substituted_from,
        "source_url": source_url,
        "n": int(X.shape[0]),
        "d": int(X.shape[1]),
        "n_classes": 0,
        "class_names": [],
        "y_mean": float(np.mean(y)),
        "y_std": float(np.std(y)),
        "categorical_columns": categorical,
        "numeric_columns": [c for c in X.columns if c not in categorical],
        "dropped_all_nan_columns": all_nan,
        "target_column": target_col,
        "n_missing_cells": int(X.isna().sum().sum()),
        "sha256": _sha256_frame(frame),
        "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "expected": entry.get("expected", {}),
    }
    (CACHE / f"{key}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {
        "key": key,
        "status": "OK",
        "n": meta["n"],
        "d": meta["d"],
        "classes": 0,
        "imbalance": float("nan"),
        "missing": meta["n_missing_cells"],
        "bytes": parquet_path.stat().st_size,
        "substituted_from": substituted_from,
        "mismatches": [],
    }


def warm_tabpfn_cache() -> dict[str, Any]:
    """Instantiate + fit + predict once so HuggingFace weights land on disk."""
    import torch

    report: dict[str, Any] = {
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"
    report["device"] = device
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 5))

    try:
        import tabpfn
        from tabpfn import TabPFNClassifier, TabPFNRegressor

        report["tabpfn_version"] = tabpfn.__version__
        t0 = time.perf_counter()
        clf = TabPFNClassifier(device=device, random_state=0)
        clf.fit(X, rng.integers(0, 3, 20))
        clf.predict_proba(X)
        report["classifier_warmup_s"] = round(time.perf_counter() - t0, 2)

        t0 = time.perf_counter()
        reg = TabPFNRegressor(device=device, random_state=0)
        reg.fit(X, rng.normal(size=20))
        reg.predict(X)
        report["regressor_warmup_s"] = round(time.perf_counter() - t0, 2)
        report["status"] = "OK"
    except Exception as exc:
        report["status"] = "FAILED"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()

    # Report where the weights actually landed, as the spec requires.
    candidates = [
        Path(os.environ.get("APPDATA", "")) / "tabpfn",
        Path.home() / ".cache" / "tabpfn",
        Path(os.environ.get("LOCALAPPDATA", "")) / "tabpfn",
    ]
    for cand in candidates:
        if cand.exists():
            ckpts = list(cand.rglob("*.ckpt"))
            if ckpts:
                report["weights_dir"] = str(cand)
                report["weights"] = {
                    c.name: round(c.stat().st_size / 1e6, 1) for c in ckpts
                }
                break
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these keys")
    ap.add_argument("--skip-models", action="store_true")
    ap.add_argument("--skip-regression", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-download cached datasets")
    args = ap.parse_args()

    CACHE.mkdir(parents=True, exist_ok=True)
    print(f"cache directory: {CACHE}")
    print("=" * 78)

    clf_cfg = yaml.safe_load((CONFIGS / "datasets_clf.yaml").read_text(encoding="utf-8"))
    reg_cfg = yaml.safe_load((CONFIGS / "datasets_reg.yaml").read_text(encoding="utf-8"))

    reports: list[dict[str, Any]] = []
    print("\nCLASSIFICATION")
    for entry in clf_cfg["datasets"]:
        if args.only and entry["key"] not in args.only:
            continue
        if not args.force and (CACHE / f"{entry['key']}.meta.json").exists():
            print(f"  {entry['key']} already cached, skipping")
            meta = json.loads((CACHE / f"{entry['key']}.meta.json").read_text(encoding="utf-8"))
            reports.append(
                {
                    "key": entry["key"], "status": "CACHED", "n": meta["n"], "d": meta["d"],
                    "classes": meta["n_classes"], "imbalance": meta.get("imbalance_ratio", 1.0),
                    "missing": meta["n_missing_cells"],
                    "bytes": (CACHE / f"{entry['key']}.parquet").stat().st_size,
                    "substituted_from": meta.get("substituted_from_openml_id"),
                    "mismatches": meta.get("shape_mismatches", []),
                }
            )
            continue
        reports.append(download_classification(entry))

    if not args.skip_regression:
        print("\nREGRESSION")
        for entry in reg_cfg["datasets"]:
            if args.only and entry["key"] not in args.only:
                continue
            if not args.force and (CACHE / f"{entry['key']}.meta.json").exists():
                print(f"  {entry['key']} already cached, skipping")
                continue
            reports.append(download_regression(entry))

    model_report = None
    if not args.skip_models:
        print("\nMODEL WEIGHTS")
        model_report = warm_tabpfn_cache()
        for k, v in model_report.items():
            if k != "traceback":
                print(f"  {k}: {v}")

    # ---------------------------------------------------------------- summary
    print("\n" + "=" * 78)
    ok = [r for r in reports if r["status"] in ("OK", "CACHED")]
    bad = [r for r in reports if r["status"] == "FAILED"]
    print(f"{'key':<26}{'status':<8}{'n':>6}{'d':>5}{'C':>4}{'imb':>7}{'miss':>7}{'KB':>8}")
    print("-" * 78)
    for r in reports:
        if r["status"] == "FAILED":
            print(f"{r['key']:<26}{'FAILED':<8}  {r['error'][:40]}")
        else:
            imb = r["imbalance"]
            print(
                f"{r['key']:<26}{r['status']:<8}{r['n']:>6}{r['d']:>5}{r['classes']:>4}"
                f"{imb:>7.2f}{r['missing']:>7}{r['bytes'] / 1024:>8.0f}"
            )
    total = sum(p.stat().st_size for p in CACHE.glob("*") if p.is_file())
    print("-" * 78)
    print(f"succeeded: {len(ok)}   failed: {len(bad)}   cache size on disk: {total / 1e6:.2f} MB")

    subs = [r for r in ok if r.get("substituted_from")]
    if subs:
        print("\nSUBSTITUTIONS (recorded in configs/datasets_clf.yaml):")
        for r in subs:
            print(f"  {r['key']}: id {r['substituted_from']} failed -> resolved by name")
    mism = [r for r in ok if r.get("mismatches")]
    if mism:
        print("\nSHAPE MISMATCHES vs the spec table (reported, not silently accepted):")
        for r in mism:
            print(f"  {r['key']}: {'; '.join(r['mismatches'])}")

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "download_report.json").write_text(
        json.dumps({"datasets": reports, "models": model_report}, indent=2, default=str),
        encoding="utf-8",
    )
    if bad:
        print(f"\n!! {len(bad)} dataset(s) FAILED -- rerun this script when connected.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
