"""Execution of a single experiment cell: fit models, build sets, measure everything.

One *cell* is ``(dataset, model, cp_method, alpha, seed, n_train)``.  A cell fits
models once and then emits one result row per ``(score, alpha)`` combination,
because lac/aps/raps and every alpha are post-hoc functions of the same predicted
probabilities.  Refitting per score would multiply the compute by 9 without
changing a single number.

Timing convention (matters for RQ1):
  ``fit_time``     -- total seconds inside ``.fit()``, summed over folds.
  ``predict_time`` -- total seconds inside ``predict_proba``, summed over folds.
  ``cp_time``      -- seconds inside the conformal set construction.
All three exclude the one-off model warm-up, which is forced before the clock
starts (see ``models.warmup``).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from .config import CPMethod, parse_cp_method
from .conformal import (
    cvplus_sets,
    mondrian_cvplus_sets,
    mondrian_split_conformal_sets,
    pooled_cv_sets,
    split_conformal_sets,
)
from .data import Dataset, build_preprocessor, make_imbalanced
from .metrics import classification_metrics, pointwise_metrics
from .models import GlobalClassAligned, make_model, reset_gpu
from .scores import compute_scores, draw_u
from .utils import Timer

__all__ = [
    "build_estimator",
    "make_splits",
    "run_classification_cell",
    "stratified_subsample",
]


def build_estimator(
    model_name: str,
    seed: int,
    dataset: Dataset,
    n_classes: int,
    device: str | None = None,
) -> GlobalClassAligned:
    """Preprocessing + model as one pipeline, wrapped for global class alignment.

    The preprocessor lives *inside* the pipeline so it is refitted on each fold's
    training part only.  That makes zero leakage structural rather than a thing
    we have to remember to do.
    """
    base = make_model(model_name, seed, task="classification", n_classes=None, device=device)
    pipe = Pipeline([("prep", build_preprocessor(dataset)), ("est", base)])
    return GlobalClassAligned(pipe, n_classes=n_classes)


def stratified_subsample(
    y: np.ndarray, n_target: int, seed: int
) -> np.ndarray:
    """Indices of a stratified subsample of size ~``n_target``.

    Every class keeps at least one point, so a fold can never lose a class
    entirely just because of subsampling.
    """
    y = np.asarray(y)
    n = y.size
    if int(n_target) >= n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    quota = np.maximum(1, np.floor(counts * n_target / n).astype(int))
    # Hand out the rounding remainder to the largest classes first.
    while quota.sum() > n_target and quota.max() > 1:
        quota[quota.argmax()] -= 1
    order = np.argsort(-counts)
    i = 0
    while quota.sum() < n_target:
        c = order[i % len(order)]
        if quota[c] < counts[c]:
            quota[c] += 1
        i += 1
    picks = [
        rng.choice(np.flatnonzero(y == c), size=q, replace=False)
        for c, q in zip(classes, quota)
    ]
    return np.sort(np.concatenate(picks))


def make_splits(
    dataset: Dataset,
    seed: int,
    test_frac: float,
    test_cap: int,
    n_train: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified train_pool / test split, then a stratified subsample of the pool.

    Test size is ``min(test_frac * n, test_cap)`` as specified.
    """
    y = dataset.y
    n = y.size
    n_test = min(round(test_frac * n), int(test_cap))
    idx_pool, idx_test = train_test_split(
        np.arange(n), test_size=n_test, random_state=seed, stratify=y
    )
    if n_train is not None and n_train < idx_pool.size:
        sub = stratified_subsample(y[idx_pool], int(n_train), seed=seed)
        idx_pool = idx_pool[sub]
    return np.sort(idx_pool), np.sort(idx_test)


def _fit_predict(
    estimator: GlobalClassAligned,
    X_fit: pd.DataFrame,
    y_fit: np.ndarray,
    X_evals: list[pd.DataFrame],
    timings: dict[str, float],
) -> list[np.ndarray]:
    """Fit once, predict on several matrices, accumulating fit/predict seconds."""
    with Timer() as t:
        estimator.fit(X_fit, y_fit)
    timings["fit_time"] = timings.get("fit_time", 0.0) + t.elapsed
    outs = []
    with Timer() as t:
        for X in X_evals:
            outs.append(estimator.predict_proba(X))
    timings["predict_time"] = timings.get("predict_time", 0.0) + t.elapsed
    timings["n_fits"] = timings.get("n_fits", 0) + 1
    return outs


def _device_of(estimator: GlobalClassAligned) -> str:
    inner = getattr(estimator, "estimator_", None)
    if inner is None:
        return "unknown"
    step = inner.named_steps.get("est") if hasattr(inner, "named_steps") else None
    return getattr(step, "device_used", "cpu")


def _split_style(
    method: CPMethod,
    dataset: Dataset,
    X_pool: pd.DataFrame,
    y_pool: np.ndarray,
    X_test: pd.DataFrame,
    model_name: str,
    seed: int,
    device: str | None,
    timings: dict[str, float],
) -> dict[str, Any]:
    """Split / Mondrian-split: one model on the proper-training part."""
    idx_tr, idx_cal = train_test_split(
        np.arange(y_pool.size),
        test_size=method.cal_frac,
        random_state=seed,
        stratify=y_pool,
    )
    est = build_estimator(model_name, seed, dataset, dataset.n_classes, device)
    p_cal, p_test = _fit_predict(
        est, X_pool.iloc[idx_tr], y_pool[idx_tr], [X_pool.iloc[idx_cal], X_test], timings
    )
    return {
        "kind": "split",
        "p_cal": p_cal,
        "y_cal": y_pool[idx_cal],
        "p_test": p_test,
        "p_point": p_test,
        "n_proper_train": int(idx_tr.size),
        "n_cal": int(idx_cal.size),
        "device_used": _device_of(est),
    }


def _cv_style(
    method: CPMethod,
    dataset: Dataset,
    X_pool: pd.DataFrame,
    y_pool: np.ndarray,
    X_test: pd.DataFrame,
    model_name: str,
    seed: int,
    device: str | None,
    timings: dict[str, float],
) -> dict[str, Any]:
    """CV+ family: K fold models, out-of-fold scores + per-fold test predictions."""
    n = y_pool.size
    n_c = dataset.n_classes
    if method.family == "jackknife_plus":
        n_folds = n
        fold_of = np.arange(n)
        folds = [(np.setdiff1d(np.arange(n), [i]), np.array([i])) for i in range(n)]
    else:
        n_folds = int(method.n_folds)
        # A stratified fold can only exist if every class has >= n_folds members.
        counts = np.bincount(y_pool, minlength=n_c)
        min_count = int(counts[counts > 0].min())
        n_folds = max(2, min(n_folds, min_count))
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        folds = list(skf.split(np.zeros(n), y_pool))
        fold_of = np.empty(n, dtype=np.int64)
        for k, (_, va) in enumerate(folds):
            fold_of[va] = k

    p_oof = np.zeros((n, n_c), dtype=np.float64)
    per_fold_test = np.zeros((n_folds, X_test.shape[0], n_c), dtype=np.float64)
    device_used = "unknown"
    for k, (tr, va) in enumerate(folds):
        est = build_estimator(model_name, seed, dataset, n_c, device)
        p_va, p_te = _fit_predict(
            est, X_pool.iloc[tr], y_pool[tr], [X_pool.iloc[va], X_test], timings
        )
        p_oof[va] = p_va
        per_fold_test[k] = p_te
        device_used = _device_of(est)
        reset_gpu()

    out: dict[str, Any] = {
        "kind": "cv",
        "p_oof": p_oof,
        "per_fold_test": per_fold_test,
        "fold_of": fold_of,
        "n_folds_used": int(n_folds),
        # Point-predictor quality for a CV method is the fold ensemble's.
        "p_point": per_fold_test.mean(axis=0),
        "device_used": device_used,
    }

    if method.family == "pooled_cv":
        # Pooled-CV additionally needs ONE model fitted on all n points.
        est_full = build_estimator(model_name, seed, dataset, n_c, device)
        (p_test_full,) = _fit_predict(est_full, X_pool, y_pool, [X_test], timings)
        out["p_test_full"] = p_test_full
        out["p_point"] = p_test_full
    return out


def _build_sets(
    method: CPMethod,
    s: dict[str, Any],
    fitted: dict[str, Any],
    y_pool: np.ndarray,
    n_classes: int,
    alpha: float,
    quantile_method: str,
    chunk_size: int,
) -> tuple[np.ndarray, float, list[int]]:
    """Prediction sets, a representative threshold, and any starved classes.

    ``qhat`` is NaN for CV+ family methods: they aggregate per training point and
    have no single scalar threshold, so reporting one would be a fiction.
    """
    if method.family == "split":
        sets, qhat = split_conformal_sets(
            s["cal"], fitted["y_cal"], s["test"], alpha, quantile_method
        )
        return sets, float(qhat), []
    if method.family == "mondrian_split":
        sets, qhat_vec, starved = mondrian_split_conformal_sets(
            s["cal"], fitted["y_cal"], s["test"], alpha, n_classes, quantile_method
        )
        finite = qhat_vec[np.isfinite(qhat_vec)]
        return sets, float(finite.mean()) if finite.size else float("inf"), starved
    if method.family in ("cvplus", "jackknife_plus"):
        sets = cvplus_sets(s["oof_true"], s["per_fold"], fitted["fold_of"], alpha, chunk_size)
        return sets, float("nan"), []
    if method.family == "mondrian_cvplus":
        sets, starved = mondrian_cvplus_sets(
            s["oof_true"], y_pool, s["per_fold"], fitted["fold_of"], alpha, n_classes, chunk_size
        )
        return sets, float("nan"), starved
    if method.family == "pooled_cv":
        sets, qhat = pooled_cv_sets(s["oof_true"], s["full"], alpha, quantile_method)
        return sets, float(qhat), []
    raise ValueError(f"unhandled cp family {method.family!r}")


def run_classification_cell(
    dataset: Dataset,
    model_name: str,
    cp_method: str,
    seed: int,
    alphas: list[float],
    scores: list[str],
    n_train: int | None = None,
    test_frac: float = 0.3,
    test_cap: int = 1000,
    chunk_size: int = 256,
    device: str | None = None,
    quantile_method: str = "tight",
    raps_lambda: float = 0.01,
    raps_k_reg: int = 1,
    imbalance: str = "natural",
    study: str = "main",
    minority_frac: float | None = None,
    extra_fields: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Run one cell end-to-end and return one row per (score, alpha).

    ``minority_frac`` (RQ5) subsamples the dataset so the rarest class holds that
    share BEFORE the train/test split, so both splits see the same skew.  The
    realised counts -- not the requested ratio -- are what gets recorded.
    """
    realised_counts: dict[int, int] | None = None
    if minority_frac is not None:
        counts = dataset.class_counts()
        minority = min(counts, key=counts.get)
        n_c_all = len(counts)
        rest = (1.0 - minority_frac) / max(1, n_c_all - 1)
        ratios = {c: (minority_frac if c == minority else rest) for c in counts}
        X_imb, y_imb, realised_counts = make_imbalanced(dataset.X, dataset.y, ratios, seed)
        dataset = replace(
            dataset, X=X_imb.reset_index(drop=True), y=y_imb, key=dataset.key
        )

    method = parse_cp_method(cp_method)
    idx_pool, idx_test = make_splits(dataset, seed, test_frac, test_cap, n_train)
    X_pool, y_pool = dataset.X.iloc[idx_pool], dataset.y[idx_pool]
    X_test, y_test = dataset.X.iloc[idx_test], dataset.y[idx_test]
    n_c = dataset.n_classes

    timings: dict[str, float] = {}
    with Timer() as total_timer:
        if method.family in ("split", "mondrian_split"):
            fitted = _split_style(
                method, dataset, X_pool, y_pool, X_test, model_name, seed, device, timings
            )
        else:
            fitted = _cv_style(
                method, dataset, X_pool, y_pool, X_test, model_name, seed, device, timings
            )
    model_total_s = total_timer.elapsed

    # Randomisation variables: one U per example, reused across every fold model
    # so that a test point is scored with the same U no matter which fold sees it.
    rng = np.random.default_rng(abs(hash((dataset.key, model_name, cp_method, seed))) % (2**32))
    u_test = draw_u(y_test.size, rng)
    u_pool = draw_u(y_pool.size, rng)

    point = pointwise_metrics(fitted["p_point"], y_test)
    train_counts = {int(k): int(v) for k, v in zip(*np.unique(y_pool, return_counts=True))}

    rows: list[dict[str, Any]] = []
    for score_name in scores:
        kw = {"lam": raps_lambda, "k_reg": raps_k_reg} if score_name == "raps" else {}
        randomized = score_name in ("aps", "raps")
        s: dict[str, Any] = {}  # this score's matrices, passed explicitly to _build_sets

        if fitted["kind"] == "split":
            s["cal"] = compute_scores(
                fitted["p_cal"], score_name,
                u=draw_u(fitted["y_cal"].size, np.random.default_rng(seed + 7)),
                randomized=randomized, **kw,
            )
            s["test"] = compute_scores(
                fitted["p_test"], score_name, u=u_test, randomized=randomized, **kw
            )
        else:
            s_oof = compute_scores(
                fitted["p_oof"], score_name, u=u_pool, randomized=randomized, **kw
            )
            s["oof_true"] = s_oof[np.arange(y_pool.size), y_pool]
            s["per_fold"] = np.stack(
                [
                    compute_scores(
                        fitted["per_fold_test"][k], score_name, u=u_test,
                        randomized=randomized, **kw,
                    )
                    for k in range(fitted["n_folds_used"])
                ]
            )

        if method.family == "pooled_cv":
            s["full"] = compute_scores(
                fitted["p_test_full"], score_name, u=u_test, randomized=randomized, **kw
            )

        with Timer(sync_cuda=False) as cp_timer:
            _build_sets(method, s, fitted, y_pool, n_c, alphas[0], quantile_method, chunk_size)
        cp_time_first = cp_timer.elapsed

        for alpha in alphas:
            sets, qhat, starved = _build_sets(
                method, s, fitted, y_pool, n_c, alpha, quantile_method, chunk_size
            )

            m = classification_metrics(sets, y_test, alpha, n_c)
            row: dict[str, Any] = {
                "dataset": dataset.key,
                "model": model_name,
                "cp_method": cp_method,
                "cp_family": method.family,
                "score": score_name,
                "alpha": float(alpha),
                "seed": int(seed),
                "n_train": int(y_pool.size),
                "n_test": int(y_test.size),
                "n_classes": n_c,
                "n_features": dataset.d,
                "imbalance": imbalance,
                "study": study,
                "n_folds": int(fitted.get("n_folds_used", 0)),
                "n_cal": int(fitted.get("n_cal", 0)),
                "n_proper_train": int(fitted.get("n_proper_train", 0)),
                "guarantee": 1.0 - 2 * alpha if method.needs_folds else 1.0 - alpha,
                "fit_time": float(timings.get("fit_time", 0.0)),
                "predict_time": float(timings.get("predict_time", 0.0)),
                "model_total_time": float(model_total_s),
                "cp_time": float(cp_time_first),
                "qhat": float(qhat),
                "n_fits": int(timings.get("n_fits", 0)),
                "device_used": fitted.get("device_used", "unknown"),
                "quantile_method": quantile_method,
                "starved_classes": len(starved),
                "train_class_counts": str(train_counts),
                "minority_frac": (
                    float(minority_frac) if minority_frac is not None else float("nan")
                ),
                "realised_class_counts": str(realised_counts) if realised_counts else "",
                **{k: v for k, v in m.items() if not isinstance(v, dict)},
                "per_class_coverage": str(m["per_class_coverage"]),
                "sscv_bins": str(m["sscv_bins"]),
                "accuracy": point["accuracy"],
                "log_loss": point["log_loss"],
                "ece": point["ece"],
            }
            if extra_fields:
                row.update(extra_fields)
            rows.append(row)

    reset_gpu()
    return rows
