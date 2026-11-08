"""Evaluation metrics for conformal prediction sets, intervals and point models.

Every metric is a pure function of arrays so it can be unit-tested without a
model.  Nothing here silently drops points: if a quantity is undefined (e.g.
per-class coverage for a class with no test points) it is omitted from the dict
and the omission is visible in the keys, never replaced by a fabricated number.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "SSCV_BINS",
    "classification_metrics",
    "expected_calibration_error",
    "pointwise_metrics",
    "regression_metrics",
    "sscv",
]

# Size-stratified coverage bins.  The spec asks for [1], [2-3], [4-6], [7+];
# empty sets (size 0) can never cover, so they get their own bin instead of
# being folded into [1] (which would mask an empty-set failure) or dropped
# (which would silently hide test points).
SSCV_BINS: tuple[tuple[str, int, float], ...] = (
    ("0", 0, 0),
    ("1", 1, 1),
    ("2-3", 2, 3),
    ("4-6", 4, 6),
    ("7+", 7, np.inf),
)


def _as_sets(sets: np.ndarray) -> np.ndarray:
    arr = np.asarray(sets)
    if arr.dtype != bool:
        arr = arr.astype(bool)
    if arr.ndim != 2:
        raise ValueError(f"sets must be (n_test, C); got shape {arr.shape}")
    return arr


def sscv(sets: np.ndarray, y_true: np.ndarray, alpha: float) -> tuple[float, dict[str, float]]:
    """Size-stratified coverage violation.

    Bin test points by the size of their predicted set, then report
    ``max_b |cov_b - (1 - alpha)|`` over non-empty bins.  A method can hit the
    marginal rate exactly while badly under-covering the hard (large-set) points;
    SSCV exposes that (Angelopoulos et al. 2021).
    """
    s = _as_sets(sets)
    y = np.asarray(y_true, dtype=np.int64)
    sizes = s.sum(axis=1)
    covered = s[np.arange(s.shape[0]), y]
    target = 1.0 - alpha
    per_bin: dict[str, float] = {}
    worst = 0.0
    for name, lo, hi in SSCV_BINS:
        m = (sizes >= lo) & (sizes <= hi)
        if not np.any(m):
            continue
        cov = float(covered[m].mean())
        per_bin[name] = cov
        worst = max(worst, abs(cov - target))
    return float(worst), per_bin


def classification_metrics(
    sets: np.ndarray,
    y_true: np.ndarray,
    alpha: float,
    n_classes: int | None = None,
) -> dict[str, float | dict]:
    """Coverage / efficiency / fairness metrics for classification sets."""
    s = _as_sets(sets)
    y = np.asarray(y_true, dtype=np.int64)
    if y.shape[0] != s.shape[0]:
        raise ValueError("y_true length does not match sets")
    n_c = int(n_classes) if n_classes is not None else s.shape[1]
    sizes = s.sum(axis=1)
    covered = s[np.arange(s.shape[0]), y]

    per_class: dict[int, float] = {}
    per_class_n: dict[int, int] = {}
    for c in range(n_c):
        m = y == c
        n_in_class = int(m.sum())
        if n_in_class == 0:
            continue  # undefined, deliberately absent rather than NaN
        per_class[c] = float(covered[m].mean())
        per_class_n[c] = n_in_class

    if per_class:
        vals = np.array(list(per_class.values()), dtype=np.float64)
        cc_min = float(vals.min())
        cc_max = float(vals.max())
        cc_gap = cc_max - cc_min
    else:
        cc_min = cc_max = cc_gap = float("nan")

    worst, per_bin = sscv(s, y, alpha)
    return {
        "coverage": float(covered.mean()),
        "avg_set_size": float(sizes.mean()),
        "median_set_size": float(np.median(sizes)),
        "frac_singleton": float(np.mean(sizes == 1)),
        "frac_empty": float(np.mean(sizes == 0)),
        "max_set_size": int(sizes.max()) if sizes.size else 0,
        "cc_coverage_min": cc_min,
        "cc_coverage_max": cc_max,
        "cc_coverage_gap": cc_gap,
        "sscv": worst,
        "per_class_coverage": {int(k): v for k, v in per_class.items()},
        "per_class_n_test": {int(k): v for k, v in per_class_n.items()},
        "sscv_bins": per_bin,
        "n_test": int(s.shape[0]),
    }


def regression_metrics(
    lo: np.ndarray, hi: np.ndarray, y_true: np.ndarray, alpha: float
) -> dict[str, float]:
    """Coverage / width / Winkler interval score for regression intervals."""
    lo_a = np.asarray(lo, dtype=np.float64)
    hi_a = np.asarray(hi, dtype=np.float64)
    y = np.asarray(y_true, dtype=np.float64)
    covered = (y >= lo_a) & (y <= hi_a)
    width = hi_a - lo_a
    # Winkler / interval score: width + (2/alpha) * miss distance on each side.
    below = np.maximum(0.0, lo_a - y)
    above = np.maximum(0.0, y - hi_a)
    interval_score = width + (2.0 / alpha) * (below + above)
    finite = np.isfinite(width)
    return {
        "coverage": float(covered.mean()),
        "avg_width": float(np.mean(width[finite])) if np.any(finite) else float("inf"),
        "median_width": float(np.median(width[finite])) if np.any(finite) else float("inf"),
        "interval_score": (
            float(np.mean(interval_score[finite])) if np.any(finite) else float("inf")
        ),
        "frac_infinite_width": float(np.mean(~finite)),
        "n_test": int(y.size),
    }


def expected_calibration_error(
    probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15
) -> float:
    """Standard top-label ECE with equal-width confidence bins (Guo et al. 2017)."""
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(y_true, dtype=np.int64)
    conf = p.max(axis=1)
    pred = p.argmax(axis=1)
    correct = (pred == y).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = conf.size
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if not np.any(m):
            continue
        ece += (m.sum() / n) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def pointwise_metrics(
    probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15
) -> dict[str, float]:
    """Accuracy / log-loss / ECE of the underlying point predictor.

    Recorded for every cell so the paper can separate "the model got better"
    from "the conformal wrapper got better".
    """
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(y_true, dtype=np.int64)
    eps = 1e-15
    p_clip = np.clip(p, eps, 1.0)
    p_clip = p_clip / p_clip.sum(axis=1, keepdims=True)
    ll = -float(np.mean(np.log(p_clip[np.arange(p.shape[0]), y])))
    return {
        "accuracy": float(np.mean(p.argmax(axis=1) == y)),
        "log_loss": ll,
        "ece": expected_calibration_error(p, y, n_bins=n_bins),
    }
