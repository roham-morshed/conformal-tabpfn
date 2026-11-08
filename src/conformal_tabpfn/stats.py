"""Statistical machinery for head-to-head claims (spec section 9).

Everything here operates on *per-dataset aggregates*, because the datasets are
the independent units; the 20 seeds within a dataset are repeated measurements
of the same problem and must not be treated as independent samples.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "critical_difference",
    "holm_bonferroni",
    "mean_ranks",
    "wilcoxon_pairwise",
]


def wilcoxon_pairwise(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Wilcoxon signed-rank test of ``a`` vs ``b`` across datasets.

    Returns the p-value plus wins/ties/losses, which the spec requires alongside
    it (a p-value without the direction and the count is not a claim).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    diff = a - b
    wins = int(np.sum(diff < 0))  # lower is better (set size, gap, time)
    losses = int(np.sum(diff > 0))
    ties = int(np.sum(diff == 0))
    if a.size < 3 or np.all(diff == 0):
        p = float("nan")
    else:
        try:
            p = float(stats.wilcoxon(a, b, zero_method="wilcox").pvalue)
        except ValueError:
            p = float("nan")
    return {
        "n": int(a.size),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "p_value": p,
        "median_diff": float(np.median(diff)) if a.size else float("nan"),
    }


def holm_bonferroni(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Holm-Bonferroni step-down correction over a family of comparisons."""
    items = [(k, v) for k, v in pvalues.items() if np.isfinite(v)]
    items.sort(key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, dict] = {}
    prev = 0.0
    for i, (key, p) in enumerate(items):
        adj = min(1.0, max(prev, (m - i) * p))  # enforce monotonicity
        prev = adj
        out[key] = {"p_raw": p, "p_holm": adj, "significant": bool(adj < alpha)}
    for key, p in pvalues.items():
        if key not in out:
            out[key] = {"p_raw": p, "p_holm": float("nan"), "significant": False}
    return out


def mean_ranks(pivot: pd.DataFrame, lower_is_better: bool = True) -> pd.Series:
    """Mean rank of each column (method) across rows (datasets)."""
    values = pivot if lower_is_better else -pivot
    return values.rank(axis=1, method="average").mean(axis=0).sort_values()


def critical_difference(n_methods: int, n_datasets: int, alpha: float = 0.05) -> float:
    """Nemenyi critical difference for mean ranks.

    ``CD = q_alpha * sqrt(k(k+1) / (6N))``.  The q values are the standard
    Studentised-range-derived constants (Demsar 2006, Table 5) for alpha=0.05;
    we interpolate nothing and fall back to the largest tabulated value rather
    than inventing one.
    """
    q05 = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031,
        9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354, 15: 3.391,
    }
    q10 = {
        2: 1.645, 3: 2.052, 4: 2.291, 5: 2.459, 6: 2.589, 7: 2.693, 8: 2.780,
        9: 2.855, 10: 2.920, 11: 2.978, 12: 3.030, 13: 3.077, 14: 3.120, 15: 3.159,
    }
    table = q05 if alpha == 0.05 else q10
    k = int(n_methods)
    q = table.get(k, table[max(table)])
    return float(q * np.sqrt(k * (k + 1) / (6.0 * max(1, n_datasets))))
