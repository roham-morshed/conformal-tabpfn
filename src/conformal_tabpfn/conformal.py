"""Conformal prediction algorithms, implemented from first principles.

Deliberately no third-party conformal library is used for the core methods
(see the project spec): every guarantee-critical line has to be auditable.

References
----------
Vovk, Gammerman & Shafer (2005), *Algorithmic Learning in a Random World*.
Lei et al. (2018), "Distribution-Free Predictive Inference for Regression", JRSS-B.
Barber, Candes, Ramdas & Tibshirani (2021), "Predictive inference with the
    jackknife+", *Annals of Statistics* 49(1):486-507.  CV+ / jackknife+ and the
    ``1 - 2*alpha`` guarantee come from this paper.
Vovk (2012); Lei et al. (2018) for Mondrian (class-conditional) taxonomies.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "QUANTILE_METHOD_DEFAULT",
    "auto_chunk_size",
    "conformal_quantile",
    "cvplus_regression_interval",
    "cvplus_sets",
    "mondrian_cvplus_sets",
    "mondrian_split_conformal_sets",
    "pooled_cv_sets",
    "split_conformal_interval",
    "split_conformal_sets",
]

# Peak bytes allowed for one CV+ comparison block before we shrink the chunk.
_DEFAULT_MEM_BUDGET_BYTES = 256 * 1024 * 1024

# Which discrete quantile rule Split-CP uses.  "tight" is the textbook k-th
# order statistic; "higher" is the project spec's literal numpy call, which is
# valid but conservative.  See conformal_quantile() for the measured difference
# and NOTES.md for why the default was changed.
QUANTILE_METHOD_DEFAULT = "tight"


def conformal_quantile(
    scores: np.ndarray, alpha: float, method: str = QUANTILE_METHOD_DEFAULT
) -> float:
    """The finite-sample conformal quantile of a set of calibration scores.

    Evaluated at level ``ceil((n + 1) * (1 - alpha)) / n``.  Writing
    ``k = ceil((n + 1) * (1 - alpha))``, the *tight* conformal quantile is the
    ``k``-th smallest calibration score (equivalently: append ``+inf`` to the
    scores and take the ``k``-th smallest of the ``n + 1`` values, which is the
    Vovk et al. 2005 formulation).

    Linear interpolation is **never** acceptable: it returns a value strictly
    between two order statistics, below the required one, and empirically
    under-covers (0.8993 at n=99, alpha=0.1 over 200k replicates).
    ``tests/test_quantile.py`` pins that counterexample.

    Two admissible discrete methods are offered:

    ``"tight"`` (default)
        The exact ``k``-th smallest score, taken with ``np.partition`` so that
        no floating-point level is involved.  Measured coverage
        0.9007 / 0.9006 / 0.8984 at n = 19 / 39 / 99 for alpha = 0.1 -- i.e. the
        textbook rate.  Note that ``np.quantile(s, k / n,
        method="inverted_cdf")`` is *not* a safe substitute: ``k / n * n`` is not
        always exactly ``k`` in binary floating point, and it picks the wrong
        order statistic in 71 of the ~9000 (n, alpha) combinations we swept.
    ``"higher"``
        numpy's ``method="higher"`` at the level above, i.e. the code literally
        given in the project spec (section 4.2).  Because numpy's virtual index
        is ``q * (n - 1)``, this lands on the ``(k + 1)``-th smallest score
        whenever ``k < n``, so it is **valid but conservative**: measured
        coverage 0.9508 / 0.9254 / 0.9084 at the same settings, against the
        tight 0.9000.  Kept for spec fidelity and pinned by a unit test, but not
        used for the paper's numbers because it would handicap the Split-CP
        baseline exactly in the small-n regime the paper studies.

    The default is set by :data:`QUANTILE_METHOD_DEFAULT` so the whole study can
    be switched with one constant and re-run; the choice is recorded in every
    result row.

    If ``(n + 1) * (1 - alpha) > n`` there are too few calibration points to
    certify the level at all, and the only valid answer is ``+inf`` (the
    prediction set becomes everything).
    """
    s = np.asarray(scores, dtype=np.float64).ravel()
    n = s.size
    if n == 0:
        return float(np.inf)
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return float(np.inf)
    if method == "tight":
        return float(np.partition(s, k - 1)[k - 1])
    if method == "higher":
        return float(np.quantile(s, k / n, method="higher"))
    raise ValueError(
        f"quantile method {method!r} is not guarantee-preserving; use 'tight' or 'higher'"
    )


# --------------------------------------------------------------------------- #
# Split conformal (classification)
# --------------------------------------------------------------------------- #


def split_conformal_sets(
    scores_cal: np.ndarray,
    y_cal: np.ndarray,
    scores_test: np.ndarray,
    alpha: float,
    quantile_method: str = QUANTILE_METHOD_DEFAULT,
) -> tuple[np.ndarray, float]:
    """Split conformal prediction sets.

    Parameters
    ----------
    scores_cal : (n_cal, C) score matrix on the calibration split.
    y_cal : (n_cal,) true labels, encoded 0..C-1.
    scores_test : (n_test, C) score matrix on the test split.

    Returns ``(sets, qhat)`` where ``sets`` is an ``(n_test, C)`` boolean matrix.
    """
    s_cal = np.asarray(scores_cal, dtype=np.float64)
    s_test = np.asarray(scores_test, dtype=np.float64)
    y = np.asarray(y_cal, dtype=np.int64)
    if s_cal.ndim != 2 or s_test.ndim != 2:
        raise ValueError("score matrices must be 2-D")
    if s_cal.shape[1] != s_test.shape[1]:
        raise ValueError(
            "calibration/test score matrices disagree on C: "
            f"{s_cal.shape[1]} vs {s_test.shape[1]}"
        )
    if y.shape[0] != s_cal.shape[0]:
        raise ValueError("y_cal length does not match scores_cal")
    s_true = s_cal[np.arange(s_cal.shape[0]), y]
    qhat = conformal_quantile(s_true, alpha, method=quantile_method)
    sets = s_test <= qhat
    return sets, qhat


def mondrian_split_conformal_sets(
    scores_cal: np.ndarray,
    y_cal: np.ndarray,
    scores_test: np.ndarray,
    alpha: float,
    n_classes: int | None = None,
    quantile_method: str = QUANTILE_METHOD_DEFAULT,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Class-conditional (Mondrian) split conformal.

    One quantile per *true* class; label ``c`` enters the set of a test point
    when its score is at or below the quantile calibrated on class-``c``
    calibration points.  This targets ``P(Y in C(X) | Y = c) >= 1 - alpha`` for
    every class instead of only the marginal guarantee.

    Classes with too few calibration points get ``qhat = +inf`` (label always
    included); those class indices are returned so the caller can log them.
    """
    s_cal = np.asarray(scores_cal, dtype=np.float64)
    s_test = np.asarray(scores_test, dtype=np.float64)
    y = np.asarray(y_cal, dtype=np.int64)
    n_c = int(n_classes) if n_classes is not None else s_cal.shape[1]
    qhat = np.empty(n_c, dtype=np.float64)
    starved: list[int] = []
    for c in range(n_c):
        mask = y == c
        if not np.any(mask):
            qhat[c] = np.inf
            starved.append(c)
            continue
        qhat[c] = conformal_quantile(s_cal[mask, c], alpha, method=quantile_method)
        if not np.isfinite(qhat[c]):
            starved.append(c)
    sets = s_test <= qhat[None, :]
    return sets, qhat, starved


# --------------------------------------------------------------------------- #
# CV+ / jackknife+ (classification)
# --------------------------------------------------------------------------- #


def auto_chunk_size(
    n_train: int,
    n_classes: int,
    requested: int,
    mem_budget_bytes: int = _DEFAULT_MEM_BUDGET_BYTES,
) -> int:
    """Largest test-point chunk whose ``(n_train, chunk, C)`` block fits the budget.

    The CV+ membership rule compares every training point against every
    (test point, candidate label) pair.  Materialising that as one array costs
    ``n_train * n_test * C * 8`` bytes, which for n=1600, n_test=1000, C=10 is
    already ~1 GB, so we stream over test points in chunks instead.
    """
    per_test_point = max(1, int(n_train) * int(n_classes) * 8)
    fits = max(1, int(mem_budget_bytes) // per_test_point)
    return int(max(1, min(int(requested), fits)))


def cvplus_sets(
    oof_true: np.ndarray,
    per_fold_test_S: np.ndarray,
    fold_of: np.ndarray,
    alpha: float,
    chunk_size: int = 256,
    mem_budget_bytes: int = _DEFAULT_MEM_BUDGET_BYTES,
) -> np.ndarray:
    """CV+ prediction sets (Barber et al. 2021); guarantee ``>= 1 - 2*alpha``.

    Parameters
    ----------
    oof_true : (n,) score of the **true** label of training point i, computed by
        the fold model that did *not* see i.
    per_fold_test_S : (K, n_test, C) score of every candidate label at every test
        point, under each fold model.
    fold_of : (n,) fold index in ``0..K-1`` for each training point.

    Membership rule::

        test_S_i = per_fold_test_S[fold_of]            # (n, n_test, C)
        counts   = (oof_true[:, None, None] >= test_S_i).sum(axis=0)
        sets     = (1 + counts) / (n + 1) > alpha

    i.e. a label is kept when enough training points are at least as
    nonconforming as the candidate.  Jackknife+ is exactly this with ``K = n``.
    """
    oof = np.asarray(oof_true, dtype=np.float64).ravel()
    fold = np.asarray(fold_of, dtype=np.int64).ravel()
    per_fold = np.asarray(per_fold_test_S, dtype=np.float64)
    if per_fold.ndim != 3:
        raise ValueError(f"per_fold_test_S must be (K, n_test, C); got {per_fold.shape}")
    n = oof.size
    if fold.size != n:
        raise ValueError("fold_of and oof_true must have the same length")
    n_folds, n_test, n_c = per_fold.shape
    if n and (fold.min() < 0 or fold.max() >= n_folds):
        raise ValueError(f"fold indices out of range [0, {n_folds - 1}]")
    if n == 0:
        return np.ones((n_test, n_c), dtype=bool)

    chunk = auto_chunk_size(n, n_c, chunk_size, mem_budget_bytes)
    sets = np.empty((n_test, n_c), dtype=bool)
    # counts > thresh  <=>  (1 + counts) / (n + 1) > alpha, without float division
    # per element.
    thresh = alpha * (n + 1) - 1.0
    for start in range(0, n_test, chunk):
        stop = min(start + chunk, n_test)
        block = per_fold[:, start:stop, :][fold]  # (n, chunk, C)
        counts = (oof[:, None, None] >= block).sum(axis=0)
        sets[start:stop] = counts > thresh
    return sets


def mondrian_cvplus_sets(
    oof_true: np.ndarray,
    y_train: np.ndarray,
    per_fold_test_S: np.ndarray,
    fold_of: np.ndarray,
    alpha: float,
    n_classes: int | None = None,
    chunk_size: int = 256,
    mem_budget_bytes: int = _DEFAULT_MEM_BUDGET_BYTES,
) -> tuple[np.ndarray, list[int]]:
    """Mondrian-CV+ (this paper's variant): class-conditional CV+.

    For candidate label ``c`` the comparison pool is restricted to training
    points whose *true* label is ``c``, giving a class-conditional analogue of
    the CV+ rule that targets ``P(Y in C(X) | Y = c)`` rather than only the
    marginal rate.

    Returns ``(sets, empty_classes)``; a class with no training points is always
    included and is reported so the caller can log it.
    """
    oof = np.asarray(oof_true, dtype=np.float64).ravel()
    y = np.asarray(y_train, dtype=np.int64).ravel()
    fold = np.asarray(fold_of, dtype=np.int64).ravel()
    per_fold = np.asarray(per_fold_test_S, dtype=np.float64)
    _, n_test, n_c_scores = per_fold.shape
    n_c = int(n_classes) if n_classes is not None else n_c_scores
    if y.size != oof.size or fold.size != oof.size:
        raise ValueError("oof_true, y_train and fold_of must have the same length")

    sets = np.zeros((n_test, n_c), dtype=bool)
    empty: list[int] = []
    chunk = auto_chunk_size(max(1, oof.size), 1, chunk_size, mem_budget_bytes)
    for c in range(n_c):
        m = y == c
        n_class = int(m.sum())
        if n_class == 0:
            sets[:, c] = True
            empty.append(c)
            continue
        oof_c = oof[m]
        fold_c = fold[m]
        thresh = alpha * (n_class + 1) - 1.0
        for start in range(0, n_test, chunk):
            stop = min(start + chunk, n_test)
            block = per_fold[:, start:stop, c][fold_c]  # (n_class, chunk)
            counts = (oof_c[:, None] >= block).sum(axis=0)
            sets[start:stop, c] = counts > thresh
    return sets, empty


def pooled_cv_sets(
    oof_true: np.ndarray,
    scores_test_full: np.ndarray,
    alpha: float,
    quantile_method: str = QUANTILE_METHOD_DEFAULT,
) -> tuple[np.ndarray, float]:
    """Pooled-CV: the shortcut practitioners actually use.

    Pool all n out-of-fold true-label scores to get one quantile, but predict
    with a **single model fitted on all n points**.  This has no finite-sample
    guarantee: the calibration scores and the test scores come from different
    predictors, so exchangeability fails.  It is included precisely to measure
    empirically whether that matters.
    """
    qhat = conformal_quantile(oof_true, alpha, method=quantile_method)
    sets = np.asarray(scores_test_full, dtype=np.float64) <= qhat
    return sets, qhat


# --------------------------------------------------------------------------- #
# Regression
# --------------------------------------------------------------------------- #


def split_conformal_interval(
    residuals_cal: np.ndarray,
    mu_test: np.ndarray,
    alpha: float,
    quantile_method: str = QUANTILE_METHOD_DEFAULT,
) -> tuple[np.ndarray, np.ndarray]:
    """Split-conformal regression interval ``mu(x) +/- qhat``."""
    q = conformal_quantile(residuals_cal, alpha, method=quantile_method)
    mu = np.asarray(mu_test, dtype=np.float64)
    return mu - q, mu + q


def cvplus_regression_interval(
    oof_pred: np.ndarray,
    y_train: np.ndarray,
    per_fold_test_pred: np.ndarray,
    fold_of: np.ndarray,
    alpha: float,
    chunk_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """CV+ / jackknife+ regression interval (Barber et al. 2021).

    ::

        R  = |y - oof_pred|                     # (n,)
        mu = per_fold_test_pred[fold_of]        # (n, n_test)
        k  = ceil((1 - alpha) * (n + 1))
        hi = sort(mu + R[:, None], axis=0)[k - 1]
        lo = sort(mu - R[:, None], axis=0)[n - k]

    ``k > n`` means there are too few training points for the requested level,
    and the only valid interval is ``(-inf, +inf)``.
    """
    oof = np.asarray(oof_pred, dtype=np.float64).ravel()
    y = np.asarray(y_train, dtype=np.float64).ravel()
    fold = np.asarray(fold_of, dtype=np.int64).ravel()
    per_fold = np.asarray(per_fold_test_pred, dtype=np.float64)
    if per_fold.ndim != 2:
        raise ValueError(f"per_fold_test_pred must be (K, n_test); got {per_fold.shape}")
    n = oof.size
    n_test = per_fold.shape[1]
    k = int(np.ceil((1.0 - alpha) * (n + 1)))
    if k > n:
        return np.full(n_test, -np.inf), np.full(n_test, np.inf)

    resid = np.abs(y - oof)
    lo = np.empty(n_test, dtype=np.float64)
    hi = np.empty(n_test, dtype=np.float64)
    chunk = max(1, int(chunk_size))
    for start in range(0, n_test, chunk):
        stop = min(start + chunk, n_test)
        mu = per_fold[:, start:stop][fold]  # (n, chunk)
        hi[start:stop] = np.sort(mu + resid[:, None], axis=0)[k - 1]
        lo[start:stop] = np.sort(mu - resid[:, None], axis=0)[n - k]
    return lo, hi
