"""Gate 7: the vectorised CV+ regression interval equals an obviously-correct reference."""

from __future__ import annotations

import numpy as np
import pytest

from conformal_tabpfn.conformal import cvplus_regression_interval, split_conformal_interval
from conformal_tabpfn.metrics import regression_metrics


def _reference_cvplus(
    oof_pred: np.ndarray,
    y_train: np.ndarray,
    per_fold_test_pred: np.ndarray,
    fold_of: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Slow, explicit, line-by-line transcription of Barber et al. (2021).

    Deliberately written with Python loops and no numpy tricks so that it is
    readable by eye; it is the ground truth for the fast implementation.
    """
    n = len(oof_pred)
    n_test = per_fold_test_pred.shape[1]
    k = int(np.ceil((1.0 - alpha) * (n + 1)))
    if k > n:
        return np.full(n_test, -np.inf), np.full(n_test, np.inf)

    resid = [abs(y_train[i] - oof_pred[i]) for i in range(n)]
    lo = np.empty(n_test)
    hi = np.empty(n_test)
    for j in range(n_test):
        upper = sorted(per_fold_test_pred[fold_of[i], j] + resid[i] for i in range(n))
        lower = sorted(per_fold_test_pred[fold_of[i], j] - resid[i] for i in range(n))
        hi[j] = upper[k - 1]
        lo[j] = lower[n - k]
    return lo, hi


@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2])
def test_matches_reference_on_30_points(alpha: float) -> None:
    rng = np.random.default_rng(123)
    n, n_folds, n_test = 30, 5, 8
    y_train = rng.standard_normal(n) * 2.0
    oof_pred = y_train + rng.standard_normal(n) * 0.5
    per_fold = rng.standard_normal((n_folds, n_test))
    fold_of = rng.integers(0, n_folds, size=n)

    lo_ref, hi_ref = _reference_cvplus(oof_pred, y_train, per_fold, fold_of, alpha)
    lo, hi = cvplus_regression_interval(oof_pred, y_train, per_fold, fold_of, alpha)
    assert np.allclose(lo, lo_ref)
    assert np.allclose(hi, hi_ref)
    assert np.all(hi >= lo)


def test_jackknife_plus_is_cvplus_with_k_equals_n() -> None:
    """K = n reduces CV+ to jackknife+; check it against the same reference."""
    rng = np.random.default_rng(321)
    n, n_test = 30, 6
    y_train = rng.standard_normal(n)
    oof_pred = y_train + rng.standard_normal(n) * 0.3
    per_fold = rng.standard_normal((n, n_test))
    fold_of = np.arange(n)
    lo_ref, hi_ref = _reference_cvplus(oof_pred, y_train, per_fold, fold_of, 0.1)
    lo, hi = cvplus_regression_interval(oof_pred, y_train, per_fold, fold_of, 0.1)
    assert np.allclose(lo, lo_ref) and np.allclose(hi, hi_ref)


@pytest.mark.parametrize("chunk", [1, 3, 1024])
def test_chunking_is_exact(chunk: int) -> None:
    rng = np.random.default_rng(55)
    n, n_folds, n_test = 60, 6, 37
    y_train = rng.standard_normal(n)
    oof_pred = y_train + rng.standard_normal(n) * 0.4
    per_fold = rng.standard_normal((n_folds, n_test))
    fold_of = rng.integers(0, n_folds, size=n)
    ref = cvplus_regression_interval(oof_pred, y_train, per_fold, fold_of, 0.1, chunk_size=10**9)
    got = cvplus_regression_interval(oof_pred, y_train, per_fold, fold_of, 0.1, chunk_size=chunk)
    assert np.allclose(got[0], ref[0]) and np.allclose(got[1], ref[1])


def test_too_few_points_gives_infinite_interval() -> None:
    rng = np.random.default_rng(1)
    n, n_test = 5, 4  # k = ceil(0.9 * 6) = 6 > 5
    lo, hi = cvplus_regression_interval(
        rng.standard_normal(n), rng.standard_normal(n), rng.standard_normal((5, n_test)),
        np.arange(n), 0.1,
    )
    assert np.all(np.isneginf(lo)) and np.all(np.isposinf(hi))


def test_split_interval_is_symmetric_around_mu() -> None:
    rng = np.random.default_rng(2)
    resid = np.abs(rng.standard_normal(200))
    mu = rng.standard_normal(50)
    lo, hi = split_conformal_interval(resid, mu, 0.1)
    assert np.allclose((lo + hi) / 2.0, mu)
    assert np.allclose(hi - lo, (hi - lo)[0])  # constant width


def test_regression_metrics_winkler_score() -> None:
    """Winkler score = width when covered, width + (2/alpha) * miss otherwise."""
    lo = np.array([0.0, 0.0, 0.0])
    hi = np.array([2.0, 2.0, 2.0])
    y = np.array([1.0, 3.0, -1.0])  # covered, above by 1, below by 1
    alpha = 0.1
    m = regression_metrics(lo, hi, y, alpha)
    assert m["coverage"] == pytest.approx(1 / 3)
    assert m["avg_width"] == pytest.approx(2.0)
    expected = np.mean([2.0, 2.0 + 20 * 1.0, 2.0 + 20 * 1.0])
    assert m["interval_score"] == pytest.approx(expected)


def test_regression_metrics_report_infinite_widths_rather_than_hiding_them() -> None:
    lo = np.array([-np.inf, 0.0])
    hi = np.array([np.inf, 2.0])
    y = np.array([5.0, 1.0])
    m = regression_metrics(lo, hi, y, 0.1)
    assert m["coverage"] == 1.0
    assert m["frac_infinite_width"] == 0.5
    assert m["avg_width"] == pytest.approx(2.0)  # averaged over finite ones only
