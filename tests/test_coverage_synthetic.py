"""Gate 3 (the most important one): end-to-end coverage on data with a known Bayes rule.

We feed the *exact* Bayes posterior in as the "model", so any deviation from the
nominal rate is attributable to the conformal machinery and not to model error.
"""

from __future__ import annotations

import numpy as np
import pytest

from conformal_tabpfn.conformal import (
    cvplus_regression_interval,
    cvplus_sets,
    split_conformal_sets,
)
from conformal_tabpfn.scores import compute_scores, draw_u
from conftest import gaussian_mixture_bayes, regression_oracle

ALPHA = 0.1
N_TEST = 40
REPS = 600  # >= 500 as required by the spec


def _binomial_se(p: float, n: int) -> float:
    return float(np.sqrt(p * (1.0 - p) / n))


# --------------------------------------------------------------------------- #
# Split conformal
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n_cal", [50, 200, 1000])
@pytest.mark.parametrize("score", ["lac", "aps", "raps"])
def test_split_conformal_hits_nominal_coverage(n_cal: int, score: str) -> None:
    rng = np.random.default_rng(1000 + n_cal)
    covered: list[float] = []
    for _ in range(REPS):
        _, y_cal, p_cal = gaussian_mixture_bayes(n_cal, rng)
        _, y_test, p_test = gaussian_mixture_bayes(N_TEST, rng)
        u_cal = draw_u(n_cal, rng)
        u_test = draw_u(N_TEST, rng)
        s_cal = compute_scores(p_cal, score, u=u_cal)
        s_test = compute_scores(p_test, score, u=u_test)
        sets, _ = split_conformal_sets(s_cal, y_cal, s_test, ALPHA)
        covered.append(float(sets[np.arange(N_TEST), y_test].mean()))

    cov = float(np.mean(covered))
    # Points inside one replicate share a qhat, so the honest standard error is
    # the between-replicate one, not sqrt(p(1-p)/(REPS*N_TEST)).
    se = float(np.std(covered, ddof=1) / np.sqrt(REPS))
    assert abs(cov - (1.0 - ALPHA)) <= 3 * se + _binomial_se(1 - ALPHA, n_cal + 1), (
        f"split/{score}/n_cal={n_cal}: coverage {cov:.4f} vs nominal {1 - ALPHA}"
    )
    assert cov >= 1.0 - ALPHA - 3 * se


# --------------------------------------------------------------------------- #
# CV+
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n_train,n_folds", [(50, 5), (200, 10), (1000, 10)])
@pytest.mark.parametrize("score", ["lac", "aps"])
def test_cvplus_coverage(n_train: int, n_folds: int, score: str) -> None:
    """CV+ guarantees >= 1 - 2*alpha; with an oracle model it should sit near 1 - alpha."""
    rng = np.random.default_rng(2000 + n_train)
    covered: list[float] = []
    for _ in range(REPS):
        _, y_tr, p_tr = gaussian_mixture_bayes(n_train, rng)
        _, y_te, p_te = gaussian_mixture_bayes(N_TEST, rng)
        u_tr = draw_u(n_train, rng)
        u_te = draw_u(N_TEST, rng)
        s_tr = compute_scores(p_tr, score, u=u_tr)
        s_te = compute_scores(p_te, score, u=u_te)

        oof_true = s_tr[np.arange(n_train), y_tr]
        # Oracle model: every fold model is the Bayes rule, so all K test score
        # matrices coincide.  This isolates the CV+ aggregation rule itself.
        per_fold = np.repeat(s_te[None, :, :], n_folds, axis=0)
        fold_of = rng.integers(0, n_folds, size=n_train)

        sets = cvplus_sets(oof_true, per_fold, fold_of, ALPHA)
        covered.append(float(sets[np.arange(N_TEST), y_te].mean()))

    cov = float(np.mean(covered))
    se = float(np.std(covered, ddof=1) / np.sqrt(REPS))
    assert cov >= 1.0 - 2 * ALPHA, f"CV+ broke its own guarantee: {cov:.4f}"
    assert cov >= 1.0 - ALPHA - 0.02, f"CV+ empirically low: {cov:.4f}"
    assert cov <= 1.0 - ALPHA + 0.05 + 3 * se, f"CV+ implausibly conservative: {cov:.4f}"


def test_cvplus_chunking_is_exact() -> None:
    """Chunking is a memory optimisation and must not change a single bit."""
    rng = np.random.default_rng(4)
    n, n_folds, n_test, n_c = 137, 7, 91, 4
    oof = rng.random(n)
    per_fold = rng.random((n_folds, n_test, n_c))
    fold_of = rng.integers(0, n_folds, size=n)
    ref = cvplus_sets(oof, per_fold, fold_of, ALPHA, chunk_size=10**9)
    for chunk in (1, 7, 32, 256):
        assert np.array_equal(cvplus_sets(oof, per_fold, fold_of, ALPHA, chunk_size=chunk), ref)


def test_cvplus_matches_naive_reference() -> None:
    """The vectorised rule must equal a literal transcription of the paper."""
    rng = np.random.default_rng(5)
    n, n_folds, n_test, n_c = 40, 5, 12, 3
    oof = rng.random(n)
    per_fold = rng.random((n_folds, n_test, n_c))
    fold_of = rng.integers(0, n_folds, size=n)

    expected = np.zeros((n_test, n_c), dtype=bool)
    for j in range(n_test):
        for c in range(n_c):
            count = sum(1 for i in range(n) if oof[i] >= per_fold[fold_of[i], j, c])
            expected[j, c] = (1 + count) / (n + 1) > ALPHA

    assert np.array_equal(cvplus_sets(oof, per_fold, fold_of, ALPHA), expected)


# --------------------------------------------------------------------------- #
# Regression
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n_train", [50, 200])
def test_cvplus_regression_interval_coverage(n_train: int) -> None:
    rng = np.random.default_rng(3000 + n_train)
    n_folds = 10
    covered: list[float] = []
    widths: list[float] = []
    for _ in range(REPS):
        _, y_tr, mu_tr = regression_oracle(n_train, rng)
        _, y_te, mu_te = regression_oracle(N_TEST, rng)
        per_fold = np.repeat(mu_te[None, :], n_folds, axis=0)
        fold_of = rng.integers(0, n_folds, size=n_train)
        lo, hi = cvplus_regression_interval(mu_tr, y_tr, per_fold, fold_of, ALPHA)
        covered.append(float(np.mean((y_te >= lo) & (y_te <= hi))))
        widths.append(float(np.mean(hi - lo)))

    cov = float(np.mean(covered))
    se = float(np.std(covered, ddof=1) / np.sqrt(REPS))
    assert cov >= 1.0 - 2 * ALPHA
    assert cov >= 1.0 - ALPHA - 0.02, f"regression CV+ low: {cov:.4f}"
    assert cov <= 1.0 - ALPHA + 0.05 + 3 * se
    # Sanity: with sigma = 1 the oracle 90% interval is about +/- 1.645.
    assert 2.5 < float(np.mean(widths)) < 4.5
