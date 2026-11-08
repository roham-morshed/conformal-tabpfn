"""Gate 5: class-conditional (Mondrian) variants close the per-class coverage gap."""

from __future__ import annotations

import numpy as np
import pytest

from conformal_tabpfn.conformal import (
    cvplus_sets,
    mondrian_cvplus_sets,
    mondrian_split_conformal_sets,
    split_conformal_sets,
)
from conformal_tabpfn.metrics import classification_metrics
from conformal_tabpfn.scores import compute_scores, draw_u
from conftest import gaussian_mixture_bayes

ALPHA = 0.1
# Strongly imbalanced 3-class problem: the rare class is where marginal-only
# methods are expected to under-cover.
PRIORS = np.array([0.80, 0.15, 0.05])
MEANS = np.array([[0.0, 0.0], [1.6, 0.0], [0.8, 1.5]])
REPS = 250
N_CAL = 600
N_TEST = 600


def _sample(n: int, rng: np.random.Generator):
    return gaussian_mixture_bayes(n, rng, means=MEANS, priors=PRIORS)


def _accumulate(per_class: list[dict[int, float]], n_c: int) -> np.ndarray:
    """Average per-class coverage over replicates, ignoring reps missing a class."""
    out = np.full(n_c, np.nan)
    for c in range(n_c):
        vals = [d[c] for d in per_class if c in d]
        if vals:
            out[c] = float(np.mean(vals))
    return out


def test_mondrian_split_closes_the_gap_versus_marginal_split() -> None:
    rng = np.random.default_rng(77)
    marg: list[dict[int, float]] = []
    mond: list[dict[int, float]] = []
    size_marg: list[float] = []
    size_mond: list[float] = []

    for _ in range(REPS):
        _, y_cal, p_cal = _sample(N_CAL, rng)
        _, y_te, p_te = _sample(N_TEST, rng)
        s_cal = compute_scores(p_cal, "lac", u=draw_u(N_CAL, rng))
        s_te = compute_scores(p_te, "lac", u=draw_u(N_TEST, rng))

        sets_m, _ = split_conformal_sets(s_cal, y_cal, s_te, ALPHA)
        sets_c, _, _ = mondrian_split_conformal_sets(s_cal, y_cal, s_te, ALPHA, n_classes=3)

        mm = classification_metrics(sets_m, y_te, ALPHA, n_classes=3)
        mc = classification_metrics(sets_c, y_te, ALPHA, n_classes=3)
        marg.append(mm["per_class_coverage"])
        mond.append(mc["per_class_coverage"])
        size_marg.append(mm["avg_set_size"])
        size_mond.append(mc["avg_set_size"])

    cov_marg = _accumulate(marg, 3)
    cov_mond = _accumulate(mond, 3)
    gap_marg = float(np.nanmax(cov_marg) - np.nanmin(cov_marg))
    gap_mond = float(np.nanmax(cov_mond) - np.nanmin(cov_mond))

    # Every class must be covered at the nominal rate under Mondrian.
    assert np.all(cov_mond >= 1.0 - ALPHA - 0.02), f"per-class coverage {cov_mond}"
    # The marginal method must visibly fail on at least one class, otherwise this
    # test would be vacuous.
    assert float(np.nanmin(cov_marg)) < 1.0 - ALPHA - 0.02, (
        f"marginal split did not under-cover any class ({cov_marg}); "
        "the imbalance is too mild for this test to have teeth"
    )
    assert gap_mond < gap_marg
    # Fairness is not free: it costs efficiency.
    assert float(np.mean(size_mond)) >= float(np.mean(size_marg)) - 1e-9


def test_mondrian_cvplus_closes_the_gap_versus_marginal_cvplus() -> None:
    rng = np.random.default_rng(78)
    n_train, n_folds = 600, 10
    marg: list[dict[int, float]] = []
    mond: list[dict[int, float]] = []

    for _ in range(REPS):
        _, y_tr, p_tr = _sample(n_train, rng)
        _, y_te, p_te = _sample(N_TEST, rng)
        s_tr = compute_scores(p_tr, "lac", u=draw_u(n_train, rng))
        s_te = compute_scores(p_te, "lac", u=draw_u(N_TEST, rng))

        oof_true = s_tr[np.arange(n_train), y_tr]
        per_fold = np.repeat(s_te[None, :, :], n_folds, axis=0)
        fold_of = rng.integers(0, n_folds, size=n_train)

        sets_m = cvplus_sets(oof_true, per_fold, fold_of, ALPHA)
        sets_c, empty = mondrian_cvplus_sets(
            oof_true, y_tr, per_fold, fold_of, ALPHA, n_classes=3
        )
        assert empty == []
        marg.append(classification_metrics(sets_m, y_te, ALPHA, 3)["per_class_coverage"])
        mond.append(classification_metrics(sets_c, y_te, ALPHA, 3)["per_class_coverage"])

    cov_marg = _accumulate(marg, 3)
    cov_mond = _accumulate(mond, 3)
    assert np.all(cov_mond >= 1.0 - ALPHA - 0.02), f"per-class coverage {cov_mond}"
    assert float(np.nanmin(cov_marg)) < 1.0 - ALPHA - 0.02, f"vacuous test: {cov_marg}"
    gap_marg = float(np.nanmax(cov_marg) - np.nanmin(cov_marg))
    gap_mond = float(np.nanmax(cov_mond) - np.nanmin(cov_mond))
    assert gap_mond < gap_marg


def test_starved_class_is_always_included_and_reported() -> None:
    """A class with too few calibration points gets qhat = inf, and we say so."""
    rng = np.random.default_rng(3)
    n_cal = 60
    s_cal = rng.random((n_cal, 3))
    # Only 4 calibration points of class 2: ceil(5 * 0.9) = 5 > 4  ->  inf.
    y_cal = np.array([0] * 30 + [1] * 26 + [2] * 4)
    s_te = rng.random((25, 3))
    sets, qhat, starved = mondrian_split_conformal_sets(s_cal, y_cal, s_te, ALPHA, n_classes=3)
    assert starved == [2]
    assert np.isinf(qhat[2])
    assert np.all(sets[:, 2])


def test_mondrian_cvplus_handles_a_class_with_no_training_points() -> None:
    rng = np.random.default_rng(4)
    n_train, n_folds, n_test = 50, 5, 10
    oof = rng.random(n_train)
    y_tr = rng.integers(0, 2, size=n_train)  # class 2 never appears
    per_fold = rng.random((n_folds, n_test, 3))
    fold_of = rng.integers(0, n_folds, size=n_train)
    sets, empty = mondrian_cvplus_sets(oof, y_tr, per_fold, fold_of, ALPHA, n_classes=3)
    assert empty == [2]
    assert np.all(sets[:, 2])


@pytest.mark.parametrize("chunk", [1, 8, 256])
def test_mondrian_cvplus_chunking_is_exact(chunk: int) -> None:
    rng = np.random.default_rng(6)
    n_train, n_folds, n_test, n_c = 120, 6, 47, 4
    oof = rng.random(n_train)
    y_tr = rng.integers(0, n_c, size=n_train)
    per_fold = rng.random((n_folds, n_test, n_c))
    fold_of = rng.integers(0, n_folds, size=n_train)
    ref, _ = mondrian_cvplus_sets(
        oof, y_tr, per_fold, fold_of, ALPHA, n_classes=n_c, chunk_size=10**9
    )
    got, _ = mondrian_cvplus_sets(
        oof, y_tr, per_fold, fold_of, ALPHA, n_classes=n_c, chunk_size=chunk
    )
    assert np.array_equal(got, ref)


def test_mondrian_cvplus_matches_naive_reference() -> None:
    """Literal transcription of the spec's per-class loop."""
    rng = np.random.default_rng(7)
    n_train, n_folds, n_test, n_c = 45, 5, 11, 3
    oof = rng.random(n_train)
    y_tr = rng.integers(0, n_c, size=n_train)
    per_fold = rng.random((n_folds, n_test, n_c))
    fold_of = rng.integers(0, n_folds, size=n_train)

    expected = np.zeros((n_test, n_c), dtype=bool)
    for c in range(n_c):
        idx = [i for i in range(n_train) if y_tr[i] == c]
        n_class = len(idx)
        for j in range(n_test):
            count = sum(1 for i in idx if oof[i] >= per_fold[fold_of[i], j, c])
            expected[j, c] = (1 + count) / (n_class + 1) > ALPHA

    got, _ = mondrian_cvplus_sets(oof, y_tr, per_fold, fold_of, ALPHA, n_classes=n_c)
    assert np.array_equal(got, expected)
