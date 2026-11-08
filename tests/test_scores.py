"""Gate 2: nonconformity scores, including the scatter-back to original labels."""

from __future__ import annotations

import numpy as np
import pytest

from conformal_tabpfn.scores import (
    aps_scores,
    compute_scores,
    draw_u,
    lac_scores,
    raps_scores,
    tune_k_reg,
)


@pytest.fixture
def probs3x4() -> np.ndarray:
    """A 3x4 probability matrix with a hand-checkable argsort."""
    return np.array(
        [
            [0.1, 0.6, 0.2, 0.1],  # desc order: 1 (0.6), 2 (0.2), 0 (0.1), 3 (0.1)
            [0.4, 0.3, 0.2, 0.1],  # desc order: 0, 1, 2, 3
            [0.25, 0.25, 0.25, 0.25],  # all tied -> stable order 0, 1, 2, 3
        ]
    )


# --------------------------------------------------------------------------- #
# LAC
# --------------------------------------------------------------------------- #


def test_lac_in_unit_interval_and_monotone() -> None:
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(5), size=200)
    s = lac_scores(p)
    assert s.shape == p.shape
    assert s.min() >= 0.0 and s.max() <= 1.0
    # higher probability => lower (less nonconforming) score, strictly
    flat_p, flat_s = p.ravel(), s.ravel()
    order = np.argsort(flat_p)
    assert np.all(np.diff(flat_s[order]) <= 1e-12)
    assert np.allclose(s, 1.0 - p)


def test_lac_rejects_invalid_probabilities() -> None:
    with pytest.raises(ValueError):
        lac_scores(np.array([[1.5, -0.5]]))
    with pytest.raises(ValueError):
        lac_scores(np.array([0.5, 0.5]))  # not 2-D
    with pytest.raises(ValueError):
        lac_scores(np.array([[np.nan, 1.0]]))


# --------------------------------------------------------------------------- #
# APS
# --------------------------------------------------------------------------- #


def test_aps_nonrandomized_top1_equals_top_probability(probs3x4: np.ndarray) -> None:
    s = aps_scores(probs3x4, randomized=False)
    top = probs3x4.argmax(axis=1)
    assert np.allclose(s[np.arange(3), top], probs3x4.max(axis=1))


def test_aps_scatter_back_is_correct_by_hand(probs3x4: np.ndarray) -> None:
    """Row 0 is [0.1, 0.6, 0.2, 0.1]; descending order is labels 1, 2, 0, 3.

    Inclusive cumsum along that order: 0.6, 0.8, 0.9, 1.0.  So the scores must
    land as label1 -> 0.6, label2 -> 0.8, label0 -> 0.9, label3 -> 1.0.
    """
    s = aps_scores(probs3x4, randomized=False)
    assert np.allclose(s[0], [0.9, 0.6, 0.8, 1.0])
    # Row 1 is already sorted descending: 0.4, 0.7, 0.9, 1.0
    assert np.allclose(s[1], [0.4, 0.7, 0.9, 1.0])
    # Row 2 is fully tied; stable argsort keeps label order 0, 1, 2, 3.
    assert np.allclose(s[2], [0.25, 0.5, 0.75, 1.0])


def test_aps_last_label_score_is_one(probs3x4: np.ndarray) -> None:
    """The least likely label always has inclusive cumsum 1 (non-randomized)."""
    s = aps_scores(probs3x4, randomized=False)
    assert np.allclose(s.max(axis=1), 1.0)


def test_aps_randomized_uses_one_u_per_example() -> None:
    """The randomised score must shift by exactly U * p_sorted, per example."""
    p = np.array([[0.5, 0.3, 0.2], [0.2, 0.3, 0.5]])
    u = np.array([0.25, 0.75])
    s_rand = aps_scores(p, u=u, randomized=True)
    s_plain = aps_scores(p, randomized=False)
    delta = s_plain - s_rand
    # delta[i, j] == u[i] * p[i, j]
    assert np.allclose(delta, u[:, None] * p)


def test_aps_randomized_requires_explicit_u() -> None:
    with pytest.raises(ValueError, match="requires an explicit"):
        aps_scores(np.array([[0.5, 0.5]]), randomized=True)
    with pytest.raises(ValueError, match="shape"):
        aps_scores(np.array([[0.5, 0.5]]), u=np.array([0.1, 0.2]), randomized=True)


def test_aps_randomized_scores_stay_in_range() -> None:
    rng = np.random.default_rng(3)
    p = rng.dirichlet(np.ones(6), size=500)
    s = aps_scores(p, u=draw_u(500, rng), randomized=True)
    assert s.min() >= -1e-12
    assert s.max() <= 1.0 + 1e-12


# --------------------------------------------------------------------------- #
# RAPS
# --------------------------------------------------------------------------- #


def test_raps_with_zero_lambda_equals_aps(probs3x4: np.ndarray) -> None:
    rng = np.random.default_rng(1)
    u = draw_u(probs3x4.shape[0], rng)
    assert np.allclose(
        raps_scores(probs3x4, u=u, lam=0.0), aps_scores(probs3x4, u=u)
    )
    assert np.allclose(
        raps_scores(probs3x4, randomized=False, lam=0.0),
        aps_scores(probs3x4, randomized=False),
    )


def test_raps_penalty_is_rank_based_and_one_indexed(probs3x4: np.ndarray) -> None:
    """With k_reg = 1, ranks 1..4 get penalties 0, lam, 2*lam, 3*lam."""
    lam = 0.1
    s_aps = aps_scores(probs3x4, randomized=False)
    s_raps = raps_scores(probs3x4, randomized=False, lam=lam, k_reg=1)
    # Row 0: order is labels 1, 2, 0, 3 at ranks 1, 2, 3, 4.
    expected_penalty_row0 = np.array([2 * lam, 0.0, 1 * lam, 3 * lam])
    assert np.allclose(s_raps[0] - s_aps[0], expected_penalty_row0)
    # Penalty never decreases with rank and is never negative.
    assert np.all(s_raps - s_aps >= -1e-12)


def test_raps_larger_lambda_never_shrinks_scores(probs3x4: np.ndarray) -> None:
    s_small = raps_scores(probs3x4, randomized=False, lam=0.01)
    s_big = raps_scores(probs3x4, randomized=False, lam=0.5)
    assert np.all(s_big >= s_small - 1e-12)


def test_tune_k_reg_picks_the_rank_quantile() -> None:
    # True label is always the top-1 -> k_reg should be 1.
    p = np.tile(np.array([0.7, 0.2, 0.1]), (100, 1))
    y = np.zeros(100, dtype=int)
    assert tune_k_reg(p, y, alpha=0.1) == 1
    # True label is always rank 3 -> need k_reg >= 3.
    y3 = np.full(100, 2, dtype=int)
    assert tune_k_reg(p, y3, alpha=0.1) == 3


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", ["lac", "aps", "raps"])
def test_compute_scores_dispatch_matches_direct_call(name: str) -> None:
    rng = np.random.default_rng(5)
    p = rng.dirichlet(np.ones(4), size=50)
    u = draw_u(50, rng)
    direct = {
        "lac": lambda: lac_scores(p),
        "aps": lambda: aps_scores(p, u=u),
        "raps": lambda: raps_scores(p, u=u),
    }[name]()
    assert np.allclose(compute_scores(p, name, u=u), direct)


def test_compute_scores_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="unknown score method"):
        compute_scores(np.array([[0.5, 0.5]]), "not-a-score")
