"""Gate 4: prove the harness can actually detect a calibration leak.

If we deliberately break the protocol by calibrating on points the model has
already memorised, coverage must fall measurably below 1 - alpha.  A test suite
that stays green under a deliberate leak is not sensitive enough to be trusted
on the real pipeline.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from conformal_tabpfn.conformal import conformal_quantile, split_conformal_sets
from conformal_tabpfn.scores import compute_scores, draw_u
from conftest import gaussian_mixture_bayes

ALPHA = 0.1
N_TRAIN = 150
N_CAL = 150
N_TEST = 200
REPS = 60


def _fit_predict(
    X_fit: np.ndarray, y_fit: np.ndarray, X_eval: list[np.ndarray], seed: int
) -> list[np.ndarray]:
    """A model that memorises its training set (fully-grown trees, no bagging noise)."""
    clf = RandomForestClassifier(
        n_estimators=60, bootstrap=False, min_samples_leaf=1, random_state=seed, n_jobs=1
    )
    clf.fit(X_fit, y_fit)
    return [clf.predict_proba(X) for X in X_eval]


def _run(leak: bool, score: str = "lac") -> tuple[float, float]:
    """Return (mean coverage, mean qhat) for the honest or leaking protocol."""
    rng = np.random.default_rng(4242)
    covs: list[float] = []
    qs: list[float] = []
    for rep in range(REPS):
        X_tr, y_tr, _ = gaussian_mixture_bayes(N_TRAIN, rng)
        X_cal, y_cal, _ = gaussian_mixture_bayes(N_CAL, rng)
        X_te, y_te, _ = gaussian_mixture_bayes(N_TEST, rng)

        if leak:
            # THE BUG: calibration points are inside the training set, so their
            # scores are those of memorised points and are far too small.
            X_fit = np.vstack([X_tr, X_cal])
            y_fit = np.concatenate([y_tr, y_cal])
        else:
            X_fit, y_fit = X_tr, y_tr

        p_cal, p_te = _fit_predict(X_fit, y_fit, [X_cal, X_te], seed=rep)
        s_cal = compute_scores(p_cal, score, u=draw_u(N_CAL, rng))
        s_te = compute_scores(p_te, score, u=draw_u(N_TEST, rng))
        sets, qhat = split_conformal_sets(s_cal, y_cal, s_te, ALPHA)
        covs.append(float(sets[np.arange(N_TEST), y_te].mean()))
        qs.append(qhat)
    return float(np.mean(covs)), float(np.mean(qs))


def test_honest_protocol_covers_and_leak_undercovers() -> None:
    cov_honest, q_honest = _run(leak=False)
    cov_leak, q_leak = _run(leak=True)

    # The honest split protocol must be at (or above) nominal.
    assert cov_honest >= 1.0 - ALPHA - 0.02, f"honest protocol under-covers: {cov_honest:.4f}"
    # The leak must be detected, not merely be "a bit different".
    assert cov_leak < 1.0 - ALPHA - 0.02, (
        f"LEAK NOT DETECTED: leaking coverage {cov_leak:.4f} is not clearly below "
        f"{1 - ALPHA}. The harness is not sensitive enough to trust."
    )
    assert cov_leak < cov_honest
    # And the mechanism must be the one we think it is: a too-small threshold.
    assert q_leak < q_honest


def test_leak_detector_is_not_trivially_always_failing() -> None:
    """Guard against a test that would 'detect' a leak even when there is none."""
    cov_honest, _ = _run(leak=False, score="aps")
    assert cov_honest >= 1.0 - ALPHA - 0.02


def test_shuffled_calibration_labels_destroy_coverage() -> None:
    """A second, independent way for the suite to notice a broken pipeline."""
    rng = np.random.default_rng(9)
    _, y_cal, p_cal = gaussian_mixture_bayes(400, rng)
    _, y_te, p_te = gaussian_mixture_bayes(400, rng)
    s_cal = compute_scores(p_cal, "lac", u=draw_u(400, rng))
    s_te = compute_scores(p_te, "lac", u=draw_u(400, rng))

    good, _ = split_conformal_sets(s_cal, y_cal, s_te, ALPHA)
    bad, _ = split_conformal_sets(s_cal, rng.permutation(y_cal), s_te, ALPHA)
    cov_good = float(good[np.arange(400), y_te].mean())
    cov_bad = float(bad[np.arange(400), y_te].mean())
    assert cov_good >= 1.0 - ALPHA - 0.03
    assert cov_bad > cov_good  # scrambled labels inflate qhat -> larger, sloppier sets
    assert bad.sum() > good.sum()


@pytest.mark.parametrize("n_cal", [5, 8])
def test_starved_calibration_returns_everything(n_cal: int) -> None:
    """Too few calibration points must yield inf, not a silently invalid quantile."""
    rng = np.random.default_rng(2)
    assert conformal_quantile(rng.random(n_cal), ALPHA) == np.inf
