"""Gate 1: the conformal quantile is the exact order statistic the theory needs."""

from __future__ import annotations

import numpy as np
import pytest

from conformal_tabpfn.conformal import conformal_quantile


def _k(n: int, alpha: float) -> int:
    return int(np.ceil((n + 1) * (1.0 - alpha)))


@pytest.mark.parametrize("n", [5, 9, 10, 19, 20, 35, 39, 77, 99, 100, 199, 200])
@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2])
def test_returns_exact_kth_order_statistic(n: int, alpha: float) -> None:
    """Default ("tight") must return the k-th smallest score, k = ceil((n+1)(1-a))."""
    scores = np.arange(1.0, n + 1.0)  # sorted, distinct
    k = _k(n, alpha)
    if k > n:
        assert conformal_quantile(scores, alpha) == np.inf
        return
    assert conformal_quantile(scores, alpha) == float(k)


@pytest.mark.parametrize("n", [5, 9, 19, 35, 99, 200])
@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2])
def test_shuffling_does_not_change_the_quantile(n: int, alpha: float) -> None:
    rng = np.random.default_rng(0)
    scores = rng.standard_normal(n)
    ref = conformal_quantile(scores, alpha)
    for _ in range(5):
        assert conformal_quantile(rng.permutation(scores), alpha) == ref


def test_too_few_calibration_points_gives_infinity() -> None:
    # (n + 1)(1 - alpha) > n  =>  cannot certify the level at all.
    assert conformal_quantile(np.arange(5.0), 0.1) == np.inf  # k = 6 > 5
    assert conformal_quantile(np.array([]), 0.1) == np.inf


def test_spec_higher_method_is_valid_but_conservative() -> None:
    """The spec's literal numpy call lands one order statistic too high.

    Kept as a pinned fact, not as the default: it is still a valid conformal
    quantile (never smaller than the tight one) but it over-covers.
    """
    n, alpha = 19, 0.1
    scores = np.arange(1.0, n + 1.0)
    tight = conformal_quantile(scores, alpha, method="tight")
    higher = conformal_quantile(scores, alpha, method="higher")
    assert tight == float(_k(n, alpha)) == 18.0
    assert higher == 19.0
    assert higher >= tight


@pytest.mark.parametrize("method", ["tight", "higher"])
def test_both_admissible_methods_actually_cover(method: str) -> None:
    """Monte-Carlo the finite-sample guarantee for both admissible rules."""
    rng = np.random.default_rng(7)
    n_cal, alpha, reps = 99, 0.1, 40000
    draws = rng.random((reps, n_cal + 1))
    cal, test = draws[:, :n_cal], draws[:, n_cal]
    q = np.array([conformal_quantile(row, alpha, method=method) for row in cal])
    cov = float(np.mean(test <= q))
    se = np.sqrt(0.9 * 0.1 / reps)
    assert cov >= 1.0 - alpha - 3 * se, f"{method} under-covers: {cov}"


def test_missing_finite_sample_correction_under_covers() -> None:
    """The real under-coverage trap: quantiling at ``1 - alpha`` directly.

    Dropping the ``(n + 1) / n`` correction and asking numpy for the plain
    ``1 - alpha`` quantile lands *below* the required order statistic and
    genuinely breaks the guarantee -- badly so at small n, which is exactly the
    regime this paper is about.  Measured over 60k replicates at alpha = 0.1:
    0.8612 (n=19), 0.8806 (n=39), 0.8905 (n=99).
    """
    rng = np.random.default_rng(11)
    alpha, reps = 0.1, 40000
    for n_cal, ceiling in ((19, 0.88), (99, 0.897)):
        draws = rng.random((reps, n_cal + 1))
        cal, test = draws[:, :n_cal], draws[:, n_cal]
        k = _k(n_cal, alpha)

        q_naive = np.quantile(cal, 1.0 - alpha, axis=1, method="linear")
        q_tight = np.partition(cal, k - 1, axis=1)[:, k - 1]

        cov_naive = float(np.mean(test <= q_naive))
        cov_tight = float(np.mean(test <= q_tight))

        assert np.all(q_naive <= q_tight + 1e-12)
        assert cov_naive < 1.0 - alpha, "the uncorrected quantile failed to under-cover"
        assert cov_naive < ceiling, f"n={n_cal}: expected clear under-coverage, got {cov_naive}"
        assert cov_tight > cov_naive


def test_linear_at_the_conformal_level_is_conservative_not_broken() -> None:
    """Records an honest negative finding about the project spec (section 4.2).

    The spec asserts that numpy's default linear interpolation *at the conformal
    level* ``k / n`` breaks the finite-sample guarantee.  It does not: the
    virtual index ``k - k / n`` always lies between the k-th and (k+1)-th order
    statistics, so linear interpolation returns something **>= the tight
    quantile** and therefore still covers.  We swept every
    ``n in [3, 800) x alpha in {0.05, 0.1, 0.2}`` and found zero violations.

    ``method`` still matters -- but because we want the *tight* quantile, not
    because linear is invalid here.
    """
    rng = np.random.default_rng(12)
    violations = 0
    for n in range(3, 800):
        for alpha in (0.05, 0.1, 0.2):
            k = _k(n, alpha)
            if k > n:
                continue
            s = np.sort(rng.random(n))
            if np.quantile(s, k / n, method="linear") < s[k - 1] - 1e-15:
                violations += 1
    assert violations == 0


def test_rejects_interpolating_methods_explicitly() -> None:
    with pytest.raises(ValueError, match="not guarantee-preserving"):
        conformal_quantile(np.arange(10.0), 0.1, method="linear")
    with pytest.raises(ValueError, match="not guarantee-preserving"):
        conformal_quantile(np.arange(10.0), 0.1, method="midpoint")


@pytest.mark.parametrize("alpha", [-0.1, 0.0, 1.0, 1.5])
def test_rejects_out_of_range_alpha(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha must be"):
        conformal_quantile(np.arange(10.0), alpha)


def test_inverted_cdf_shortcut_is_not_safe() -> None:
    """Documents why we use np.partition instead of a float quantile level.

    ``k / n * n`` is not always exactly ``k`` in binary floating point, so
    ``np.quantile(..., method="inverted_cdf")`` picks the wrong order statistic
    for some (n, alpha).  n = 35, alpha = 0.2 is one such case.
    """
    n, alpha = 35, 0.2
    scores = np.arange(1.0, n + 1.0)
    k = _k(n, alpha)
    assert conformal_quantile(scores, alpha) == float(k) == 29.0
    assert np.quantile(scores, k / n, method="inverted_cdf") == 30.0
