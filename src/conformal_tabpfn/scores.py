"""Nonconformity scores for conformal classification and regression.

Convention (used everywhere in this package): **higher score = more
nonconforming**.  Every classification score function returns an ``(n, C)``
matrix giving the score of *every* candidate label at every example, not just
the score of the true label.  CV+ needs the full matrix (see conformal.py).

References
----------
LAC / THR : Sadinle, Lei & Wasserman (2019), "Least Ambiguous Set-Valued
    Classifiers with Bounded Error Levels", JASA.
APS       : Romano, Sesia & Candes (2020), "Classification with Valid and
    Adaptive Coverage", NeurIPS.
RAPS      : Angelopoulos, Bates, Malik & Jordan (2021), "Uncertainty Sets for
    Image Classifiers using Conformal Prediction", ICLR.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "SCORE_NAMES",
    "absolute_residual",
    "aps_scores",
    "compute_scores",
    "draw_u",
    "lac_scores",
    "normalized_residual",
    "raps_scores",
]

SCORE_NAMES: tuple[str, ...] = ("lac", "aps", "raps")

_EPS = 1e-12


def draw_u(n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw the randomisation variables used by APS/RAPS.

    One draw **per example**, shared across all candidate labels of that
    example.  Drawing per (example, label) instead would destroy the
    exchangeability argument: the APS score must be a deterministic function of
    (x, y, U) with a single U attached to the example.
    """
    return rng.uniform(0.0, 1.0, size=int(n))


def _check_probs(probs: np.ndarray) -> np.ndarray:
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError(f"probs must be 2-D (n, C); got shape {p.shape}")
    if not np.all(np.isfinite(p)):
        raise ValueError("probs contains non-finite values")
    if p.min() < -1e-9 or p.max() > 1.0 + 1e-9:
        raise ValueError(f"probs outside [0,1]: min={p.min()}, max={p.max()}")
    return np.clip(p, 0.0, 1.0)


def lac_scores(probs: np.ndarray) -> np.ndarray:
    """LAC / THR score: ``s(x, y) = 1 - p_hat(y | x)``.  Shape ``(n, C)``."""
    return 1.0 - _check_probs(probs)


def _sorted_view(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (order, p_sorted, inclusive cumsum) with probabilities descending."""
    # ``kind="stable"`` makes tie-breaking deterministic, which the determinism
    # test relies on.
    order = np.argsort(-probs, axis=1, kind="stable")
    p_sorted = np.take_along_axis(probs, order, axis=1)
    cum = np.cumsum(p_sorted, axis=1)
    return order, p_sorted, cum


def _scatter_back(order: np.ndarray, s_sorted: np.ndarray) -> np.ndarray:
    """Move scores from sorted-label order back to the original label order."""
    out = np.empty_like(s_sorted)
    np.put_along_axis(out, order, s_sorted, axis=1)
    return out


def aps_scores(
    probs: np.ndarray,
    u: np.ndarray | None = None,
    randomized: bool = True,
) -> np.ndarray:
    """APS score (Romano et al. 2020).

    With the labels sorted by decreasing probability, the label sitting at
    sorted rank ``r`` (1-indexed) gets

        s = cumsum[r] - U * p_sorted[r]           (randomized)
        s = cumsum[r]                             (randomized=False)

    where ``cumsum`` is the *inclusive* cumulative sum and ``U ~ Unif(0, 1)`` is
    drawn once per example.
    """
    p = _check_probs(probs)
    order, p_sorted, cum = _sorted_view(p)
    if randomized:
        if u is None:
            raise ValueError(
                "randomized=True requires an explicit `u` array (use draw_u) so "
                "that runs stay reproducible."
            )
        u_arr = np.asarray(u, dtype=np.float64)
        if u_arr.shape != (p.shape[0],):
            raise ValueError(f"u must have shape ({p.shape[0]},); got {u_arr.shape}")
        s_sorted = cum - u_arr[:, None] * p_sorted
    else:
        s_sorted = cum
    return _scatter_back(order, s_sorted)


def raps_scores(
    probs: np.ndarray,
    u: np.ndarray | None = None,
    randomized: bool = True,
    lam: float = 0.01,
    k_reg: int = 1,
) -> np.ndarray:
    """RAPS score (Angelopoulos et al. 2021).

        s = cumsum[r] - U * p_sorted[r] + lam * max(0, r - k_reg)

    with ``r`` 1-indexed.  The extra term penalises deep ranks and therefore
    shrinks the tail of large prediction sets.  ``lam=0`` reduces exactly to APS.
    """
    p = _check_probs(probs)
    order, p_sorted, cum = _sorted_view(p)
    if randomized:
        if u is None:
            raise ValueError("randomized=True requires an explicit `u` array (use draw_u).")
        u_arr = np.asarray(u, dtype=np.float64)
        if u_arr.shape != (p.shape[0],):
            raise ValueError(f"u must have shape ({p.shape[0]},); got {u_arr.shape}")
        s_sorted = cum - u_arr[:, None] * p_sorted
    else:
        s_sorted = cum
    ranks = np.arange(1, p.shape[1] + 1, dtype=np.float64)  # 1-indexed rank r
    s_sorted = s_sorted + float(lam) * np.maximum(0.0, ranks - float(k_reg))[None, :]
    return _scatter_back(order, s_sorted)


def compute_scores(
    probs: np.ndarray,
    method: str,
    u: np.ndarray | None = None,
    randomized: bool = True,
    lam: float = 0.01,
    k_reg: int = 1,
) -> np.ndarray:
    """Dispatch to a named score function.  Returns an ``(n, C)`` score matrix."""
    m = method.lower()
    if m in ("lac", "thr"):
        return lac_scores(probs)
    if m == "aps":
        return aps_scores(probs, u=u, randomized=randomized)
    if m == "raps":
        return raps_scores(probs, u=u, randomized=randomized, lam=lam, k_reg=k_reg)
    raise ValueError(f"unknown score method {method!r}; expected one of {SCORE_NAMES}")


def tune_k_reg(
    probs_tune: np.ndarray,
    y_tune: np.ndarray,
    alpha: float,
    candidates: tuple[int, ...] = (1, 2, 3, 4, 5),
) -> int:
    """Standard RAPS ``k_reg`` tuning on a held-out slice.

    Picks the smallest k such that the ``1 - alpha`` quantile of the true-label
    rank is covered, i.e. k_reg = the (1-alpha) empirical quantile of the rank of
    the true label (Angelopoulos et al. 2021, App. "Choosing k_reg").
    """
    p = _check_probs(probs_tune)
    order = np.argsort(-p, axis=1, kind="stable")
    ranks = np.argmax(order == np.asarray(y_tune)[:, None], axis=1) + 1
    target = float(np.quantile(ranks, 1.0 - alpha, method="higher"))
    feasible = [k for k in candidates if k >= target]
    return int(feasible[0]) if feasible else int(max(candidates))


def absolute_residual(y: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Regression score ``|y - mu_hat(x)|``."""
    return np.abs(np.asarray(y, dtype=np.float64) - np.asarray(mu, dtype=np.float64))


def normalized_residual(
    y: np.ndarray, mu: np.ndarray, sigma: np.ndarray, eps: float = 1e-6
) -> np.ndarray:
    """Locally-adaptive regression score ``|y - mu| / (sigma + eps)``.

    ``sigma`` comes from a second model fitted on absolute residuals of the mean
    model (secondary result in the paper).
    """
    sig = np.maximum(np.asarray(sigma, dtype=np.float64), 0.0)
    return absolute_residual(y, mu) / (sig + float(eps))
