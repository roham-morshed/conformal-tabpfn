"""Shared fixtures and synthetic generators with closed-form Bayes probabilities."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def gaussian_mixture_bayes(
    n: int,
    rng: np.random.Generator,
    means: np.ndarray | None = None,
    priors: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Isotropic Gaussian class-conditionals, so the Bayes posterior is closed form.

    ``X | Y = c ~ N(mu_c, I)`` gives
    ``p(c | x) = pi_c exp(-||x - mu_c||^2 / 2) / sum_j (...)``.

    Returns ``(X, y, P)`` where ``P`` is the exact Bayes posterior, used as a
    perfectly-calibrated "model" so that the tests probe the conformal machinery
    and nothing else.
    """
    if means is None:
        means = np.array([[0.0, 0.0], [2.2, 0.0], [1.1, 2.0]])
    n_c = means.shape[0]
    if priors is None:
        priors = np.full(n_c, 1.0 / n_c)
    priors = np.asarray(priors, dtype=np.float64)
    priors = priors / priors.sum()

    y = rng.choice(n_c, size=n, p=priors)
    X = means[y] + rng.standard_normal((n, means.shape[1]))
    log_lik = -0.5 * ((X[:, None, :] - means[None, :, :]) ** 2).sum(axis=2)
    log_post = log_lik + np.log(priors)[None, :]
    log_post -= log_post.max(axis=1, keepdims=True)
    P = np.exp(log_post)
    P /= P.sum(axis=1, keepdims=True)
    return X, y, P


def regression_oracle(
    n: int, rng: np.random.Generator, sigma: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``y = sin(3 x) + 2 x + eps``; returns ``(x, y, mu_oracle)``."""
    x = rng.uniform(-2.0, 2.0, size=n)
    mu = np.sin(3.0 * x) + 2.0 * x
    y = mu + sigma * rng.standard_normal(n)
    return x, y, mu


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20240517)
