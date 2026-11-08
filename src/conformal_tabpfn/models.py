"""Model zoo.

Every estimator is exposed through :func:`make_model` and is sklearn-compatible.
Two things here are load-bearing for the paper and are unit-tested:

1.  **Global class alignment.**  ``predict_proba`` must return columns indexed by
    the *global* ``0..C-1`` labelling even when a CV fold happens to contain no
    examples of some class.  sklearn estimators return columns ordered by
    ``self.classes_``, which in that situation is a strict subset -- writing
    those columns straight into an ``(n, C)`` score matrix would silently shift
    every label by one and corrupt every number in the paper.
    :class:`GlobalClassAligned` scatters them back.

2.  **Honest timing.**  ``fit_time`` and ``predict_time`` are measured
    separately, and any lazy one-off initialisation is forced by a documented
    warm-up *before* the clock starts.  For TabPFN ``fit_time`` should be ~0 --
    that fact is the paper's thesis, so it must not be flattered by accident.
"""

from __future__ import annotations

import os
import warnings
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

__all__ = [
    "CLASSIFIER_NAMES",
    "REGRESSOR_NAMES",
    "GlobalClassAligned",
    "gpu_available",
    "make_model",
    "reset_gpu",
    "tabpfn_available",
]

CLASSIFIER_NAMES: tuple[str, ...] = ("tabpfn", "xgboost", "rf", "logreg")
REGRESSOR_NAMES: tuple[str, ...] = ("tabpfn", "xgboost", "rf", "ridge")

# TabPFN refuses to run on CPU above 1000 rows by default ("slow performance").
# The spec requires a WORKING CPU fallback, so we lift that guard explicitly and
# document it here rather than letting a cell die on a machine without CUDA.
os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")

_TABPFN_IMPORT_ERROR: str | None = None


def tabpfn_available() -> bool:
    global _TABPFN_IMPORT_ERROR
    try:
        import tabpfn  # noqa: F401

        return True
    except Exception as exc:
        _TABPFN_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
        return False


def gpu_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def reset_gpu() -> None:
    """Free cached CUDA blocks between experiment cells (spec section 5)."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# --------------------------------------------------------------------------- #
# Global class alignment
# --------------------------------------------------------------------------- #


class GlobalClassAligned(BaseEstimator, ClassifierMixin):
    """Wrap a classifier so ``predict_proba`` always has ``n_classes`` columns.

    A CV fold can easily miss a rare class.  The wrapped estimator then reports
    ``classes_ = [0, 2]`` and returns a 2-column probability matrix; dropping it
    into an ``(n, 3)`` slot would misalign every score.  We allocate the full
    ``(n, n_classes)`` matrix, scatter the fitted columns into their global
    positions with ``model.classes_``, and leave absent classes at probability 0
    (they were never observed in training, so 0 is the honest estimate, and the
    conformal set can still include them via the calibration threshold).
    """

    def __init__(self, estimator: Any, n_classes: int) -> None:
        self.estimator = estimator
        self.n_classes = int(n_classes)

    def fit(self, X: Any, y: np.ndarray) -> GlobalClassAligned:
        y_global = np.asarray(y, dtype=np.int64)
        if y_global.size and y_global.max() >= self.n_classes:
            raise ValueError(
                f"saw class {y_global.max()} but n_classes={self.n_classes}; "
                "labels must be encoded 0..C-1 globally before fitting"
            )
        # Global labels of the classes present in THIS fold, ascending.
        self.seen_classes_ = np.unique(y_global)

        # Relabel to a contiguous 0..m-1 range before fitting.  XGBoost's sklearn
        # wrapper hard-rejects a gap in the label set ("Invalid classes inferred
        # from unique values of y"), which is exactly what happens when a CV fold
        # misses a rare class.  Remapping is a no-op for estimators that tolerate
        # gaps (sklearn sorts classes anyway), so we do it unconditionally rather
        # than special-casing one library.
        y_local = np.searchsorted(self.seen_classes_, y_global)

        self.estimator_ = clone(self.estimator)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.estimator_.fit(X, y_local)

        # Map the estimator's own column order back to global label indices,
        # rather than assuming it matches seen_classes_ positionally.
        local_classes = np.asarray(self.estimator_.classes_, dtype=np.int64)
        self.global_of_column_ = self.seen_classes_[local_classes]
        self.classes_ = np.arange(self.n_classes)
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            local = np.asarray(self.estimator_.predict_proba(X), dtype=np.float64)
        n = local.shape[0]
        if local.shape[1] != self.global_of_column_.size:
            raise ValueError(
                f"predict_proba returned {local.shape[1]} columns but the fitted model "
                f"has {self.global_of_column_.size} classes; cannot align safely"
            )
        # Classes absent from this fold keep probability 0: they were never
        # observed in training, so 0 is the honest estimate.  The conformal
        # threshold can still admit them into the prediction set.
        out = np.zeros((n, self.n_classes), dtype=np.float64)
        out[:, self.global_of_column_] = local
        return out

    def predict(self, X: Any) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)


# --------------------------------------------------------------------------- #
# TabPFN wrapper
# --------------------------------------------------------------------------- #


class TabPFNWrapper(BaseEstimator, ClassifierMixin):
    """TabPFN with CUDA-OOM handling, honest timing, and a CPU fallback.

    TabPFN is a prior-data fitted network: ``fit()`` performs no gradient
    training, it only stores the in-context training set.  That is exactly why
    CV+ is nearly free for it, so we must measure the cost honestly rather than
    hide lazy work inside the timed region.

    ``device_used`` records what actually ran, including a downgrade to CPU
    after repeated OOM, so the result row never claims GPU timing for CPU work.
    """

    def __init__(
        self,
        seed: int = 0,
        task: str = "classification",
        device: str | None = None,
        n_estimators: int = 4,
        max_oom_retries: int = 3,
    ) -> None:
        self.seed = seed
        self.task = task
        self.device = device
        self.n_estimators = n_estimators
        self.max_oom_retries = max_oom_retries

    def _resolve_device(self) -> str:
        if self.device is not None:
            return self.device
        return "cuda" if gpu_available() else "cpu"

    def _build(self, device: str) -> Any:
        from tabpfn import TabPFNClassifier, TabPFNRegressor

        kwargs: dict[str, Any] = {
            "device": device,
            "random_state": self.seed,
            "n_estimators": self.n_estimators,
            # Our datasets are far inside TabPFN v2's pretraining envelope
            # (<= ~2k rows, <= 70 features); this only silences the CPU guard.
            "ignore_pretraining_limits": True,
        }
        if self.task == "classification":
            return TabPFNClassifier(**kwargs)
        return TabPFNRegressor(**kwargs)

    def fit(self, X: Any, y: np.ndarray) -> TabPFNWrapper:
        import torch

        torch.manual_seed(self.seed)
        self.device_used_ = self._resolve_device()
        self.oom_retries_ = 0
        self._X_fit = np.asarray(X, dtype=np.float64)
        self._y_fit = np.asarray(y)
        self.model_ = self._build(self.device_used_)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model_.fit(self._X_fit, self._y_fit)
        if self.task == "classification":
            self.classes_ = np.asarray(self.model_.classes_)
        return self

    def _call_with_oom_guard(self, fn, X: np.ndarray) -> np.ndarray:
        """Halve the inference batch on CUDA OOM; fall back to CPU after retries."""
        import torch

        batch = X.shape[0]
        for attempt in range(self.max_oom_retries + 1):
            try:
                if batch >= X.shape[0]:
                    return fn(X)
                chunks = [fn(X[i : i + batch]) for i in range(0, X.shape[0], batch)]
                return np.concatenate(chunks, axis=0)
            except torch.cuda.OutOfMemoryError:
                self.oom_retries_ += 1
                reset_gpu()
                if attempt == self.max_oom_retries:
                    break
                batch = max(1, batch // 2)
        # Repeated OOM: rebuild on CPU and record the downgrade truthfully.
        reset_gpu()
        self.device_used_ = "cpu (fell back after CUDA OOM)"
        self.model_ = self._build("cpu")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model_.fit(self._X_fit, self._y_fit)
        return fn(X)

    def predict_proba(self, X: Any) -> np.ndarray:
        Xa = np.asarray(X, dtype=np.float64)

        def _inner(chunk: np.ndarray) -> np.ndarray:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return np.asarray(self.model_.predict_proba(chunk), dtype=np.float64)

        return self._call_with_oom_guard(_inner, Xa)

    def predict(self, X: Any) -> np.ndarray:
        Xa = np.asarray(X, dtype=np.float64)

        def _inner(chunk: np.ndarray) -> np.ndarray:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return np.asarray(self.model_.predict(chunk), dtype=np.float64)

        return self._call_with_oom_guard(_inner, Xa)

    @property
    def device_used(self) -> str:
        return getattr(self, "device_used_", "unfitted")


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #


def _xgb_classifier(seed: int, tuned: bool):
    try:
        from xgboost import XGBClassifier

        params: dict[str, Any] = {
            "random_state": seed,
            "n_jobs": 1,
            "tree_method": "hist",
            "verbosity": 0,
            "eval_metric": "mlogloss",
        }
        if tuned:
            params.update({"n_estimators": 300, "max_depth": 4, "learning_rate": 0.1})
        return XGBClassifier(**params), "xgboost"
    except ImportError:
        # Documented fallback (spec section 5): HistGB has the same cost profile.
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(random_state=seed), "histgb"


def make_model(
    name: str,
    seed: int,
    task: str = "classification",
    n_classes: int | None = None,
    device: str | None = None,
) -> Any:
    """Build a named estimator.

    Classification models are returned already wrapped in
    :class:`GlobalClassAligned` when ``n_classes`` is given, so callers can rely
    on globally-aligned ``predict_proba`` columns unconditionally.
    """
    name = name.lower()
    if task == "classification":
        if name == "tabpfn":
            est: Any = TabPFNWrapper(seed=seed, task="classification", device=device)
        elif name in ("xgboost", "xgboost_tuned"):
            est, _ = _xgb_classifier(seed, tuned=name.endswith("_tuned"))
        elif name == "rf":
            est = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=1)
        elif name == "logreg":
            est = Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=2000, random_state=seed)),
                ]
            )
        else:
            raise ValueError(f"unknown classifier {name!r}; expected {CLASSIFIER_NAMES}")
        if n_classes is None:
            return est
        return GlobalClassAligned(est, n_classes=n_classes)

    if task == "regression":
        if name == "tabpfn":
            return TabPFNWrapper(seed=seed, task="regression", device=device)
        if name == "xgboost":
            try:
                from xgboost import XGBRegressor

                return XGBRegressor(random_state=seed, n_jobs=1, tree_method="hist", verbosity=0)
            except ImportError:
                from sklearn.ensemble import HistGradientBoostingRegressor

                return HistGradientBoostingRegressor(random_state=seed)
        if name == "rf":
            return RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=1)
        if name == "ridge":
            return Pipeline([("scale", StandardScaler()), ("reg", Ridge())])
        raise ValueError(f"unknown regressor {name!r}; expected {REGRESSOR_NAMES}")

    raise ValueError(f"unknown task {task!r}")


def warmup(
    name: str, seed: int = 0, task: str = "classification", device: str | None = None
) -> None:
    """Force any lazy one-off initialisation (weight load, CUDA context, JIT).

    Called once per (model, device) *before* any timed region, so that the
    ``fit_time ~ 0`` claim for TabPFN reflects steady-state cost rather than a
    one-time model load charged to whichever cell happened to run first.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(24, 4))
    if task == "classification":
        y = np.array([0, 1] * 12)
        model = make_model(name, seed, task="classification", n_classes=2, device=device)
        model.fit(X, y)
        model.predict_proba(X[:8])
    else:
        model = make_model(name, seed, task="regression", device=device)
        model.fit(X, rng.normal(size=24))
        model.predict(X[:8])
    reset_gpu()
