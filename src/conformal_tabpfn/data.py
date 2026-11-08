"""Dataset loading from the local cache, preprocessing, and imbalance simulation.

**This module never touches the network.**  Everything it reads was written by
``scripts/download_assets.py``, which is the single networked entry point in the
repository.  If the cache is missing, we raise an actionable error rather than
silently reaching out.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

__all__ = [
    "CACHE_DIR",
    "Dataset",
    "DatasetMissingError",
    "available_datasets",
    "build_preprocessor",
    "load_dataset",
    "make_imbalanced",
]

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"

# Sentinel index for unseen / missing categorical levels.  Ordinal encoding maps
# every unknown level here, so a fold that never saw a category cannot crash and
# cannot silently collide with a real level.
UNKNOWN_CATEGORY = -1


class DatasetMissingError(FileNotFoundError):
    """Raised when a dataset is not in the local cache."""


@dataclass
class Dataset:
    """A cached tabular dataset plus everything needed to describe it in the paper."""

    key: str
    X: pd.DataFrame
    y: np.ndarray  # label-encoded 0..C-1
    categorical_columns: list[str]
    numeric_columns: list[str]
    class_names: list[str]
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.X.shape[0])

    @property
    def d(self) -> int:
        return int(self.X.shape[1])

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    def class_counts(self) -> dict[int, int]:
        vals, counts = np.unique(self.y, return_counts=True)
        return {int(v): int(c) for v, c in zip(vals, counts)}

    def imbalance_ratio(self) -> float:
        """Majority-to-minority class-count ratio (1.0 = perfectly balanced)."""
        counts = np.array(list(self.class_counts().values()), dtype=np.float64)
        return float(counts.max() / counts.min())

    def describe(self) -> str:
        counts = self.class_counts()
        return (
            f"{self.key:<24} n={self.n:<6} d={self.d:<4} C={self.n_classes:<3} "
            f"imb={self.imbalance_ratio():<5.2f} counts={counts}"
        )


def available_datasets(cache_dir: str | Path = CACHE_DIR) -> list[str]:
    """Keys currently present in the local cache."""
    cache = Path(cache_dir)
    if not cache.exists():
        return []
    return sorted(p.stem for p in cache.glob("*.parquet"))


def load_dataset(key: str, cache_dir: str | Path = CACHE_DIR) -> Dataset:
    """Read one dataset from the local cache.  Never downloads anything."""
    cache = Path(cache_dir)
    parquet = cache / f"{key}.parquet"
    meta_path = cache / f"{key}.meta.json"
    if not parquet.exists() or not meta_path.exists():
        raise DatasetMissingError(
            f"Dataset {key!r} is not in the local cache ({cache}).\n"
            "Run `python scripts/download_assets.py` while connected to the network. "
            f"Currently cached: {available_datasets(cache) or '(nothing)'}"
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    frame = pd.read_parquet(parquet)

    target = meta["target_column"]
    y = frame[target].to_numpy()
    X = frame.drop(columns=[target])

    cat_cols = [c for c in meta["categorical_columns"] if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols]
    return Dataset(
        key=key,
        X=X,
        y=y.astype(np.int64),
        categorical_columns=cat_cols,
        numeric_columns=num_cols,
        class_names=[str(c) for c in meta["class_names"]],
        meta=meta,
    )


def build_preprocessor(dataset: Dataset) -> ColumnTransformer:
    """Leakage-free shared preprocessing, refitted inside every fold.

    Per the spec this stage deliberately does **no** standardisation: TabPFN and
    the tree models want raw values, and LogisticRegression carries its own
    ``StandardScaler`` inside its own ``Pipeline`` (see ``models.py``).  Keeping
    scaling model-side is what makes zero leakage structurally guaranteed rather
    than merely intended.

    - categorical -> ordinal codes, unseen levels -> ``UNKNOWN_CATEGORY``
    - missing numeric -> median (fitted on the fold's training part only)
    - missing categorical -> its own ordinal index
    """
    numeric = Pipeline([("impute", SimpleImputer(strategy="median"))])
    categorical = Pipeline(
        [
            (
                "ordinal",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=UNKNOWN_CATEGORY,
                    encoded_missing_value=UNKNOWN_CATEGORY,
                ),
            )
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric, dataset.numeric_columns),
            ("cat", categorical, dataset.categorical_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_imbalanced(
    X: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    ratios: dict[int, float] | list[float],
    seed: int,
) -> tuple[Any, np.ndarray, dict[int, int]]:
    """Subsample classes to target proportions, for the RQ5 fairness study.

    ``ratios`` gives the desired *share of the retained sample* per class.  We
    keep as many points as the scarcest constraint allows, so the realised
    proportions match the request as closely as the data permits.  The realised
    counts are returned and must be recorded in the result row -- the requested
    ratio is not evidence of what actually happened.
    """
    y = np.asarray(y)
    classes = np.unique(y)
    if isinstance(ratios, list):
        if len(ratios) != len(classes):
            raise ValueError(f"expected {len(classes)} ratios, got {len(ratios)}")
        ratios = {int(c): float(r) for c, r in zip(classes, ratios)}
    total_ratio = sum(ratios.values())
    if total_ratio <= 0:
        raise ValueError("ratios must sum to something positive")
    ratios = {int(c): float(r) / total_ratio for c, r in ratios.items()}

    available = {int(c): int((y == c).sum()) for c in classes}
    # Largest total N such that ceil(N * ratio_c) <= available_c for every class.
    n_total = min(int(available[c] // r) for c, r in ratios.items() if r > 0)
    if n_total < len(classes):
        raise ValueError(
            f"requested ratios {ratios} leave only {n_total} points; not enough data"
        )

    rng = np.random.default_rng(seed)
    keep: list[np.ndarray] = []
    realised: dict[int, int] = {}
    for c in sorted(ratios):
        want = round(n_total * ratios[c])
        want = max(1, min(want, available[c]))
        idx = np.flatnonzero(y == c)
        keep.append(rng.choice(idx, size=want, replace=False))
        realised[c] = want
    sel = np.sort(np.concatenate(keep))

    X_out = X.iloc[sel] if isinstance(X, pd.DataFrame) else X[sel]
    return X_out, y[sel], realised
