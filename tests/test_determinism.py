"""Gates 6 and 8: bit-identical reruns, and globally-aligned probability columns.

These need the model zoo, so they are the only tests that touch ``models.py``.
TabPFN cases are skipped (not silently passed) when its weights are absent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conformal_tabpfn.data import Dataset
from conformal_tabpfn.models import GlobalClassAligned, make_model, tabpfn_available
from conformal_tabpfn.runners import run_classification_cell, stratified_subsample
from conformal_tabpfn.utils import cell_hash

CHEAP_MODELS = ["rf", "logreg", "xgboost"]
ALL_MODELS = [*CHEAP_MODELS, "tabpfn"]


def _toy_dataset(n: int = 160, n_classes: int = 3, seed: int = 0) -> Dataset:
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(n_classes), n // n_classes)
    means = rng.normal(scale=2.0, size=(n_classes, 4))
    X = pd.DataFrame(means[y] + rng.standard_normal((y.size, 4)), columns=list("abcd"))
    X["cat"] = pd.Series(rng.choice(["p", "q", "r"], size=y.size))
    return Dataset(
        key="toy",
        X=X,
        y=y.astype(np.int64),
        categorical_columns=["cat"],
        numeric_columns=list("abcd"),
        class_names=[str(i) for i in range(n_classes)],
        meta={},
    )


# --------------------------------------------------------------------------- #
# Gate 8: class alignment when a fold is missing a class
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("model_name", ALL_MODELS)
def test_missing_class_still_yields_aligned_columns(model_name: str) -> None:
    """A fold with no examples of class 1 must still produce 3 aligned columns.

    Getting this wrong shifts every label by one and silently corrupts every
    number in the paper, so it is checked for every model in the zoo.
    """
    if model_name == "tabpfn" and not tabpfn_available():
        pytest.skip("tabpfn not installed")
    if model_name == "xgboost":
        pytest.importorskip("xgboost")

    rng = np.random.default_rng(3)
    n_classes = 3
    # Training labels deliberately omit class 1 entirely.
    y_fit = np.array([0] * 30 + [2] * 30)
    X_fit = rng.normal(loc=y_fit[:, None], size=(60, 4))
    X_eval = rng.normal(size=(15, 4))

    base = make_model(model_name, seed=0, task="classification", n_classes=None)
    model = GlobalClassAligned(base, n_classes=n_classes)
    model.fit(X_fit, y_fit)
    proba = model.predict_proba(X_eval)

    assert proba.shape == (15, n_classes)
    # The unseen class occupies its own global column and is exactly zero there.
    assert np.allclose(proba[:, 1], 0.0)
    # The seen classes carry all the mass.  (An individual seen class may still
    # get probability 0 on a given row -- a random forest does that routinely --
    # so we assert on the mass, not on strict positivity.)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-8)
    assert proba[:, 0].max() > 0 and proba[:, 2].max() > 0
    # And the surviving columns must match the raw estimator's, in global order.
    # The inner estimator is fitted on contiguous LOCAL labels (0..m-1), because
    # XGBoost refuses a gapped label set; `global_of_column_` is the map from its
    # column order back to global label indices, and that map is what must be
    # right for the paper's numbers to mean anything.
    raw = model.estimator_.predict_proba(X_eval)
    seen = np.asarray(model.global_of_column_, dtype=int)
    assert np.allclose(proba[:, seen], raw)
    assert list(seen) == [0, 2]
    assert list(model.seen_classes_) == [0, 2]


@pytest.mark.parametrize("model_name", CHEAP_MODELS)
def test_alignment_is_identity_when_no_class_is_missing(model_name: str) -> None:
    if model_name == "xgboost":
        pytest.importorskip("xgboost")
    rng = np.random.default_rng(4)
    y = np.array([0, 1, 2] * 30)
    X = rng.normal(loc=y[:, None], size=(90, 3))
    model = GlobalClassAligned(
        make_model(model_name, 0, "classification", n_classes=None), n_classes=3
    )
    model.fit(X, y)
    assert np.allclose(model.predict_proba(X[:10]), model.estimator_.predict_proba(X[:10]))


def test_alignment_rejects_out_of_range_labels() -> None:
    rng = np.random.default_rng(5)
    y = np.array([0, 5] * 20)  # label 5 with n_classes=2 is a caller bug
    X = rng.normal(size=(40, 3))
    model = GlobalClassAligned(make_model("logreg", 0, "classification"), n_classes=2)
    with pytest.raises(ValueError, match="labels must be encoded"):
        model.fit(X, y)


# --------------------------------------------------------------------------- #
# Gate 6: determinism
# --------------------------------------------------------------------------- #

_METRIC_KEYS = [
    "coverage", "avg_set_size", "median_set_size", "frac_singleton", "frac_empty",
    "cc_coverage_gap", "sscv", "accuracy", "log_loss", "ece", "qhat",
]


@pytest.mark.parametrize("model_name", CHEAP_MODELS)
@pytest.mark.parametrize("cp_method", ["split_0.3", "cvplus_K5", "mondrian_cvplus_K5"])
def test_same_seed_gives_bit_identical_metrics(model_name: str, cp_method: str) -> None:
    if model_name == "xgboost":
        pytest.importorskip("xgboost")
    ds = _toy_dataset()
    kwargs = dict(
        dataset=ds, model_name=model_name, cp_method=cp_method, seed=7,
        alphas=[0.1, 0.2], scores=["lac", "aps", "raps"], test_frac=0.3, test_cap=100,
    )
    first = run_classification_cell(**kwargs)
    second = run_classification_cell(**kwargs)

    assert len(first) == len(second) == 6  # 3 scores x 2 alphas
    for row_a, row_b in zip(first, second):
        assert row_a["score"] == row_b["score"] and row_a["alpha"] == row_b["alpha"]
        for key in _METRIC_KEYS:
            a, b = row_a[key], row_b[key]
            if isinstance(a, float) and np.isnan(a):
                assert np.isnan(b), key
            else:
                assert a == b, f"{key} differs: {a} != {b}"
        assert row_a["per_class_coverage"] == row_b["per_class_coverage"]


@pytest.mark.parametrize("model_name", CHEAP_MODELS)
def test_different_seeds_actually_differ(model_name: str) -> None:
    """Guard against 'deterministic' passing because the seed is ignored."""
    if model_name == "xgboost":
        pytest.importorskip("xgboost")
    ds = _toy_dataset()
    common = dict(
        dataset=ds, model_name=model_name, cp_method="split_0.3",
        alphas=[0.1], scores=["lac"], test_frac=0.3, test_cap=100,
    )
    a = run_classification_cell(seed=1, **common)[0]
    b = run_classification_cell(seed=2, **common)[0]
    assert (a["coverage"], a["avg_set_size"]) != (b["coverage"], b["avg_set_size"])


def test_cell_hash_is_stable_and_key_sensitive() -> None:
    base = {"dataset": "d", "model": "m", "cp_method": "split_0.3", "alpha": 0.1,
            "seed": 0, "n_train": 100, "imbalance": "natural", "study": "main"}
    assert cell_hash(base) == cell_hash(dict(reversed(list(base.items()))))
    assert cell_hash(base) != cell_hash({**base, "seed": 1})
    assert len(cell_hash(base)) == 16


def test_stratified_subsample_preserves_every_class() -> None:
    rng = np.random.default_rng(0)
    y = np.concatenate([np.zeros(200), np.ones(50), np.full(8, 2)]).astype(int)
    for target in (20, 60, 150):
        idx = stratified_subsample(y, target, seed=0)
        assert idx.size == target
        assert set(np.unique(y[idx])) == {0, 1, 2}, "a class was dropped entirely"
        assert np.array_equal(idx, stratified_subsample(y, target, seed=0))
    # Asking for more than we have returns everything, not an error.
    assert stratified_subsample(y, 10_000, seed=0).size == y.size
    _ = rng
