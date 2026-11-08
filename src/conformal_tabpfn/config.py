"""Experiment configuration: cell keys, CP-method parsing, and YAML loading."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "CONFIG_DIR",
    "CPMethod",
    "Cell",
    "ExperimentConfig",
    "load_experiment_config",
    "parse_cp_method",
]

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"
RESULTS_RAW = ROOT / "results" / "raw"
RESULTS_FAILURES = ROOT / "results" / "failures"


@dataclass(frozen=True)
class CPMethod:
    """A parsed conformal-prediction method specification.

    ``family`` is one of ``split``, ``mondrian_split``, ``cvplus``,
    ``mondrian_cvplus``, ``pooled_cv``, ``jackknife_plus``.
    """

    name: str
    family: str
    cal_frac: float | None = None
    n_folds: int | None = None

    @property
    def needs_folds(self) -> bool:
        return self.family in ("cvplus", "mondrian_cvplus", "pooled_cv", "jackknife_plus")

    @property
    def is_mondrian(self) -> bool:
        return self.family.startswith("mondrian")


def parse_cp_method(name: str) -> CPMethod:
    """Parse strings like ``split_0.3``, ``cvplus_K10``, ``mondrian_cvplus_K10``."""
    raw = name.strip()
    if raw == "jackknife_plus":
        return CPMethod(raw, "jackknife_plus")
    if raw.startswith("mondrian_split_"):
        return CPMethod(raw, "mondrian_split", cal_frac=float(raw.rsplit("_", 1)[1]))
    if raw.startswith("split_"):
        return CPMethod(raw, "split", cal_frac=float(raw.rsplit("_", 1)[1]))
    for prefix, family in (
        ("mondrian_cvplus_K", "mondrian_cvplus"),
        ("cvplus_K", "cvplus"),
        ("pooled_cv_K", "pooled_cv"),
    ):
        if raw.startswith(prefix):
            return CPMethod(raw, family, n_folds=int(raw[len(prefix) :]))
    raise ValueError(
        f"cannot parse cp_method {name!r}; expected e.g. split_0.3, cvplus_K10, "
        "mondrian_cvplus_K10, pooled_cv_K10, jackknife_plus"
    )


@dataclass(frozen=True)
class Cell:
    """One experiment cell.  The key fields fully determine the result."""

    dataset: str
    model: str
    cp_method: str
    alpha: float
    seed: int
    n_train: int
    imbalance: str = "natural"
    study: str = "main"
    minority_frac: float | None = None

    def key(self) -> dict[str, Any]:
        """The dict that gets hashed for the result filename.

        Deliberately excludes the score name: one fitted cell produces the score
        matrices for lac/aps/raps at once, so all three (and all alphas) share a
        cell and are emitted as separate rows of the same parquet file.
        """
        return asdict(self)


@dataclass
class ExperimentConfig:
    datasets: list[str]
    models: list[str]
    cp_methods: list[str]
    scores: list[str]
    alphas: list[float]
    seeds: list[int]
    test_frac: float = 0.3
    test_cap: int = 1000
    n_train: int | None = None
    jackknife_max_n: int = 500
    chunk_size: int = 256
    n_jobs: int = 1
    raps_lambda: float = 0.01
    raps_k_reg: int = 1
    quantile_method: str = "tight"
    budget_cal_fracs: list[float] | None = None
    budget_models: list[str] | None = None
    crossover_n_trains: list[int] | None = None
    crossover_datasets: list[str] | None = None
    crossover_methods: list[str] | None = None
    imbalance_datasets: list[str] | None = None
    imbalance_ratios: list[float] | None = None
    raw_dir: Path = RESULTS_RAW
    failures_dir: Path = RESULTS_FAILURES


def load_experiment_config(path: str | Path | None = None) -> ExperimentConfig:
    p = Path(path) if path else CONFIG_DIR / "experiment.yaml"
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    return ExperimentConfig(
        datasets=list(cfg["datasets"]),
        models=list(cfg["models"]),
        cp_methods=list(cfg["cp_methods"]),
        scores=list(cfg["scores"]),
        alphas=[float(a) for a in cfg["alphas"]],
        seeds=list(range(int(cfg["n_seeds"]))) if "n_seeds" in cfg else list(cfg["seeds"]),
        test_frac=float(cfg.get("test_frac", 0.3)),
        test_cap=int(cfg.get("test_cap", 1000)),
        n_train=cfg.get("n_train"),
        jackknife_max_n=int(cfg.get("jackknife_max_n", 500)),
        chunk_size=int(cfg.get("chunk_size", 256)),
        n_jobs=int(cfg.get("n_jobs", 1)),
        raps_lambda=float(cfg.get("raps_lambda", 0.01)),
        raps_k_reg=int(cfg.get("raps_k_reg", 1)),
        quantile_method=str(cfg.get("quantile_method", "tight")),
        budget_cal_fracs=cfg.get("budget", {}).get("cal_fracs"),
        budget_models=cfg.get("budget", {}).get("models"),
        crossover_n_trains=cfg.get("crossover", {}).get("n_trains"),
        crossover_datasets=cfg.get("crossover", {}).get("datasets"),
        crossover_methods=cfg.get("crossover", {}).get("methods"),
        imbalance_datasets=cfg.get("imbalance", {}).get("datasets"),
        imbalance_ratios=cfg.get("imbalance", {}).get("minority_fracs"),
    )
