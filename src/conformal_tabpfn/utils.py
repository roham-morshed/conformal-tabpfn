"""Seeding, hashing, timing and environment capture.

Nothing here touches the network.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata as importlib_metadata
import json
import os
import platform
import random
import time
import traceback
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "Timer",
    "atomic_write_parquet",
    "cell_hash",
    "environment_info",
    "package_versions",
    "set_all_seeds",
    "stable_json",
    "timed",
    "write_failure",
]

_TRACKED_PACKAGES = (
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "torch",
    "tabpfn",
    "xgboost",
    "pyarrow",
    "matplotlib",
)


def set_all_seeds(seed: int) -> None:
    """Seed every RNG we can reach, including torch if it is importable."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # torch is optional for the pure-math tests
        pass


def stable_json(obj: Any) -> str:
    """Deterministic JSON encoding used for hashing cell keys."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def cell_hash(key: Mapping[str, Any]) -> str:
    """Deterministic 16-hex-char id for an experiment cell key."""
    return hashlib.sha256(stable_json(dict(key)).encode("utf-8")).hexdigest()[:16]


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def package_versions() -> dict[str, str]:
    """Installed versions of the packages whose behaviour can move our numbers."""
    out: dict[str, str] = {"python": platform.python_version()}
    for name in _TRACKED_PACKAGES:
        try:
            out[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            out[name] = "not-installed"
    return out


def environment_info() -> dict[str, Any]:
    """Machine + GPU description recorded into paper/reproducibility.json."""
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "packages": package_versions(),
        "gpu_name": None,
        "cuda_available": False,
        "cuda_version": None,
    }
    try:
        import torch

        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["gpu_total_memory_gb"] = round(props.total_memory / 1024**3, 2)
    except ImportError:
        pass
    return info


class Timer:
    """Wall-clock timer using ``perf_counter``; also syncs CUDA when relevant.

    CUDA kernels are asynchronous, so timing GPU work without a synchronise call
    measures queueing, not compute.  Every timing that feeds RQ1 goes through
    this class.
    """

    def __init__(self, sync_cuda: bool = True) -> None:
        self.sync_cuda = sync_cuda
        self.elapsed: float = float("nan")

    def _sync(self) -> None:
        if not self.sync_cuda:
            return
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except ImportError:
            pass

    def __enter__(self) -> Timer:
        self._sync()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self._sync()
        self.elapsed = time.perf_counter() - self._t0


@contextlib.contextmanager
def timed(store: dict[str, float], key: str, sync_cuda: bool = True) -> Iterator[None]:
    """Context manager writing the elapsed seconds into ``store[key]``."""
    with Timer(sync_cuda=sync_cuda) as t:
        yield
    store[key] = t.elapsed


def write_failure(
    failures_dir: str | Path, cell_key: Mapping[str, Any], exc: BaseException
) -> Path:
    """Record a failed cell: never swallow an exception silently."""
    failures = Path(failures_dir)
    failures.mkdir(parents=True, exist_ok=True)
    path = failures / f"{cell_hash(cell_key)}.json"
    payload = {
        "cell": dict(cell_key),
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def atomic_write_parquet(df: Any, path: str | Path) -> Path:
    """Write a parquet file via a temp file + rename, so a crash cannot leave a
    half-written result that the resume logic would then treat as complete."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, target)
    return target
