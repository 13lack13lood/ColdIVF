"""Small synthetic datasets for the tests, with exact ground truth."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

PY = Path(__file__).resolve().parents[1]

if str(PY) not in sys.path:
    sys.path.insert(0, str(PY))

import data  # noqa: E402


def exact(base: np.ndarray, queries: np.ndarray, k: int, metric: str = "cosine") -> tuple[np.ndarray, np.ndarray]:
    """Each query's k nearest rows, nearest first and ties to the lower id, with their distances."""
    ids = np.empty((len(queries), k), dtype=np.int64)
    dists = np.empty((len(queries), k))

    for i, q in enumerate(queries):
        d = data.distances(metric, q, base)
        order = np.argsort(d, kind="stable")[:k]
        ids[i], dists[i] = order, d[order]

    return ids, dists


def make_dataset(root: Path, name: str = "tiny-16-cosine", n: int = 3000, dim: int = 16, nq: int = 40,
                 k: int = 100, seed: int = 0, tie_query: int | None = 3) -> data.Dataset:
    """Writes a dataset of unit vectors. With `tie_query`, that query's 10th nearest row gets a duplicate at the last
    id, so the truth has a tie at rank 10 that the lower id wins."""
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((n, dim)).astype(np.float32)
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    queries = rng.standard_normal((nq, dim)).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    if tie_query is not None:
        ids, _ = exact(base, queries[tie_query : tie_query + 1], 10)
        base[n - 1] = base[ids[0, 9]]

    ids, _ = exact(base, queries, k)
    root.mkdir(parents=True, exist_ok=True)
    data.write_fbin(root / f"{name}.base.fbin", base)
    data.write_fbin(root / f"{name}.query.fbin", queries)
    data.write_ibin(root / f"{name}.gt.ibin", ids.astype(np.int32))
    return data.Dataset(name, root, root / "derived")


def python() -> str:
    return sys.executable


def env(extra: dict | None = None) -> dict:
    out = dict(os.environ)
    out["PYTHONPATH"] = str(PY)
    out["PYTHONDONTWRITEBYTECODE"] = "1"
    out.update(extra or {})
    return out
