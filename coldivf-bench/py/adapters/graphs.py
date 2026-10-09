"""Helpers the graph adapters share, for usearch, qdrant-edge and AsterVec."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np

from adapters.base import Unsupported

META = "meta.json"

# A row whose length is this close to 1 counts as unit length.
UNIT_TOL = 1e-3


def pad(ids: Sequence[int] | np.ndarray, k: int) -> np.ndarray:
    """The first `k` ids as int64, padded with -1."""
    out = np.full(k, -1, dtype=np.int64)
    found = np.asarray(ids).ravel()[:k].astype(np.int64)
    out[: len(found)] = found
    return out


def check_keys(settings: dict, allowed: Iterable[str], what: str, system: str) -> None:
    """Raises Unsupported for a setting the adapter doesn't know, so a typo never runs the default."""
    unknown = sorted(set(settings) - set(allowed))

    if unknown:
        raise Unsupported(f"{system} takes no {what} {unknown}")


def is_unit(rows: np.ndarray, sample: int = 1024) -> bool:
    """Whether up to `sample` evenly spaced rows all have unit length, so the inner product ranks them as cosine
    does."""
    if len(rows) == 0:
        return True

    picks = np.unique(np.linspace(0, len(rows) - 1, min(sample, len(rows))).astype(np.int64))
    norms = np.linalg.norm(np.asarray(rows[picks], dtype=np.float64), axis=1)
    return bool(np.all(np.abs(norms - 1.0) <= UNIT_TOL))


def batches(n: int, size: int) -> Iterator[tuple[int, int]]:
    """The row ranges [start, end) that split n rows into batches of at most `size`."""
    for start in range(0, n, size):
        yield start, min(n, start + size)


def write_meta(folder: Path, meta: dict) -> None:
    (Path(folder) / META).write_text(json.dumps(meta, indent=1, sort_keys=True))


def read_meta(folder: Path) -> dict:
    return json.loads((Path(folder) / META).read_text())
