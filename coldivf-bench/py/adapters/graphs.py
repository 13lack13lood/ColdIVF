"""Helpers the graph adapters share, for usearch, qdrant-edge and AsterVec, and the whole-file durable point of usearch
and RaBitQ-Library."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

import numpy as np

from adapters.base import Unsupported
import writes

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


def save_whole(file: Path | str, save: Callable[[str], Any]) -> dict:
    """The durable point of a library that only saves a whole file, which the harness makes survive a power loss.
    `save` writes the index to a temporary file beside `file`. That file gets F_FULLFSYNC and a rename over `file`,
    and then the folder gets F_FULLFSYNC, so the rename survives too. A file the library writes beside the temporary
    one, eg. its name with ".hnsw" added, takes the same steps to the same name beside `file`. Nothing ever saves over
    the old file, so a process that maps it keeps the old bytes. It returns the files and bytes it wrote."""
    file = Path(file)
    tmp = file.with_name(f"{file.name}.save-{os.getpid()}.tmp")
    save(str(tmp))
    written = sorted(p for p in file.parent.iterdir() if p.name.startswith(tmp.name))

    if tmp not in written:
        raise FileNotFoundError(f"the library's save left no file at {tmp}")

    names, size = [], 0

    for part in written:
        target = file.with_name(file.name + part.name[len(tmp.name):])
        size += part.stat().st_size
        writes.fullfsync(part)
        os.replace(part, target)
        names.append(target.name)

    writes.fullfsync(file.parent)
    return {"call": "save", "files": names, "bytes": size, "sync": "F_FULLFSYNC", "rename": True}
