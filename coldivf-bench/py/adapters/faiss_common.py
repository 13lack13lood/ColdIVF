"""What the faiss adapters share. That is the library's setup, the metric, training samples, shared partitions, the
meta file of an index, file ranges, read-ahead and residency checks.

faiss is imported inside the functions that use it, never at the top of a module, so the thread environment is set
before the library starts its OpenMP (Open Multi-Processing) pool. The helpers that don't need faiss can be tested
without it.
"""

from __future__ import annotations

import fcntl
import json
import os
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from adapters import faiss_layout
from adapters.base import Adapter, ServeContext, Unsupported
import data
import evictor
import schema

META = "meta.json"
TRAIN_ROWS = 500_000
CHUNK_ROWS = 100_000
# A vector counts as unit length when its norm is within this of 1.
UNIT_TOL = 1e-3
UNIT_SAMPLE = 1000

# macOS's fcntl command that asks the kernel to read a byte range into the cache and returns at once. Python's fcntl
# module doesn't name it.
F_RDADVISE = 44
# The most one F_RDADVISE asks for, since its count is a 32-bit int. It is a whole number of 16 KiB pages.
ADVISE_MAX = (1 << 31) - (1 << 14)


def set_threads(threads: int) -> None:
    """Sets faiss's OpenMP threads, on top of OMP_NUM_THREADS."""
    import faiss

    faiss.omp_set_num_threads(max(1, int(threads)))


def simd_level() -> str:
    """The SIMD (single instruction, multiple data) level faiss dispatches to, eg. "ARM_NEON"."""
    import faiss

    try:
        return str(faiss.SIMDConfig.get_level_name())
    except AttributeError:
        return compiled_simd(faiss.get_compile_options())


def compiled_simd(options: str) -> str:
    """The best SIMD level named in faiss's compile options, eg. "OPTIMIZE DD ARM_NEON" gives "ARM_NEON"."""
    words = set(options.split())

    for level in ("AVX512_SPR", "AVX512", "AVX2", "ARM_SVE", "ARM_NEON"):
        if level in words:
            return level

    return "NONE"


def faiss_metric(metric: str) -> str:
    """The faiss metric for a dataset's metric. Cosine runs as L2 (squared Euclidean distance) on unit vectors, which
    ranks rows the same, and routes to k-means centroids by the distance k-means minimized. It is "l2" or "ip"."""
    if metric in ("cosine", "l2"):
        return "l2"

    if metric == "ip":
        return "ip"

    raise Unsupported(f"faiss has no metric for {metric!r}")


def metric_type(name: str) -> int:
    import faiss

    return faiss.METRIC_L2 if name == "l2" else faiss.METRIC_INNER_PRODUCT


def check_unit(rows: np.ndarray, metric: str, sample: int = UNIT_SAMPLE) -> None:
    """Checks that a cosine dataset holds unit vectors, on rows spread over the whole set, since L2 ranks only those
    the way cosine does."""
    if metric != "cosine" or len(rows) == 0:
        return

    picks = np.unique(np.linspace(0, len(rows) - 1, min(sample, len(rows))).astype(np.int64))
    norms = np.linalg.norm(np.asarray(rows[picks], dtype=np.float64), axis=1)
    worst = float(np.max(np.abs(norms - 1.0)))

    if worst > UNIT_TOL:
        raise ValueError(f"cosine runs as L2 on unit vectors, but a row's norm is off from 1 by {worst:.4f}")


def sample_rows(rows: np.ndarray, count: int, seed: int) -> np.ndarray:
    """`count` rows picked at random with `seed`, in row order, as one contiguous float32 array. With no more rows than
    `count` it is all of them, without a copy."""
    if len(rows) <= count:
        return np.ascontiguousarray(rows, dtype=np.float32)

    picks = np.sort(np.random.default_rng(seed).choice(len(rows), size=count, replace=False))
    return np.ascontiguousarray(rows[picks], dtype=np.float32)


def chunks(total: int, size: int) -> Iterator[tuple[int, int]]:
    """The row ranges [lo, hi) of `total` rows in steps of `size`."""
    for lo in range(0, total, max(1, size)):
        yield lo, min(total, lo + size)


@dataclass
class Partition:
    """A shared partition of the first rows of a base, its centroids and each row's cluster."""

    centroids: np.ndarray
    assign: np.ndarray

    @property
    def clusters(self) -> int:
        return len(self.centroids)


def partition(stem: Path | str, rows: int, dim: int) -> Partition:
    """The partition at `stem`, i.e. <stem>.centroids.fbin and <stem>.assign.ibin, for the first `rows` base rows."""
    centroids, assign = data.clusters(stem)

    if centroids.shape[1] != dim:
        raise ValueError(f"the clusters at {stem} have {centroids.shape[1]} dimensions, not {dim}")

    if len(assign) < rows:
        raise ValueError(f"the clusters at {stem} assign {len(assign)} rows, fewer than {rows}")

    return Partition(np.ascontiguousarray(centroids, dtype=np.float32), np.asarray(assign[:rows], dtype=np.int64))


def write_meta(folder: Path, record: dict) -> None:
    path = Path(folder) / META
    path.write_text(json.dumps(schema.clean(record), indent=1, sort_keys=True) + "\n")


def read_meta(folder: Path | str) -> dict:
    """The meta file a build wrote beside its index files."""
    path = Path(folder) / META

    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing, so {folder} isn't a faiss index this harness built")

    return json.loads(path.read_text())


def parts_record(folder: Path, names: list[str], loaded: bool = False) -> list[dict]:
    """Every part of every index file named, as {file, name, offset, length, kind}. On-disk lists files are lists
    from end to end. A file that loads at open whole is one part named "file"."""
    out = []

    for name in names:
        path = Path(folder) / name
        size = path.stat().st_size

        if loaded or path.suffix == ".ivfdata":
            part = "file" if loaded else "lists"
            out.append({"file": name, "name": part, "offset": 0, "length": size, "kind": "bulk"})
            continue

        for p in faiss_layout.read(path).parts:
            out.append({"file": name, "name": p.name, "offset": p.offset, "length": p.length, "kind": p.kind})

    return out


def whole(path: Path, kind: str, loaded: bool = False) -> dict:
    """A file entry for files() that is one range of `kind`, marked "loaded" when an open copies the whole file into
    memory."""
    size = path.stat().st_size
    span = {"offset": 0, "length": size, "kind": kind} | ({"loaded": True} if loaded else {})
    return {"path": str(path), "bytes": size, "ranges": [span] if size else []}


def parsed(path: Path, loaded: frozenset[str] = frozenset()) -> dict:
    """A file entry for files() with the ranges of the faiss index file at `path`. The parts named in `loaded`, which
    the open copies into memory, are marked "loaded"."""
    layout = faiss_layout.read(path)
    return {"path": str(path), "bytes": layout.size, "ranges": layout.ranges(loaded)}


def merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Byte ranges (offset, length) in order, with overlapping and touching ranges joined."""
    out: list[list[int]] = []

    for offset, length in sorted((int(o), int(n)) for o, n in spans if n > 0):
        if out and offset <= out[-1][0] + out[-1][1]:
            out[-1][1] = max(out[-1][1], offset + length - out[-1][0])
        else:
            out.append([offset, length])

    return [(o, n) for o, n in out]


def read_ahead(fd: int, offset: int, length: int) -> None:
    """Asks the kernel to read bytes offset to offset + length into the cache, and returns before they arrive. It uses
    F_RDADVISE on macOS and posix_fadvise elsewhere."""
    if sys.platform == "darwin":
        while length > 0:
            n = min(length, ADVISE_MAX)
            fcntl.fcntl(fd, F_RDADVISE, struct.pack("qi4x", offset, n))
            offset, length = offset + n, length - n
    elif hasattr(os, "posix_fadvise") and length > 0:
        os.posix_fadvise(fd, offset, length, os.POSIX_FADV_WILLNEED)


def resident_parts(folder: Path, parts: list[dict]) -> dict[str, int]:
    """How many bytes of each named part are cached now, eg. {"coarse": 19202048, "lists": 3407872}, counted in whole
    pages. It maps each file anew and reads nothing. A new mapping can make another process read the file's first page
    back soon after, so use it only once the timed search is done."""
    out: dict[str, int] = {}
    maps: dict[str, evictor.HeldMap] = {}

    try:
        for p in parts:
            path = str(Path(folder) / p["file"])

            if path not in maps:
                maps[path] = evictor.HeldMap(path)

            pages, _ = maps[path].resident(int(p["offset"]), int(p["offset"]) + int(p["length"]))
            out[p["name"]] = out.get(p["name"], 0) + pages * evictor.PAGE
    finally:
        for held in maps.values():
            held.close()

    return out


class FaissAdapter(Adapter):
    """The parts every faiss adapter shares. A subclass names its files and builds and opens its indexes."""

    binding = "python"
    # faiss's search takes many queries in one call, runs them on its OpenMP threads and releases Python's lock.
    batch_api = True
    workers = "OpenMP's pool, for batch searches"

    def __init__(self) -> None:
        super().__init__()
        self._meta: dict[str, dict] = {}
        # Set while a cold trial's first query runs, so stats() also reports which parts it read.
        self._first_query = False
        self._serving: Path | None = None

    def load(self) -> None:
        set_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))

    def version(self) -> str:
        import faiss

        return faiss.__version__

    def wheels(self) -> dict[str, str | None]:
        return {"faiss-cpu": schema.dist_sha256("faiss-cpu")}

    def simd_level(self) -> str:
        return simd_level()

    def meta(self, path: Path | str) -> dict:
        """The meta file of the index at `path`, read once per process."""
        key = str(Path(path))

        if key not in self._meta:
            self._meta[key] = read_meta(path)

        return self._meta[key]

    def cold_setup(self, ctx: ServeContext) -> None:
        # The meta file is read before READY, so the timed open reads only the index files.
        self.meta(ctx.index)
        self._serving = Path(ctx.index)

        if ctx.regime != "c1":
            super().cold_setup(ctx)
            return

        ctx.handle = self.open(ctx, ctx.index)
        self.hold_routing(ctx.handle)
        self.search(ctx.handle, self.warmup_query(ctx), ctx.k, ctx.params)

    def hold_routing(self, handle: Any) -> None:
        """Brings the routing data that C1 keeps for every query into memory, eg. a mapped coarse quantizer. By default
        the routing ranges just stay cached, since the driver drops only bulk ranges."""

    def cold_trial(self, ctx: ServeContext, go: dict) -> dict:
        self._first_query = True

        try:
            return super().cold_trial(ctx, go)
        finally:
            self._first_query = False

    def first_query_parts(self) -> dict:
        """After a cold trial's first query, the cached bytes of each part of the index, eg. how much of the coarse
        table it faulted in."""
        if not self._first_query or self._serving is None:
            return {}

        self._first_query = False
        parts = self.meta(self._serving).get("parts") or []
        return {"resident_after_first": resident_parts(self._serving, parts)} if parts else {}
