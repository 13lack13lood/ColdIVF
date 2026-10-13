"""What the three RaBitQ-Library adapters share.

RaBitQ-Library (pip rabitqlib) builds an index in one call from a float32 array, which it reads without copying. A
saved index loads whole at open, so a cold first query pays for reading all of it. The library ranks by inner product
or by L2 distance, so a cosine dataset runs as inner product on its unit vectors. Its search calls hold Python's global
interpreter lock (GIL), so queries per second (QPS) at several threads go through its batch call, which spreads one
batch over the library's OpenMP threads. Every clustering, build and search call gets its thread count, and the base
class sets OMP_NUM_THREADS to match.

An IVF (inverted file index) and an HNSW (hierarchical navigable small world) graph code every row relative to the
center of its cluster, so their builds take a partition. It is a shared one from --clusters, or the library's own
k-means, which trains on every row and then assigns each row to its nearest center.

The IVF and the HNSW take writes on the loaded index, through `RabitqWrites`. An insert is add, which numbers the rows
itself, the next ids after the last, and codes each row against the center nearest it, which doesn't move. A delete is
remove, which leaves the row's code in place and only keeps it out of later results. The library saves only a whole
file, so the durable point saves the index beside the old file, syncs it and renames it over the old one. SymphonyQG's
index has no add or remove, so it keeps the base's rebuild-only role.
"""

from __future__ import annotations

import importlib.metadata
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from adapters import graphs, inproc
from adapters.base import Adapter, BuildContext, ServeContext, Unsupported, WriteContext
import data
import schema

# The id of a result slot that the library couldn't fill.
MISSING = 2**32 - 1

KMEANS = ("qg", "rabitq")
# QGKMeans links the centers in a graph of degree 32, so it needs more clusters than that.
QG_MIN_CLUSTERS = 33


def library_metric(metric: str) -> str:
    """The library's metric for a dataset's. It has "l2" and "ip", and a cosine dataset runs as "ip"."""
    if metric == "l2":
        return "l2"

    if metric in ("ip", "cosine"):
        return "ip"

    raise Unsupported(f"RaBitQ-Library has no metric {metric}")


def cluster_count(rows: int, nlist: Any = None, vectors_per_list: Any = None) -> int:
    """The cluster count, given as `nlist` or as `vectors_per_list`, the average rows per cluster."""
    if (nlist is None) == (vectors_per_list is None):
        raise ValueError("give the cluster count as nlist or as vectors_per_list, one of the two")

    count = int(nlist) if nlist is not None else int(round(rows / float(vectors_per_list)))

    if not 1 <= count <= rows:
        raise ValueError(f"{count} clusters don't fit {rows} rows")

    return count


def kmeans_method(method: str | None, count: int) -> str:
    """The library's k-means for `count` clusters. It is QGKMeans, the graph method, wherever that can run, and
    RaBitQKMeans, the flat scan, below that."""
    if method is None:
        return "qg" if count >= QG_MIN_CLUSTERS else "rabitq"

    if method not in KMEANS:
        raise ValueError(f"kmeans is {method}, not one of {list(KMEANS)}")

    if method == "qg" and count < QG_MIN_CLUSTERS:
        raise ValueError(f"QGKMeans needs at least {QG_MIN_CLUSTERS} clusters, not {count}")

    return method


@dataclass
class Partition:
    """Each row's cluster and the clusters' centers. `seconds` is the clustering time, or None for a shared partition,
    whose time is charged where it was made."""

    centers: np.ndarray
    assign: np.ndarray
    seconds: float | None = None
    info: dict = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.centers)


def imported(stem: Path, rows: int, dim: int) -> Partition:
    """The shared partition at `stem`, for the first `rows` rows of the base."""
    centers, assign = data.clusters(stem)

    if len(assign) < rows:
        raise ValueError(f"{stem}.assign.ibin covers {len(assign)} rows, not {rows}")

    if centers.shape[1] != dim:
        raise ValueError(f"{stem}.centroids.fbin has {centers.shape[1]} dimensions, not {dim}")

    return Partition(np.ascontiguousarray(centers, dtype=np.float32),
                     np.ascontiguousarray(assign[:rows], dtype=np.uint32), None,
                     {"partition": "imported", "clusters": str(stem)})


def train(lib: Any, rows: np.ndarray, count: int, method: str, niter: int, threads: int, seed: int,
          spherical: bool) -> Partition:
    """The library's own k-means on every row, with exact assignment at the end. Spherical k-means keeps unit
    centers and assigns by inner product, which suits unit vectors."""
    kind = lib.QGKMeans if method == "qg" else lib.RaBitQKMeans
    km = kind(rows.shape[1], count, niter=int(niter), num_threads=int(threads), seed=int(seed), spherical=spherical,
              final_assignment=lib.FinalAssignmentMode.Exact)
    start = time.perf_counter()
    km.train(rows)
    seconds = time.perf_counter() - start
    info = {"partition": "kmeans", "kmeans": method, "niter": int(niter), "spherical": spherical,
            "objective": float(km.final_obj), "iterations": len(km.iteration_stats or [])}
    return Partition(np.ascontiguousarray(km.centroids, dtype=np.float32),
                     np.ascontiguousarray(km.assignments, dtype=np.uint32), seconds, info)


def compact(part: Partition) -> Partition:
    """The partition without the clusters that no row uses, renumbered in their order."""
    used = np.unique(part.assign)

    if len(used) == part.count:
        return part

    remap = np.full(part.count, -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return Partition(np.ascontiguousarray(part.centers[used]), remap[part.assign].astype(np.uint32), part.seconds,
                     part.info | {"dropped_clusters": part.count - len(used)})


def list_sizes(part: Partition) -> dict:
    """The cluster count and the smallest and largest cluster."""
    sizes = np.bincount(part.assign, minlength=part.count)
    return {"clusters": part.count, "list_min": int(sizes.min()), "list_max": int(sizes.max()),
            "empty_lists": int(np.count_nonzero(sizes == 0))}


def code_bytes(rows: int, dim: int, bits: int) -> int:
    """The bytes of the codes that searches score with, without the factors stored per vector. The library pads the
    rotated dimension to a multiple of 32. At 32 bits it keeps float32 rows beside the 1-bit codes it filters with."""
    padded = -(-dim // 32) * 32

    if bits == 32:
        return rows * dim * 4 + rows * padded // 8

    return rows * padded * bits // 8


@dataclass
class Writer:
    """The loaded index of a write session, the file its durable point replaces, and the threads its calls take."""

    index: Any
    file: Path
    threads: int


class RabitqAdapter(Adapter):
    """One RaBitQ-Library index type. A subclass names its file and its library class, builds, and runs a batch of
    queries in `query`."""

    binding = "python"
    batch_api = True
    workers = "OpenMP's pool, for batch searches"
    # Every file of the index loads whole at open.
    loads_at_open = True
    # The index file in the index's folder, and the library class that loads it.
    file_name = ""
    kind = ""

    def load(self) -> None:
        import rabitqlib

        self.lib = rabitqlib

    def version(self) -> str:
        return importlib.metadata.version("rabitqlib")

    def wheels(self) -> dict[str, str | None]:
        return {"rabitqlib": schema.dist_sha256("rabitqlib")}

    def simd_level(self) -> str:
        return inproc.simd("neon", "avx2, or avx512 where the CPU has it")

    def index_file(self, path: Path) -> Path:
        path = Path(path)
        return path / self.file_name if path.is_dir() else path

    def open(self, ctx: ServeContext, path: Path) -> Any:
        return getattr(self.lib, self.kind).load(str(self.index_file(path)))

    def check_search(self, params: dict) -> dict:
        """The search settings with their defaults. An unknown or missing key raises ValueError."""
        raise NotImplementedError

    def query(self, index: Any, queries: np.ndarray, k: int, params: dict, threads: int) -> np.ndarray:
        """The library's ids for a batch of queries, one row each."""
        raise NotImplementedError

    def configure(self, handle: Any, params: dict) -> None:
        self.check_search(params)

    def cold_setup(self, ctx: ServeContext) -> None:
        self.check_search(ctx.params)
        super().cold_setup(ctx)

    def search(self, handle: Any, query: np.ndarray, k: int, params: dict) -> np.ndarray:
        one = np.ascontiguousarray(query, dtype=np.float32).reshape(1, -1)
        return inproc.as_ids(self.query(handle, one, k, params, 1)[0], k, MISSING)

    def search_batch(self, handle: Any, queries: np.ndarray, k: int, params: dict, threads: int) -> np.ndarray:
        found = self.query(handle, np.ascontiguousarray(queries, dtype=np.float32), k, params, threads)
        return np.stack([inproc.as_ids(row, k, MISSING) for row in found])

    def partition(self, ctx: BuildContext, count: int | None, method: str | None, niter: int) -> Partition:
        """The build's partition, the shared one from ctx.clusters or the library's k-means into `count` clusters."""
        if ctx.clusters is not None:
            part = imported(ctx.clusters, ctx.rows, ctx.dim)

            if count is not None and count != part.count:
                raise ValueError(f"the config asks for {count} clusters, and {ctx.clusters} has {part.count}")

            return part

        method = kmeans_method(method, count)
        print(f"clustering {ctx.rows} rows into {count} clusters with {method} k-means", file=sys.stderr, flush=True)
        return train(self.lib, ctx.data, count, method, niter, ctx.threads, ctx.seed, ctx.metric == "cosine")

    def save(self, index: Any, out: Path) -> float:
        """Saves the index into the folder `out`, and returns the seconds it took."""
        start = time.perf_counter()
        index.save(str(Path(out) / self.file_name))
        return time.perf_counter() - start


class RabitqWrites:
    """The write hooks of the indexes with add and remove, the IVF and the HNSW. Put it before RabitqAdapter among the
    bases. The library numbers the rows an add takes, so `insert` returns those ids for the session to check."""

    write_role = "full"
    durability = {"call": "memory", "sync": "power"}
    sync_kind = "whole"

    def writer_open(self, ctx: WriteContext, path: Path) -> Writer:
        file = self.index_file(path)
        return Writer(index=getattr(self.lib, self.kind).load(str(file)), file=file, threads=max(1, ctx.threads))

    def insert(self, w: Writer, ids: np.ndarray, rows: np.ndarray) -> np.ndarray:
        """One add. The library has no call for a single row, so a batch of one is an add of one row."""
        return w.index.add(np.ascontiguousarray(rows, dtype=np.float32), num_threads=w.threads)

    def delete(self, w: Writer, ids: np.ndarray) -> None:
        removed = int(w.index.remove(np.asarray(ids, dtype=np.int64)))

        if removed != len(ids):
            raise RuntimeError(f"{self.system} removed {removed} of {len(ids)} ids, so some were removed already")

    def sync(self, w: Writer) -> dict:
        return graphs.save_whole(w.file, w.index.save)

    def writer_close(self, w: Writer) -> None:
        w.index = None
