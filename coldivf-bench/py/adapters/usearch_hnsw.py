"""usearch's hierarchical navigable small world (HNSW) graph, served from its file with view().

A build adds row i with key i on the build's threads and saves one file. The file holds an 8-byte size header and the
vectors, then a 64-byte header and the graph. The graph is a 40-byte header, a 16-bit level for each node, then each
node's key, level and neighbor lists. view() maps the file without loading it, and searches fault in the pages they
touch. The vectors are bulk data. The rest is routing data, which the disk-resident steady state (C1) leaves cached as
the process found it.

Vectors are stored as i8 with cosine, which usearch's 8-bit codes are made for, or as f16 or f32 with the inner
product, which ranks unit vectors as cosine does. A search takes expansion_search, the size of its candidate list, and
runs on one thread. The batch call runs its own threads outside Python's lock, so queries per second (QPS) at several
threads use it.

Writes go to the index loaded whole, since a view can't change, with key lookups on, which remove needs. An insert is
add with the stream's ids as keys and a delete is remove, one key or many per call, on the session's threads. A remove
leaves the node in the graph with a free key, so searches still walk it and leave it out of their results, and a later
add takes over the node and its vector slot. The durable point saves the whole file beside the old one, syncs it and
renames it over the old one, since a save over a file that a view maps kills the viewing process. There is no
maintenance action.
"""

from __future__ import annotations

import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from adapters import graphs
from adapters.base import Adapter, BuildContext, ServeContext, Unsupported, WriteContext, main
import schema

FILE = "index.usearch"
DTYPES = ("i8", "f16", "f32")
BUILD_KEYS = ("dtype", "connectivity", "expansion_add")
SEARCH_KEYS = ("expansion_search",)
VECTOR_BYTES = {"i8": 1, "f16": 2, "bf16": 2, "f32": 4, "f64": 8}
ADD_BATCH = 100_000

# The parts of the file, as usearch's save writes them.
SIZES_BYTES = 8
DENSE_HEAD_BYTES = 64
GRAPH_HEAD_BYTES = 40
LEVEL_BYTES = 2
MAGIC = b"usearch"


@dataclass
class Layout:
    """Where the parts of an index file lie. `vector_bytes` is the size of one stored vector."""

    rows: int
    vector_bytes: int
    nodes: int
    connectivity: int
    connectivity_base: int
    size: int

    @property
    def vectors_end(self) -> int:
        return SIZES_BYTES + self.rows * self.vector_bytes

    @property
    def graph_start(self) -> int:
        return self.vectors_end + DENSE_HEAD_BYTES


def layout(path: Path | str) -> Layout:
    """The layout of a usearch file saved with its vectors and 32-bit sizes, read from its headers."""
    path = Path(path)
    size = path.stat().st_size

    with open(path, "rb") as f:
        rows, vector_bytes = struct.unpack("<II", f.read(SIZES_BYTES))
        f.seek(SIZES_BYTES + rows * vector_bytes)
        dense = f.read(DENSE_HEAD_BYTES)
        graph = f.read(GRAPH_HEAD_BYTES)

    if not dense.startswith(MAGIC) or len(graph) < GRAPH_HEAD_BYTES:
        raise ValueError(f"{path} isn't a usearch file with its vectors in it")

    nodes, connectivity, connectivity_base, _, _ = struct.unpack("<QQQQQ", graph)
    out = Layout(rows, vector_bytes, nodes, connectivity, connectivity_base, size)

    if size < out.graph_start + GRAPH_HEAD_BYTES + LEVEL_BYTES * nodes:
        raise ValueError(f"{path} is shorter than its headers say")

    return out


def index_file(path: Path | str) -> Path:
    """The usearch file of an index, given the file or its folder."""
    path = Path(path)
    return path if path.is_file() else path / FILE


def library_metric(dtype: str, metric: str, unit: bool) -> str:
    """The usearch metric for vectors stored as `dtype`, for a dataset of `metric` whose rows are unit length or not.
    It is the inner product wherever that ranks the rows the same, since it is the fastest."""
    if dtype == "i8":
        if metric == "cosine" or unit:
            return "cos"

        raise Unsupported(f"usearch stores i8 only with cosine, which ranks these {metric} rows differently")

    if unit:
        return "ip"

    return {"cosine": "cos", "ip": "ip", "l2": "l2sq"}[metric]


@dataclass
class Handle:
    index: Any
    vector_bytes: int | None
    computed: int = 0
    visited: int = 0


@dataclass
class Writer:
    """The loaded index of a write session, the file its durable point replaces, and the threads its calls take."""

    index: Any
    file: Path
    threads: int


class USearch(Adapter):
    system = "usearch"
    batch_api = True
    workers = "its own threads, for batch searches"
    write_role = "full"
    durability = {"call": "memory", "sync": "power"}
    sync_kind = "whole"

    def load(self) -> None:
        import usearch
        from usearch import compiled
        from usearch.index import Index

        self.usearch, self.compiled, self.Index = usearch, compiled, Index

    def version(self) -> str:
        return self.usearch.__version__

    def wheels(self) -> dict[str, str | None]:
        return {name: schema.dist_sha256(name) for name in ("usearch", "numkong")}

    def simd_level(self) -> str:
        # The kernels usearch picks for each stored type, at 768 dimensions.
        kinds = self.compiled.ScalarKind
        metrics = self.compiled.MetricKind
        picks = {"i8": (kinds.I8, metrics.Cos), "f16": (kinds.F16, metrics.IP), "f32": (kinds.F32, metrics.IP)}
        return " ".join(f"{name}:{self.compiled.hardware_acceleration(dtype=kind, ndim=768, metric_kind=metric)}"
                        for name, (kind, metric) in picks.items())

    def build(self, ctx: BuildContext) -> dict:
        graphs.check_keys(ctx.config, BUILD_KEYS, "build setting", self.system)
        dtype = str(ctx.config.get("dtype", "i8"))

        if dtype not in DTYPES:
            raise Unsupported(f"usearch here stores {DTYPES}, not {dtype}")

        rows = ctx.data
        metric = library_metric(dtype, ctx.metric, graphs.is_unit(rows))
        connectivity = int(ctx.config.get("connectivity", self.compiled.DEFAULT_CONNECTIVITY))
        expansion_add = int(ctx.config.get("expansion_add", self.compiled.DEFAULT_EXPANSION_ADD))
        index = self.Index(ndim=ctx.dim, metric=metric, dtype=dtype, connectivity=connectivity,
                           expansion_add=expansion_add, enable_key_lookups=False)
        start = time.perf_counter()

        for a, b in graphs.batches(len(rows), ADD_BATCH):
            index.add(np.arange(a, b, dtype=np.uint64), rows[a:b], threads=ctx.threads)

        add_seconds = time.perf_counter() - start
        start = time.perf_counter()
        index.save(str(ctx.out / FILE))
        save_seconds = time.perf_counter() - start
        levels = [s.nodes for s in index.levels_stats]
        acceleration = index.hardware_acceleration
        index.reset()
        lay = layout(ctx.out / FILE)
        settings = {"dtype": dtype, "metric": metric, "connectivity": connectivity, "expansion_add": expansion_add}
        graphs.write_meta(ctx.out, {"system": self.system, "settings": settings, "rows": len(rows), "dim": ctx.dim})
        return {
            "stored_vector_bytes": lay.rows * lay.vector_bytes,
            "exact_f32_stored": dtype == "f32",
            "extra": {
                "settings": settings, "add_seconds": add_seconds, "save_seconds": save_seconds,
                "hardware_acceleration": acceleration, "nodes_per_level": levels, "seed_used": False,
                "vector_section_bytes": lay.rows * lay.vector_bytes, "graph_section_bytes": lay.size - lay.vectors_end,
            },
        }

    def overhead_config(self) -> dict:
        return {"dtype": "i8"}

    def overhead_params(self) -> dict:
        return {"expansion_search": int(self.compiled.DEFAULT_EXPANSION_SEARCH)}

    def open(self, ctx: ServeContext, path: Path) -> Handle:
        graphs.check_keys(ctx.params, SEARCH_KEYS, "search setting", self.system)
        index = self.Index.restore(str(index_file(path)), view=True, enable_key_lookups=False)

        if index is None:
            raise FileNotFoundError(f"{index_file(path)} isn't a usearch index")

        width = VECTOR_BYTES.get(str(index.dtype).split(".")[-1].lower())
        handle = Handle(index=index, vector_bytes=width * index.ndim if width else None)
        self.configure(handle, ctx.params)
        return handle

    def configure(self, handle: Handle, params: dict) -> None:
        graphs.check_keys(params, SEARCH_KEYS, "search setting", self.system)
        handle.index.expansion_search = int(params.get("expansion_search", self.compiled.DEFAULT_EXPANSION_SEARCH))

    def search(self, handle: Handle, query: np.ndarray, k: int, params: dict) -> np.ndarray:
        found = handle.index.search(query, count=k, threads=1)
        handle.computed, handle.visited = found.computed_distances, found.visited_members
        return graphs.pad(found.keys, k)

    def search_batch(self, handle: Handle, queries: np.ndarray, k: int, params: dict, threads: int) -> np.ndarray:
        found = handle.index.search(queries, count=k, threads=threads)

        if not hasattr(found, "counts"):
            # usearch hands back a single query's result for a batch of one.
            return graphs.pad(found.keys, k)[None, :]

        ids = found.keys.astype(np.int64)
        ids[np.arange(ids.shape[1])[None, :] >= found.counts[:, None]] = -1
        return ids

    def stats(self, handle: Handle) -> dict:
        scanned = handle.computed * handle.vector_bytes if handle.vector_bytes else None
        return {"dist_evals": handle.computed, "bytes_scanned": scanned, "visited": handle.visited}

    def close(self, handle: Handle) -> None:
        handle.index.reset()

    def files(self, path: Path) -> list[dict]:
        file = index_file(path)
        lay = layout(file)
        ranges = [
            {"offset": 0, "length": SIZES_BYTES, "kind": "routing"},
            {"offset": SIZES_BYTES, "length": lay.rows * lay.vector_bytes, "kind": "bulk"},
            {"offset": lay.vectors_end, "length": lay.size - lay.vectors_end, "kind": "routing"},
        ]
        out = [{"path": str(file), "bytes": lay.size, "ranges": ranges}]
        meta = file.parent / graphs.META

        if meta.is_file():
            out.append({"path": str(meta), "bytes": meta.stat().st_size, "ranges": []})

        return out

    def writer_open(self, ctx: WriteContext, path: Path) -> Writer:
        file = index_file(path)
        index = self.Index.restore(str(file), view=False, enable_key_lookups=True)

        if index is None:
            raise FileNotFoundError(f"{file} isn't a usearch index")

        # The file keeps the links per node, but not the candidate list of adds, which comes from the build's meta.
        meta = file.parent / graphs.META
        settings = graphs.read_meta(file.parent).get("settings", {}) if meta.is_file() else {}

        if "expansion_add" in settings:
            index.expansion_add = int(settings["expansion_add"])

        return Writer(index=index, file=file, threads=max(1, ctx.threads))

    def insert(self, w: Writer, ids: np.ndarray, rows: np.ndarray) -> None:
        if len(ids) == 1:
            w.index.add(int(ids[0]), rows[0], threads=w.threads)
        else:
            w.index.add(np.asarray(ids, dtype=np.uint64), rows, threads=w.threads)

    def delete(self, w: Writer, ids: np.ndarray) -> None:
        if len(ids) == 1:
            removed = int(w.index.remove(int(ids[0])))
        else:
            removed = int(w.index.remove(np.asarray(ids, dtype=np.uint64)))

        if removed != len(ids):
            raise RuntimeError(f"usearch removed {removed} of {len(ids)} keys, so the index lacked some of them")

    def sync(self, w: Writer) -> dict:
        return graphs.save_whole(w.file, lambda tmp: w.index.save(tmp))

    def pending(self, w: Writer) -> dict:
        """The live keys and the graph's nodes, whose difference is the removed nodes that searches still walk."""
        nodes = int(w.index.stats.nodes)
        return {"live": len(w.index), "nodes": nodes, "removed_nodes": nodes - len(w.index)}

    def writer_close(self, w: Writer) -> None:
        w.index.reset()


if __name__ == "__main__":
    main(USearch)
