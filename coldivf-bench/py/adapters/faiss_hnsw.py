"""faiss hierarchical navigable small world (HNSW) graphs, opened with IO_FLAG_MMAP_IFC so their codes stay mapped.

| Storage | Index | Rescoring |
|---|---|---|
| sq8 | IndexHNSWSQ over 8-bit scalar quantized (SQ8) codes | None |
| rabitq | IndexHNSWRaBitQ over RaBitQ codes, 4 bits by default | The best k × k_factor candidates against SQ8 rows |
| flat | IndexHNSWFlat over float32 rows | None |

A build sets efConstruction, 200 by default where faiss's own default is 40, and seeds the levels with the build's
seed. It adds every row in one call, so the codes and neighbor lists are sized once. read_index maps the codes but reads
the whole graph section, i.e. the levels, the offsets and the neighbor lists, ~272 bytes per vector at M 32. So files()
marks the graph section as routing data and the codes as bulk data, and the disk-resident steady state (C1) keeps the
graph cached. Searches set efSearch and k_factor through search parameters, since this faiss can't downcast an
IndexHNSWRaBitQ read back from a file.

It takes inserts only. faiss's graph indexes have no remove_ids, and their one way to leave rows out is a search filter
that still walks the deleted nodes. An insert is add on the index loaded whole, since a mapped one can't take writes,
and faiss numbers the rows itself from its count, which a stream checks against the stream's ids. A search finds an
insert at once. The durable point saves the whole file and rewrites meta.json with the new file's parts.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from adapters.base import BuildContext, ServeContext, Unsupported, WriteContext, main
from adapters.faiss_common import (
    META, TRAIN_ROWS, FaissAdapter, Writer, check_unit, faiss_metric, metric_type, parsed, parts_record,
    sample_rows, set_threads, whole, write_meta,
)
from adapters.inproc import settings

INDEX = "index.faiss"
BUILD_KEYS = {"m", "storage", "bits", "refine", "ef_construction", "train_rows"}
STORAGES = ("sq8", "rabitq", "flat")
MAX_BITS = 9
SEARCH_KEYS = {"ef_search", "k_factor"}
# faiss's own efSearch, which a search without ef_search keeps.
DEFAULT_EF_SEARCH = 16


@dataclass
class Spec:
    """A build's settings, from its config."""

    m: int = 32
    storage: str = "sq8"
    bits: int | None = None
    refine: str | None = None
    ef_construction: int = 200
    train_rows: int = TRAIN_ROWS

    @classmethod
    def parse(cls, config: dict) -> Spec:
        unknown = sorted(set(config) - BUILD_KEYS)

        if unknown:
            raise ValueError(f"the faiss HNSW build takes no {unknown}")

        spec = cls(**config)

        if spec.storage not in STORAGES:
            raise ValueError(f"storage is one of {STORAGES}, not {spec.storage!r}")

        if spec.storage == "rabitq":
            spec.bits = 4 if spec.bits is None else int(spec.bits)
            spec.refine = "sq8" if spec.refine is None else spec.refine

            if not 1 <= spec.bits <= MAX_BITS:
                raise ValueError(f"RaBitQ takes 1 to {MAX_BITS} bits, not {spec.bits}")
        elif spec.bits is not None:
            raise ValueError(f"bits is a setting of RaBitQ storage, not of {spec.storage}")

        spec.refine = spec.refine or "none"

        if spec.refine not in ("sq8", "none"):
            raise ValueError(f"refine is sq8 or none, not {spec.refine!r}")

        if spec.refine == "sq8" and spec.storage != "rabitq":
            raise ValueError(f"only RaBitQ storage takes refine rows, and {spec.storage} codes don't need them")

        for name in ("m", "ef_construction", "train_rows"):
            if int(getattr(spec, name)) < 1:
                raise ValueError(f"{name} must be at least 1, not {getattr(spec, name)}")

        return spec


@dataclass
class Handle:
    """An open graph index and the settings its searches use."""

    top: Any
    refine: bool
    ef_search: int = DEFAULT_EF_SEARCH
    k_factor: float = 1.0
    params: Any = None
    # The search parameters faiss points at, which must live as long as `params`.
    keep: list = field(default_factory=list)
    # The bytes of one node's code and of one refine row, and how many rows the last search rescored.
    code_bytes: int = 0
    refine_bytes: int = 0
    k_base: int = 0


def search_params(ef_search: int, k_factor: float, refine: bool) -> tuple[Any, list]:
    """faiss search parameters for one setting, and the objects they point at."""
    import faiss

    graph = faiss.SearchParametersHNSW()
    graph.efSearch = int(ef_search)

    if not refine:
        return graph, [graph]

    top = faiss.IndexRefineSearchParameters()
    top.k_factor = float(k_factor)
    top.base_index_params = graph
    return top, [graph, top]


class FaissHNSW(FaissAdapter):
    system = "faiss-hnsw"
    write_role = "inserts"

    def load(self) -> None:
        super().load()
        import faiss

        self._stats = faiss.cvar.hnsw_stats
        self._rabitq = faiss.cvar.rabitq_stats

    def overhead_config(self) -> dict:
        return {"m": 16, "ef_construction": 40}

    def overhead_params(self) -> dict:
        return {"ef_search": 16}

    def build(self, ctx: BuildContext) -> dict:
        import faiss

        if ctx.clusters is not None:
            raise Unsupported("faiss HNSW takes no shared clusters")

        set_threads(ctx.threads)
        spec = Spec.parse(ctx.config)
        metric = faiss_metric(ctx.metric)
        rows = ctx.data
        check_unit(rows, ctx.metric)

        if spec.storage == "rabitq" and metric != "l2":
            raise Unsupported("faiss's HNSW over RaBitQ codes takes only L2")

        if spec.storage == "sq8":
            base = faiss.IndexHNSWSQ(ctx.dim, faiss.ScalarQuantizer.QT_8bit, int(spec.m), metric_type(metric))
        elif spec.storage == "flat":
            base = faiss.IndexHNSWFlat(ctx.dim, int(spec.m), metric_type(metric))
        else:
            base = faiss.IndexHNSWRaBitQ(ctx.dim, int(spec.m), int(spec.bits))

        base.hnsw.efConstruction = int(spec.ef_construction)
        base.hnsw.rng = faiss.RandomGenerator(int(ctx.seed))
        sample = sample_rows(rows, int(spec.train_rows), ctx.seed)
        trained_rows = len(sample)
        t0 = time.perf_counter()
        base.train(sample)
        rows_index = None

        if spec.refine == "sq8":
            rows_index = faiss.IndexScalarQuantizer(ctx.dim, faiss.ScalarQuantizer.QT_8bit, metric_type(metric))
            rows_index.train(sample)

        del sample
        t1 = time.perf_counter()
        everything = np.ascontiguousarray(rows, dtype=np.float32)
        base.add(everything)

        if rows_index is not None:
            rows_index.add(everything)

        t2 = time.perf_counter()
        top = faiss.IndexRefine(base, rows_index) if rows_index is not None else base
        faiss.write_index(top, str(ctx.out / INDEX))
        parts = parts_record(ctx.out, [INDEX])
        graph = sum(p["length"] for p in parts if p["name"] == "graph")
        codes = sum(p["length"] for p in parts if p["name"] == "storage")
        refine_bytes = int(rows_index.code_size) if rows_index is not None else 0
        write_meta(ctx.out, {
            "system": self.system, "variant": self.variant, "config": ctx.config, "storage": spec.storage,
            "bits": spec.bits, "refine": spec.refine, "m": int(spec.m), "ef_construction": int(spec.ef_construction),
            "metric": ctx.metric, "faiss_metric": metric, "dim": ctx.dim, "rows": ctx.rows, "seed": ctx.seed,
            "code_bytes": codes // max(1, ctx.rows), "refine_bytes": refine_bytes, "train_rows": trained_rows,
            "faiss_version": faiss.__version__, "files": [INDEX], "parts": parts,
        })
        return {
            "cluster_seconds": None,
            "encode_seconds": t2 - t1,
            "stored_vector_bytes": codes + ctx.rows * refine_bytes,
            "exact_f32_stored": spec.storage == "flat",
            "extra": {"train_seconds": t1 - t0, "train_rows": trained_rows, "graph_bytes": graph,
                      "storage_bytes": codes, "refine_bytes": refine_bytes, "faiss_metric": metric},
        }

    def open(self, ctx: ServeContext, path: Path) -> Handle:
        import faiss

        meta = self.meta(path)
        top = faiss.read_index(str(Path(path) / INDEX), faiss.IO_FLAG_MMAP_IFC)
        h = Handle(top=top, refine=meta.get("refine") == "sq8", code_bytes=int(meta.get("code_bytes") or 0),
                   refine_bytes=int(meta.get("refine_bytes") or 0))
        self.configure(h, ctx.params)
        return h

    def configure(self, h: Handle, params: dict) -> None:
        """Applies a search setting. A key the setting leaves out gets the value a fresh open has, efSearch 16 and
        k_factor 1, so a warm setting searches as a fresh open with the same settings does."""
        unknown = sorted(set(params) - SEARCH_KEYS)

        if unknown:
            raise Unsupported(f"faiss HNSW takes no search setting {unknown}")

        ef_search = params.get("ef_search")
        k_factor = float(params["k_factor"]) if params.get("k_factor") is not None else 1.0

        if not h.refine and k_factor != 1:
            raise Unsupported("an index without refine rows takes only k_factor 1")

        h.ef_search = int(ef_search) if ef_search is not None else DEFAULT_EF_SEARCH
        h.k_factor = k_factor
        h.params, h.keep = search_params(h.ef_search, h.k_factor, h.refine)

    def search(self, h: Handle, query: np.ndarray, k: int, params: dict) -> np.ndarray:
        self._stats.reset()
        self._rabitq.reset()
        h.k_base = int(k * h.k_factor) if h.refine else 0
        _, ids = h.top.search(query.reshape(1, -1), k, params=h.params)
        return ids[0]

    def search_batch(self, h: Handle, queries: np.ndarray, k: int, params: dict, threads: int) -> np.ndarray:
        set_threads(threads)
        return h.top.search(np.ascontiguousarray(queries, dtype=np.float32), k, params=h.params)[1]

    def stats(self, h: Handle) -> dict:
        s, r = self._stats, self._rabitq
        # Every distance reads one node's code, counted at its full size, and every rescored candidate one row.
        scanned = s.ndis * h.code_bytes + h.k_base * h.refine_bytes
        out = {"dist_evals": float(s.ndis), "bytes_scanned": float(scanned), "hops": int(s.nhops),
               "rabitq_1bit": int(r.n_1bit), "rabitq_refine": int(r.n_refine)}
        out.update(self.first_query_parts())
        return out

    def files(self, path: Path) -> list[dict]:
        path = Path(path)

        if path.is_file():
            return [parsed(path)]

        out = []

        for f in sorted(p for p in path.iterdir() if p.is_file()):
            # IO_FLAG_MMAP_IFC maps the graph and the codes, so only the meta file, read before any trial, is loaded.
            out.append(whole(f, "routing", loaded=True) if f.name == META else parsed(f))

        return out

    def writer_open(self, ctx: WriteContext, path: Path) -> Writer:
        """Reads the index whole, without a map, since faiss writes only to a loaded index."""
        import faiss

        settings(ctx.params, {}, "faiss HNSW write settings")
        path = Path(path)
        return Writer(folder=path, file=path / INDEX, top=faiss.read_index(str(path / INDEX)), meta=self.meta(path))

    def insert(self, w: Writer, ids: np.ndarray, rows: np.ndarray) -> np.ndarray:
        """One add, which links each row into the graph and numbers it from the index's count. It returns those
        numbers, which a stream checks."""
        x = np.ascontiguousarray(rows, dtype=np.float32)
        start = int(w.top.ntotal)
        w.top.add(x)
        return np.arange(start, start + len(x), dtype=np.int64)

    def pending(self, w: Writer) -> dict:
        """What the loaded index holds. Each insert is in the graph at once, so nothing waits."""
        return {"live_rows": int(w.top.ntotal)}

    def saved_meta(self, w: Writer) -> dict:
        return w.meta | {"parts": parts_record(w.folder, [INDEX]), "state_rows": int(w.top.ntotal),
                         "live_rows": int(w.top.ntotal)}


if __name__ == "__main__":
    main(FaissHNSW)
