"""AsterVec, the log-structured merge (LSM) graph index once called LSM-VEC, through its Python binding aster-vec.

The upper layers of its hierarchical navigable small world (HNSW) graph live in RAM, the bottom layer's edges in a
RocksDB store, and the vectors as 8-bit codes in a paged file. A build loads every row with bulk_build, which gives row
i the id i. The first open after bulk_build moves its write-ahead log into tables, so a build opens and closes the
store once more before it seals it. Every open writes a new manifest, options file and log, and every close rewrites
the graph's metadata, so the store is sealed (see stores.py).

Caches are set when the store opens. "default" keeps the library's caches, a 32 MiB block cache, 8,192 pages of 4 KiB
for vectors and 100,000 edge lists. "minimal" sets all three to 0. A search takes ef, the size of its candidate list,
and runs on the calling thread. Searches from several Python threads don't run in parallel in this binding, so queries
per second (QPS) at several threads run one process per worker.
"""

from __future__ import annotations

import importlib.metadata
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from adapters import graphs, stores
from adapters.base import BuildContext, ServeContext, Unsupported, main
import schema

DB = "db"
VECTORS = "vectors.bin"
BUILD_KEYS = ("m", "m_max", "m_level", "ef_construction")
CACHE_KEYS = ("block_cache_bytes", "edge_cache_size", "paged_max_cached_pages")
SEARCH_KEYS = ("ef", "caches", *CACHE_KEYS)
CACHES = {"default": {}, "minimal": dict.fromkeys(CACHE_KEYS, 0)}
DEFAULT_EF = 128


def library_metric(metric: str, unit: bool) -> str:
    """The AsterVec metric for a dataset of `metric`. AsterVec offers L2 and cosine, and cosine ranks unit vectors as
    the inner product does."""
    if metric == "l2":
        return "L2"

    if metric == "cosine" or unit:
        return "Cosine"

    raise Unsupported("AsterVec offers L2 and cosine, which rank these inner product rows differently")


def cache_settings(params: dict) -> dict:
    """The cache sizes of a search setting, from its `caches` preset and any size given on its own."""
    name = str(params.get("caches", "default"))

    if name not in CACHES:
        raise Unsupported(f"AsterVec here has caches {sorted(CACHES)}, not {name}")

    return CACHES[name] | {k: int(params[k]) for k in CACHE_KEYS if params.get(k) is not None}


def file_kind(rel: str) -> str:
    """Bulk data is what a search reads from disk, the vector file and the RocksDB tables of edges. The rest is read at
    open."""
    return "bulk" if rel == VECTORS or PurePosixPath(rel).suffix == ".sst" else "routing"


@dataclass
class Handle:
    db: Any
    store: stores.Store
    folder: Path
    meta: dict
    caches: dict = field(default_factory=dict)
    ef: int = DEFAULT_EF
    params: dict | None = None


class AsterVec(stores.StoreAdapter):
    system = "astervec"
    data = "store"
    qps_mode = "processes"

    def load(self) -> None:
        import astervec

        self.av = astervec

    def version(self) -> str:
        return importlib.metadata.version("aster-vec")

    def wheels(self) -> dict[str, str | None]:
        return {"aster-vec": schema.dist_sha256("aster-vec")}

    def options(self, folder: Path, settings: dict, dim: int, caches: dict) -> Any:
        o = self.av.AsterVecDBOptions()
        o.dim = dim
        o.metric = getattr(self.av, settings["metric"])
        o.m, o.m_max, o.m_level = settings["m"], settings["m_max"], settings["m_level"]
        o.ef_construction = float(settings["ef_construction"])
        o.random_seed = settings["seed"]
        o.vector_file_path = str(folder / VECTORS)
        o.reinit = False

        for key, value in caches.items():
            setattr(o, key, value)

        return o

    def settings(self, ctx: BuildContext) -> dict:
        graphs.check_keys(ctx.config, BUILD_KEYS, "build setting", self.system)
        defaults = self.av.AsterVecDBOptions()
        out = {k: int(ctx.config.get(k, getattr(defaults, k))) for k in ("m", "m_max", "m_level")}
        out["ef_construction"] = float(ctx.config.get("ef_construction", defaults.ef_construction))
        return out | {"metric": library_metric(ctx.metric, graphs.is_unit(ctx.data)), "seed": int(ctx.seed)}

    def build(self, ctx: BuildContext) -> dict:
        s = self.settings(ctx)
        store = self.store(ctx.out)
        (store.data / DB).mkdir(parents=True)
        opts = self.options(store.data, s, ctx.dim, {})
        opts.reinit = True
        db = self.av.AsterVecDB.open(str(store.data / DB), opts)
        start = time.perf_counter()
        report = db.bulk_build(ctx.data, threads=ctx.threads)
        db.close()
        build_seconds = time.perf_counter() - start
        start = time.perf_counter()
        self.av.AsterVecDB.open(str(store.data / DB), self.options(store.data, s, ctx.dim, {})).close()
        settle_seconds = time.perf_counter() - start
        meta = {"system": self.system, "settings": s, "rows": len(ctx.data), "dim": ctx.dim}
        query = np.asarray(ctx.dataset.queries[0], dtype=np.float32)
        record = store.seal(lambda folder: self._session(folder, meta, query))
        graphs.write_meta(ctx.out, meta)
        tables = sum(size for rel, (size, _) in record["files"].items() if file_kind(rel) == "bulk" and rel != VECTORS)
        return {
            "stored_vector_bytes": record["files"][VECTORS][0],
            "exact_f32_stored": False,
            "extra": {
                "settings": s, "bulk_build_seconds": build_seconds, "settle_seconds": settle_seconds,
                "report": report, "edge_table_bytes": tables, "stashed_files": record["stash"],
                "store_bytes": store.store_bytes(), "stash_bytes": store.stash_bytes(), "seed_used": True,
            },
        }

    def _session(self, folder: Path, meta: dict, query: np.ndarray) -> None:
        db = self.av.AsterVecDB.open(str(folder / DB), self.options(folder, meta["settings"], meta["dim"], {}))
        db.search_knn(query, 10, DEFAULT_EF)
        db.close()

    def overhead_params(self) -> dict:
        return {"ef": 64}

    def open(self, ctx: ServeContext, path: Path) -> Handle:
        graphs.check_keys(ctx.params, SEARCH_KEYS, "search setting", self.system)
        caches = cache_settings(ctx.params)
        meta = self.meta(path)
        store, folder = self.begin(path, meta)
        db = self.av.AsterVecDB.open(str(folder / DB), self.options(folder, meta["settings"], meta["dim"], caches))
        handle = Handle(db=db, store=store, folder=folder, meta=meta, caches=caches)
        self.configure(handle, ctx.params)
        return handle

    def configure(self, handle: Handle, params: dict) -> None:
        graphs.check_keys(params, SEARCH_KEYS, "search setting", self.system)
        caches = cache_settings(params)

        if caches != handle.caches:
            # Caches are set at open. So a setting with other caches closes the store, restores it if this process
            # holds it, and opens it again.
            handle.db.close()

            if handle.folder == handle.store.data:
                handle.store.restore()

            options = self.options(handle.folder, handle.meta["settings"], handle.meta["dim"], caches)
            handle.db, handle.caches = self.av.AsterVecDB.open(str(handle.folder / DB), options), caches

        handle.ef, handle.params = int(params.get("ef", DEFAULT_EF)), params

    def search(self, handle: Handle, query: np.ndarray, k: int, params: dict) -> np.ndarray:
        if params is not handle.params:
            self.configure(handle, params)

        return graphs.pad([r.id for r in handle.db.search_knn(query, k, handle.ef)], k)

    def close(self, handle: Handle) -> None:
        handle.db.close()
        self.end(handle.store, handle.folder)

    def kind(self, rel: str, meta: dict) -> str:
        return file_kind(rel)

    def files(self, path: Path) -> list[dict]:
        return self.store_files(Path(path), self.meta(path))


if __name__ == "__main__":
    main(AsterVec)
