"""qdrant-edge, Qdrant's embedded vector store, through its Python binding qdrant-edge-py.

A build upserts every row as point i into one shard, with the original vectors on disk, then optimizes the shard into
one segment indexed by a hierarchical navigable small world (HNSW) graph. The codes are 8-bit scalar quantization (SQ8)
or 2-bit binary codes. By default they stay in RAM with the graph. With `inline` the graph is on disk and keeps each
node's neighbors' codes beside its links, so a hop reads ~2 pages and a load reads little more than the id tracker. A
search walks the graph on the codes and rescores `oversampling` times k candidates with the original vectors.

qdrant's write-ahead log (WAL) keeps every upsert, and nothing truncates it, so a load would read all of it. A build
removes the log after its final flush and loads the shard once, which starts a fresh empty log. Every load and close
also rewrites the shard's config and a few flag files, so the shard is a sealed store (see stores.py).

Searches run on one thread, which the build sets in the shard's config. Searches from several Python threads don't run
in parallel in this binding, so queries per second (QPS) at several threads run one process per worker.
"""

from __future__ import annotations

import importlib.metadata
import platform
import shutil
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from adapters import graphs, stores
from adapters.base import BuildContext, ServeContext, Unsupported, main
import schema

CODES = ("sq8", "bq2")
BUILD_KEYS = ("codes", "inline", "m", "ef_construct", "quantile")
SEARCH_KEYS = ("hnsw_ef", "oversampling", "rescore")
UPSERT_BATCH = 2000

# Thresholds of 1 KB make qdrant index the segment and search its graph at any size. A segment limit of 10^9 KB keeps
# every point in one segment.
THRESHOLD_KB = 1
MAX_SEGMENT_KB = 10**9


def library_distance(metric: str, unit: bool) -> str:
    """The qdrant distance for a dataset of `metric`. Unit vectors use the dot product, which ranks them as cosine and
    L2 do and needs no normalizing."""
    if unit:
        return "Dot"

    return {"cosine": "Cosine", "ip": "Dot", "l2": "Euclid"}[metric]


def settings_of(config: dict, metric: str, unit: bool) -> dict:
    """The build settings, with qdrant's defaults of m 16 and ef_construct 100 where the config leaves them out."""
    graphs.check_keys(config, BUILD_KEYS, "build setting", "qdrant-edge")
    codes = str(config.get("codes", "sq8"))

    if codes not in CODES:
        raise Unsupported(f"qdrant-edge here takes codes {CODES}, not {codes}")

    return {"codes": codes, "inline": bool(config.get("inline", False)), "m": int(config.get("m", 16)),
            "ef_construct": int(config.get("ef_construct", 100)), "quantile": config.get("quantile"),
            "distance": library_distance(metric, unit)}


def file_kind(rel: str, settings: dict) -> str:
    """Bulk data is what a search reads from disk. That is the original vectors, and with `inline` also the graph and
    the codes. The rest loads at open."""
    path = PurePosixPath(rel)

    if path.name == "matrix.dat":
        return "bulk"

    if settings["inline"] and (path.name == "quantized.data" or path.name.startswith("links")):
        return "bulk"

    return "routing"


def folder_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.is_dir() else 0


@dataclass
class Handle:
    shard: Any
    store: stores.Store
    folder: Path
    params: dict | None = None
    request: Any = None


class QdrantEdge(stores.StoreAdapter):
    system = "qdrant-edge"
    data = "shard"
    qps_mode = "processes"

    def load(self) -> None:
        import qdrant_edge

        self.qe = qdrant_edge

    def version(self) -> str:
        return importlib.metadata.version("qdrant-edge-py")

    def wheels(self) -> dict[str, str | None]:
        return {"qdrant-edge-py": schema.dist_sha256("qdrant-edge-py")}

    def simd_level(self) -> str | None:
        # qdrant picks NEON distance and quantization kernels at run time on Arm.
        return "neon" if platform.machine() == "arm64" else None

    def edge_config(self, s: dict, dim: int, threads: int) -> Any:
        qe = self.qe
        hnsw = qe.HnswIndexConfig(m=s["m"], ef_construct=s["ef_construct"], full_scan_threshold=THRESHOLD_KB,
                                  max_indexing_threads=threads, on_disk=s["inline"], inline_storage=s["inline"])
        in_ram = not s["inline"]

        if s["codes"] == "sq8":
            codes = qe.ScalarQuantizationConfig(type=qe.ScalarType.Int8, quantile=s["quantile"], always_ram=in_ram)
        else:
            codes = qe.BinaryQuantizationConfig(always_ram=in_ram, encoding=qe.BinaryQuantizationEncoding.TwoBits)

        optimizers = qe.EdgeOptimizersConfig(default_segment_number=1, max_segment_size=MAX_SEGMENT_KB,
                                              indexing_threshold=THRESHOLD_KB)
        vectors = qe.EdgeVectorParams(size=dim, distance=getattr(qe.Distance, s["distance"]), on_disk=True)
        return qe.EdgeConfig(vectors=vectors, hnsw_config=hnsw, quantization_config=codes, optimizers=optimizers,
                             max_search_threads=1)

    def build(self, ctx: BuildContext) -> dict:
        rows = ctx.data
        s = settings_of(ctx.config, ctx.metric, graphs.is_unit(rows))
        store = self.store(ctx.out)
        store.data.mkdir(parents=True)
        shard = self.qe.EdgeShard.create(str(store.data), self.edge_config(s, ctx.dim, ctx.threads))
        start = time.perf_counter()

        for a, b in graphs.batches(len(rows), UPSERT_BATCH):
            block = np.asarray(rows[a:b], dtype=np.float32)
            points = [self.qe.Point(id=a + i, vector=row.tolist()) for i, row in enumerate(block)]
            shard.update(self.qe.UpdateOperation.upsert_points(points))

        shard.flush()
        upsert_seconds = time.perf_counter() - start
        start = time.perf_counter()
        shard.optimize()
        shard.flush()
        optimize_seconds = time.perf_counter() - start
        info = shard.info()
        shard.close()
        wal_bytes = folder_bytes(store.data / "wal")
        shutil.rmtree(store.data / "wal")
        start = time.perf_counter()
        self.qe.EdgeShard.load(str(store.data)).close()
        settle_seconds = time.perf_counter() - start
        query = np.asarray(ctx.dataset.queries[0], dtype=np.float32)
        record = store.seal(lambda folder: self._session(folder, query))
        graphs.write_meta(ctx.out, {"system": self.system, "settings": s, "rows": len(rows), "dim": ctx.dim})
        names = {rel: PurePosixPath(rel).name for rel in record["files"]}
        parts = {part: sum(size for rel, (size, _) in record["files"].items() if names[rel].startswith(part))
                 for part in ("matrix.dat", "quantized.data", "links", "id_tracker")}
        return {
            "stored_vector_bytes": len(rows) * ctx.dim * 4,
            "exact_f32_stored": True,
            "extra": {
                "settings": s, "upsert_seconds": upsert_seconds, "optimize_seconds": optimize_seconds,
                "settle_seconds": settle_seconds, "segments": info.segments_count, "points": info.points_count,
                "indexed_vectors": info.indexed_vectors_count, "wal_bytes_removed": wal_bytes, "wal_removed": True,
                "bytes_by_part": parts, "stashed_files": record["stash"], "store_bytes": store.store_bytes(),
                "stash_bytes": store.stash_bytes(), "seed_used": False,
            },
        }

    def _session(self, folder: Path, query: np.ndarray) -> None:
        shard = self.qe.EdgeShard.load(str(folder))
        shard.search(self.request(query, 10, self.search_params({})))
        shard.close()

    def overhead_config(self) -> dict:
        return {"codes": "sq8"}

    def overhead_params(self) -> dict:
        return {"hnsw_ef": 64}

    def search_params(self, params: dict) -> Any:
        graphs.check_keys(params, SEARCH_KEYS, "search setting", self.system)
        oversampling = float(params["oversampling"]) if params.get("oversampling") is not None else None
        codes = self.qe.QuantizationSearchParams(rescore=bool(params.get("rescore", True)), oversampling=oversampling)
        ef = int(params["hnsw_ef"]) if params.get("hnsw_ef") is not None else None
        return self.qe.SearchParams(hnsw_ef=ef, quantization=codes)

    def request(self, query: np.ndarray, k: int, params: Any) -> Any:
        return self.qe.SearchRequest(query=self.qe.Query.Nearest(query.tolist()), limit=k, params=params,
                                     with_vector=False, with_payload=False)

    def open(self, ctx: ServeContext, path: Path) -> Handle:
        store, folder = self.begin(path, self.meta(path))
        handle = Handle(shard=self.qe.EdgeShard.load(str(folder)), store=store, folder=folder)
        self.configure(handle, ctx.params)
        return handle

    def configure(self, handle: Handle, params: dict) -> None:
        handle.request, handle.params = self.search_params(params), params

    def search(self, handle: Handle, query: np.ndarray, k: int, params: dict) -> np.ndarray:
        if params is not handle.params:
            self.configure(handle, params)

        found = handle.shard.search(self.request(query, k, handle.request))
        return graphs.pad([int(p.id) for p in found], k)

    def close(self, handle: Handle) -> None:
        handle.shard.close()
        self.end(handle.store, handle.folder)

    def kind(self, rel: str, meta: dict) -> str:
        return file_kind(rel, meta["settings"])

    def loaded(self, rel: str, meta: dict) -> bool:
        """Codes kept in RAM are copied at load, as a cold load's footprint grows by their size. Without `inline`
        that is quantized.data."""
        return PurePosixPath(rel).name == "quantized.data" and not meta["settings"]["inline"]

    def files(self, path: Path) -> list[dict]:
        return self.store_files(Path(path), self.meta(path))


if __name__ == "__main__":
    main(QdrantEdge)
