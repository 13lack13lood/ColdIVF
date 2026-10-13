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

For the write benchmark it takes full writes, `upsert_points` and `delete_points`, one operation per call, which live
in memory until `flush`, the sync. Its one maintenance action, `optimize`, merges every point into one segment with the
build's config, so it rebuilds the whole graph. A write session opens a full clone of a state in place, never a private
copy. qdrant applies an operation to a point only when the operation's number is at least the point's version. A log
the build removed starts again at 0, while the base points keep the numbers of the build's upserts, so before its
first write a session on such a log sends deletes of an absent id, untimed, until the next number is past them. The
states keep their log, since qdrant never truncates it, and a load replays it. So a state records in meta.json the
operations its log holds, and its lines record the log's bytes. A state then takes one more load and close, as the
build does, and is sealed again.
"""

from __future__ import annotations

import importlib.metadata
import json
import platform
import shutil
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

import numpy as np

from adapters import graphs, stores
from adapters.base import BuildContext, ServeContext, Unsupported, WriteContext, main
import schema
import writes

CODES = ("sq8", "bq2")
BUILD_KEYS = ("codes", "inline", "m", "ef_construct", "quantile")
SEARCH_KEYS = ("hnsw_ef", "oversampling", "rescore")
UPSERT_BATCH = 2000

# Thresholds of 1 KB make qdrant index the segment and search its graph at any size. A segment limit of 10^9 KB keeps
# every point in one segment.
THRESHOLD_KB = 1
MAX_SEGMENT_KB = 10**9
# The id the version fix-up deletes, which no state holds.
ABSENT_ID = 1 << 62
WAL = "wal"
CONFIG = "edge_config.json"


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


def fixup_ops(rows: int) -> int:
    """The operations a log the build removed must hold before a delete of any base point applies. The build's last
    upsert numbered its points ceil(rows / UPSERT_BATCH) - 1, and one more makes room, eg. 501 at 1M rows."""
    return -(-rows // UPSERT_BATCH) + 1


def log_bytes(shard: Path) -> dict:
    """The size of qdrant's log in the shard folder and its files. qdrant makes each file of its log 32 MB at once and
    fills it, so the size is the log's to within one file."""
    files = [p for p in (Path(shard) / WAL).rglob("*") if p.is_file()] if (Path(shard) / WAL).is_dir() else []
    return {"wal_bytes": sum(p.stat().st_size for p in files), "wal_files": len(files)}


@dataclass
class Handle:
    shard: Any
    store: stores.Store
    folder: Path
    params: dict | None = None
    request: Any = None


@dataclass
class Writer:
    """A shard open for writes, in place in a write session's folder. `ops` counts the operations its log holds since
    the build removed it, and `fixup` is how many it needs before a delete of a base point applies."""

    shard: Any
    store: stores.Store
    folder: Path
    meta: dict
    ops: int
    fixup: int
    fixed: int = 0
    inserts: int = 0
    deletes: int = 0


class QdrantEdge(stores.StoreAdapter):
    system = "qdrant-edge"
    data = "shard"
    qps_mode = "processes"
    write_role = "full"
    # A write lives in memory until flush, even with its log, and flush syncs with F_FULLFSYNC.
    durability = {"call": "memory", "sync": "power"}
    sync_kind = "incremental"
    maintenance_actions = ("optimize",)
    # A write goes into files of a set size that qdrant made earlier, eg. the log's segments, the vector chunks and the
    # flags of deleted points.
    writes_in_place = True

    def __init__(self) -> None:
        super().__init__()
        self.logs: dict[str, dict] = {}

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

    def prepare(self, path: Path) -> None:
        super().prepare(path)
        self.log_of(path)

    def log_of(self, path: Path) -> dict:
        """The size of the log of the index at `path` as the state left it, read once, for the lines' extra."""
        key = str(path)

        if key not in self.logs:
            self.logs[key] = log_bytes(self.store(path).data) | {"log_ops": self.meta(path).get("log_ops", 0)}

        return self.logs[key]

    def stats(self, handle: Handle) -> dict:
        return super().stats(handle) | self.log_of(handle.store.index)

    def warm_run(self, ctx: ServeContext, grid: list[dict], qids: np.ndarray, ids_dir: Path, passes: int,
                 warmup: int, qps_seconds: float, qps_repeats: int) -> Iterator[dict]:
        # The log as the state left it, once the claim has restored the store.
        self.claim(ctx.index)
        log = self.log_of(ctx.index)

        for result in super().warm_run(ctx, grid, qids, ids_dir, passes, warmup, qps_seconds, qps_repeats):
            result["fields"]["extra"] = (result["fields"].get("extra") or {}) | log
            yield result

    def writer_open(self, ctx: WriteContext, path: Path) -> Writer:
        """Loads the shard of a state for writes. A maintenance run, the only session at more than 1 thread, loads it
        with the build's config at its own indexing threads, which qdrant writes into the shard."""
        meta = self.meta(path)
        store = self.write_store(path)
        config = None

        if ctx.threads > 1 and indexing_threads(store.data) != ctx.threads:
            config = self.edge_config(meta["settings"], meta["dim"], ctx.threads)

        shard = self.qe.EdgeShard.load(str(store.data), config)
        return Writer(shard=shard, store=store, folder=store.data, meta=meta, ops=int(meta.get("log_ops", 0)),
                      fixup=fixup_ops(int(meta["rows"])))

    def prepare_writes(self, w: Writer) -> None:
        """Sends the version fix-up before a session's first write, untimed, as deletes of an absent id, while the log
        holds fewer operations than the build's upserts numbered."""
        while w.ops < w.fixup:
            w.shard.update(self.qe.UpdateOperation.delete_points([ABSENT_ID]))
            w.ops, w.fixed = w.ops + 1, w.fixed + 1

    def insert(self, w: Writer, ids: np.ndarray, rows: np.ndarray) -> None:
        points = [self.qe.Point(id=i, vector=row.tolist()) for i, row in zip(ids.tolist(), rows)]
        w.shard.update(self.qe.UpdateOperation.upsert_points(points))
        w.ops, w.inserts = w.ops + 1, w.inserts + len(ids)

    def delete(self, w: Writer, ids: np.ndarray) -> None:
        w.shard.update(self.qe.UpdateOperation.delete_points(ids.tolist()))
        w.ops, w.deletes = w.ops + 1, w.deletes + len(ids)

    def sync(self, w: Writer) -> dict:
        w.shard.flush()
        return {"call": "flush"}

    def maintain(self, w: Writer, action: str, threads: int, params: dict) -> dict:
        """optimize with the build's one-segment config, which merges every point into one new segment and builds its
        whole graph again, then flush, which makes it survive a power loss. Each step's seconds go in the report."""
        if action not in self.maintenance_actions:
            raise Unsupported(f"{self.system} has no maintenance action {action}")

        before = w.shard.info()
        t0 = time.perf_counter()
        w.shard.optimize()
        t1 = time.perf_counter()
        w.shard.flush()
        t2 = time.perf_counter()
        after = w.shard.info()
        return {
            "rows_live": int(after.points_count), "durability": "power", "rebuild": True,
            "optimize_seconds": t1 - t0, "flush_seconds": t2 - t1, "indexing_threads": indexing_threads(w.folder),
            "points": int(after.points_count), "indexed_before": int(before.indexed_vectors_count),
            "indexed_after": int(after.indexed_vectors_count), "segments_before": int(before.segments_count),
            "segments_after": int(after.segments_count),
        }

    def pending(self, w: Writer) -> dict:
        """The points qdrant holds and has indexed, the log, and this session's writes and fix-up."""
        info = w.shard.info()
        return {"points": int(info.points_count), "indexed": int(info.indexed_vectors_count),
                "segments": int(info.segments_count), "log_ops": w.ops, "fixup_ops": w.fixed, "inserts": w.inserts,
                "deletes": w.deletes} | log_bytes(w.folder)

    def writer_close(self, w: Writer) -> None:
        w.shard.close()
        w.store.release()
        meta = self.meta(w.store.index)

        if meta.get("log_ops", 0) != w.ops:
            self._meta[str(w.store.index)] = meta | {"log_ops": w.ops}

    def seal_state(self, ctx: WriteContext, path: Path) -> dict:
        """Records the log's operations in meta.json, then one more load and close, as the build does, and the
        build's seal."""
        meta = self.meta(path)
        writes.write_json(Path(path) / graphs.META, meta)
        writes.fullfsync(Path(path) / graphs.META)
        store = self.store(path)

        if not store.claim():
            raise RuntimeError(f"another process holds the store at {store.data}")

        self.qe.EdgeShard.load(str(store.data)).close()
        query = np.asarray(ctx.dataset.queries[0], dtype=np.float32)
        return self.reseal(path, lambda folder: self._session(folder, query)) | log_bytes(store.data) | \
            {"log_ops": meta.get("log_ops", 0)}


def indexing_threads(shard: Path) -> int | None:
    """The indexing threads in the shard's config, or None where it doesn't say."""
    try:
        return int(json.loads((Path(shard) / CONFIG).read_text())["hnsw_config"]["max_indexing_threads"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


if __name__ == "__main__":
    main(QdrantEdge)
