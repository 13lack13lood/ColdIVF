"""DiskANN from the diskann-disk crate, through the build tool and the library in baselines/diskann3.

`diskann3 build` builds an index of rows 0 to N of the base with DiskANN's own builder. The library libdiskann3.dylib
opens and searches it over ctypes, which lets go of Python's global interpreter lock (GIL) during every call. An index
is a folder with DiskANN's three files and a meta.json that records the build. index_disk.index holds the graph and the
full vectors in 4 KiB sectors. index_pq_pivots.bin and index_pq_compressed.bin hold the product quantization (PQ)
codes, which every search keeps in memory.

The variant picks the reader that serves DiskANN's sector reads.
- "pool" reads the sectors of a beam at once on a pool of threads, through a handle with F_NOCACHE.
- "advise" starts every read of a beam with F_RDADVISE, then reads each sector in order, through the same kind of
  handle.
- "stock" is DiskANN's own macOS reader, which copies the whole index file into memory at open. It runs only as a
  control of at most 1M vectors at one thread.

An open loads the PQ data, opens or copies the index file and reads its header, so in a cold trial it is the first
timed call. The read threads and the searcher's runtime are made before READY. A search takes "list", the search list
size, and "beam", the beam width. "cache_nodes" keeps that many nodes nearest the graph's entry point in memory, found
at open. C1 keeps 10,000 unless the settings say otherwise, and the other regimes keep none.

The pool reader's reads skip the page cache, so a trial's stats name the bytes it read of each file, which mincore can't
see. It leaves nothing cached for a rerun of the first query either, so its rerun first fills the cache through
F_RDADVISE.

A warm run in mode "whole" reads every file first, as for every system. Mode "pq" is the warm state of an index larger
than memory, as DiskANN serves one: the open holds the PQ files in memory, the index file's cached pages are dropped,
and every search reads its sectors from the disk. Only the pool reader runs it, since its reads leave the page cache as
it is, while the advise reader's fill it.
"""

from __future__ import annotations

import ctypes
import json
import os
import platform
import threading
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

from adapters.base import Adapter, BuildContext, ServeContext, Unsupported, main
import evictor
import schema

REPO = Path(__file__).resolve().parents[3]
BASELINES = REPO / "baselines"
RELEASE = Path(os.environ.get("DISKANN3_DIR") or BASELINES / "target" / "release")
LIB = RELEASE / "libdiskann3.dylib"
BIN = RELEASE / "diskann3"

PREFIX = "index"
SECTOR = 4096
PQ_FILES = ("index_pq_pivots.bin", "index_pq_compressed.bin")
DISK_FILE = "index_disk.index"

READERS = {"pool": 0, "advise": 1, "stock": 2}

# Build settings, each with its default and the build tool's flag. A default of None leaves DiskANN's own.
BUILD = {
    "degree": (64, "--degree"),
    "build_list": (100, "--list"),
    "pq_bytes": (96, "--pq-bytes"),
    "alpha": (None, "--alpha"),
    "saturate": (False, "--saturate"),
    "metric": ("l2", "--metric"),
    "build_ram_gb": (7.0, "--ram-gb"),
    "build_pq_bytes": (None, "--build-pq-bytes"),
}
SEARCH = ("list", "beam", "cache_nodes")

# The nodes C1 keeps in memory when its settings don't say.
C1_CACHE_NODES = 10_000

# The stock reader runs only as a control of at most this many vectors, at one thread.
STOCK_MAX_ROWS = 1_000_000

# A build metric other than the dataset's ranks the same only on unit vectors. Rows and queries whose lengths are
# within this of 1 count as unit vectors.
UNIT_TOL = 1e-3


class OpenInfo(ctypes.Structure):
    _fields_ = [("points", ctypes.c_uint64), ("dim", ctypes.c_uint32), ("reader", ctypes.c_uint32),
                ("pq_us", ctypes.c_double), ("reader_us", ctypes.c_double), ("searcher_us", ctypes.c_double),
                ("bytes", ctypes.c_uint64)]


class SearchInfo(ctypes.Structure):
    _fields_ = [("hops", ctypes.c_uint32), ("ios", ctypes.c_uint32), ("cmps", ctypes.c_uint32),
                ("results", ctypes.c_uint32), ("reads", ctypes.c_uint64), ("read_calls", ctypes.c_uint64),
                ("bytes_requested", ctypes.c_uint64), ("reader_ns", ctypes.c_uint64), ("io_us", ctypes.c_double),
                ("pq_us", ctypes.c_double), ("total_us", ctypes.c_double)]


def load_library(path: Path) -> ctypes.CDLL:
    """The library at `path`, with every function's argument and result types set."""
    lib = ctypes.CDLL(str(path))
    p, u32, u64 = ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint64
    lib.diskann3_version.argtypes, lib.diskann3_version.restype = [], ctypes.c_char_p
    lib.diskann3_reserve.argtypes, lib.diskann3_reserve.restype = [u32], ctypes.c_int64
    lib.diskann3_prepare.argtypes, lib.diskann3_prepare.restype = [], ctypes.c_int32
    lib.diskann3_open.argtypes = [ctypes.c_char_p, u32, ctypes.c_char_p, u32, u64, ctypes.POINTER(OpenInfo)]
    lib.diskann3_open.restype = p
    lib.diskann3_search.argtypes = [p, p, u32, u32, u32, u32, p, p, ctypes.POINTER(SearchInfo)]
    lib.diskann3_search.restype = ctypes.c_int64
    lib.diskann3_search_batch.argtypes = [p, p, u64, u32, u32, u32, u32, u32, p]
    lib.diskann3_search_batch.restype = ctypes.c_int32
    lib.diskann3_close.argtypes, lib.diskann3_close.restype = [p], None
    lib.diskann3_error.argtypes, lib.diskann3_error.restype = [ctypes.c_char_p, ctypes.c_size_t], ctypes.c_size_t
    lib.diskann3_prefill.argtypes, lib.diskann3_prefill.restype = [p, u32], ctypes.c_int32
    return lib


def build_config(config: dict) -> dict:
    """The build settings with their defaults filled in. An unknown setting is refused, so a typo never builds the
    default."""
    unknown = sorted(set(config) - set(BUILD))

    if unknown:
        raise Unsupported(f"DiskANN takes no build setting {unknown}")

    return {key: config.get(key, default) for key, (default, _) in BUILD.items()}


def build_flags(config: dict) -> list[str]:
    """The build tool's flags for filled-in build settings."""
    out = []

    for key, value in config.items():
        if value is not None:
            out += [BUILD[key][1], str(value).lower() if isinstance(value, bool) else str(value)]

    return out


def search_params(params: dict, k: int) -> tuple[int, int]:
    """The search list size and the beam width of a setting, checked."""
    unknown = sorted(set(params) - set(SEARCH))

    if unknown:
        raise Unsupported(f"DiskANN takes no search setting {unknown}")

    if "list" not in params or "beam" not in params:
        raise ValueError(f"a DiskANN search needs a list size and a beam width, not {params}")

    size, beam = int(params["list"]), int(params["beam"])

    if size < k or beam < 1:
        raise ValueError(f"the list size {size} must be at least k {k}, and the beam width {beam} at least 1")

    return size, beam


def cache_nodes(params: dict, regime: str) -> int:
    """How many nodes a search keeps in memory. C1 keeps 10,000 by default, and everything else none."""
    return int(params.get("cache_nodes", C1_CACHE_NODES if regime == "c1" else 0))


def reader_name(variant: str) -> str:
    """The reader a variant names. No variant means the pool reader."""
    name = variant or "pool"

    if name not in READERS:
        raise Unsupported(f"DiskANN has no reader {variant!r}. The readers are {sorted(READERS)}")

    return name


def index_folder(path: Path | str) -> Path:
    """The folder of an index, given the folder or one of its files."""
    path = Path(path)
    return path.parent if path.is_file() else path


def files_of(path: Path | str, reader: str = "pool") -> list[dict]:
    """The index's files and their ranges. The PQ files, the meta and the index file's first sector, its header, are
    routing data. The searcher holds them in memory after an open. The index file's other sectors are bulk data. The
    PQ files and the meta are also "loaded", since an open copies them into memory, and so is the whole index file for
    the stock reader, which copies it too."""
    folder = index_folder(path)
    disk = folder / DISK_FILE
    size = disk.stat().st_size
    copied = {"loaded": True} if reader == "stock" else {}
    out = []

    for name in (*PQ_FILES, "meta.json"):
        p = folder / name
        n = p.stat().st_size
        out.append({"path": str(p), "bytes": n,
                    "ranges": [{"offset": 0, "length": n, "kind": "routing", "loaded": True}]})

    ranges = [{"offset": 0, "length": min(SECTOR, size), "kind": "routing"} | copied]

    if size > SECTOR:
        ranges.append({"offset": SECTOR, "length": size - SECTOR, "kind": "bulk"} | copied)

    out.append({"path": str(disk), "bytes": size, "ranges": ranges})
    return out


def last_json(stdout: bytes) -> dict:
    """The last JSON object a program printed at the end of a line, past anything else it printed. Other text may come
    before it on its line, eg. the dots that DiskANN's merge of shards prints to stdout without a newline."""
    for line in reversed(stdout.decode(errors="replace").splitlines()):
        line = line.strip()
        start = line.find("{")

        while start >= 0:
            try:
                found = json.loads(line[start:])
            except ValueError:
                found = None

            if isinstance(found, dict):
                return found

            start = line.find("{", start + 1)

    raise RuntimeError("the build tool printed no report")


def build_fields(report: dict, rows: int) -> dict:
    """The build fields from the build tool's report. The index stores every full vector, for the rerank, and every
    PQ code."""
    meta = report.get("meta") or {}
    stages = {str(name): seconds for name, seconds in report.get("stages") or []}
    dim, pq_bytes = int(meta.get("dim") or 0), int(meta.get("pq_bytes") or 0)
    return {
        "seconds": report.get("seconds"),
        "cpu_seconds": report.get("cpu_seconds"),
        "peak_footprint": report.get("peak_footprint"),
        "bytes_read": report.get("bytes_read"),
        "bytes_written": report.get("bytes_written"),
        "cluster_seconds": None,
        "encode_seconds": stages.get("PqConstruction"),
        "stored_vector_bytes": rows * (dim * 4 + pq_bytes),
        "exact_f32_stored": True,
        "extra": {"stages": stages, "files": report.get("files"), "meta": meta,
                  "strategy": "sharded" if "PartitionData" in stages else "one shot"},
    }


def unit_rows(rows: np.ndarray) -> bool:
    """Whether every row has length 1, within UNIT_TOL."""
    norms = np.linalg.norm(np.asarray(rows, dtype=np.float64), axis=1)
    return bool(np.all(np.abs(norms - 1.0) <= UNIT_TOL))


def search_stats(info: SearchInfo, opened: dict, open_bytes: int = 0) -> dict:
    """The counters of a search, with the times of the open before it as stages too. `open_bytes` are the bytes that
    open read, which count toward the first search after it."""
    stages = dict(opened) | {
        "open_bytes_requested": open_bytes,
        "pq_table_us": info.pq_us,
        "node_reads_us": info.io_us,
        "search_us": info.total_us,
        "reader_us": info.reader_ns / 1e3,
        "hops": info.hops,
        "nodes_read": info.ios,
        "sector_reads": info.reads,
        "reader_calls": info.read_calls,
    }
    return {"stages": stages, "bytes_requested": open_bytes + info.bytes_requested, "dist_evals": info.cmps,
            "bytes_scanned": info.bytes_requested, "results": info.results}


def crate_version() -> str:
    try:
        return tomllib.loads((BASELINES / "Cargo.toml").read_text())["workspace"]["package"]["version"]
    except (OSError, KeyError, ValueError):
        return "unknown"


@dataclass
class Handle:
    """An open index. Each thread keeps its own output buffers and the counters of its last search. `unclaimed` holds
    the bytes the open read until the first search claims them, `open_bytes` the bytes of the index file among them,
    and `pq_sizes` the PQ files' sizes by path."""

    ptr: int | None
    dim: int
    points: int
    threads: int
    opened: dict
    unclaimed: int = 0
    local: threading.local = field(default_factory=threading.local)
    folder: Path | None = None
    regime: str = "c0"
    cache_nodes: int = 0
    open_bytes: int = 0
    pq_sizes: dict = field(default_factory=dict)


class DiskANN(Adapter):
    system = "diskann"
    binding = "ctypes"
    regimes = ("c0", "c1", "intent")
    batch_api = True

    def __init__(self) -> None:
        super().__init__()
        self._lib: ctypes.CDLL | None = None
        self.metas: dict[str, dict] = {}
        self.pq_bytes: dict[str, int] = {}
        self.hashes: dict[str, str | None] | None = None
        self.reserved = 0

    @property
    def lib(self) -> ctypes.CDLL:
        """The library, loaded on first use, so listing files and info work without it."""
        if self._lib is None:
            if not LIB.exists():
                raise FileNotFoundError(f"{LIB} is missing. Build it with cargo build --release in {BASELINES}")

            self._lib = load_library(LIB)

        return self._lib

    @property
    def reader(self) -> str:
        return reader_name(self.variant)

    def load(self) -> None:
        batch = "the threads of a batch search"
        self.workers = f"the pool reader's read threads, and {batch}" if self.reader == "pool" else batch

    def version(self) -> str:
        diskann = self.lib.diskann3_version().decode() if LIB.exists() else "unknown"
        tag = (self.wheels().get("libdiskann3") or "missing")[:12]
        return f"diskann {diskann}, diskann3 {crate_version()}+{tag}"

    def wheels(self) -> dict[str, str | None]:
        if self.hashes is None:
            self.hashes = {"libdiskann3": schema.file_sha256(LIB) if LIB.exists() else None,
                           "diskann3": schema.file_sha256(BIN) if BIN.exists() else None}

        return self.hashes

    def simd_level(self) -> str:
        return "neon" if platform.machine() == "arm64" else "runtime"

    def supported_regimes(self) -> tuple[str, ...]:
        # The stock reader holds the whole file in memory, so there is nothing to keep warm in C1.
        return ("c0", "intent") if self.reader == "stock" else self.regimes

    def allow(self, rows: int, threads: int) -> None:
        """Refuses the stock reader beyond its control of 1M vectors at one thread."""
        if self.reader == "stock" and (rows > STOCK_MAX_ROWS or threads > 1):
            raise Unsupported("DiskANN's stock reader runs only as a control of at most 1M vectors at one thread")

    def error(self) -> str:
        buf = ctypes.create_string_buffer(4096)
        self.lib.diskann3_error(buf, len(buf))
        return buf.value.decode(errors="replace")

    def reserve(self, threads: int) -> None:
        """Starts read threads until the process runs `threads` of them."""
        if threads > self.reserved:
            got = self.lib.diskann3_reserve(threads)

            if got < 0:
                raise RuntimeError(f"couldn't start read threads: {self.error()}")

            self.reserved = int(got)

    def meta(self, path: Path | str) -> dict:
        """The meta.json of the index at `path`, read once, with the size of its PQ files, which an open reads whole."""
        key = str(path)

        if key not in self.metas:
            folder = index_folder(path)
            self.metas[key] = json.loads((folder / "meta.json").read_text())
            self.pq_bytes[key] = sum((folder / name).stat().st_size for name in PQ_FILES)

        return self.metas[key]

    def check_metric(self, ctx: BuildContext, metric: str) -> None:
        """Refuses a build metric that may rank unlike the dataset's, which only unit vectors rule out."""
        if metric == ctx.metric:
            return

        rows = ctx.data[: min(ctx.rows, 1000)]
        queries = ctx.dataset.queries[: min(100, ctx.dataset.queries.shape[0])]

        if not (unit_rows(rows) and unit_rows(queries)):
            raise ValueError(f"metric {metric} ranks like {ctx.metric} only on unit vectors, and these aren't. Set "
                             f"metric = {ctx.metric!r} in the build config")

    def build(self, ctx: BuildContext) -> dict:
        if ctx.clusters is not None:
            raise Unsupported("DiskANN builds its own graph and imports no clusters")

        if not BIN.exists():
            raise FileNotFoundError(f"{BIN} is missing. Build it with cargo build --release in {BASELINES}")

        config = build_config(ctx.config)
        self.check_metric(ctx, config["metric"])
        cmd = [str(BIN), "build", "--data", str(ctx.data_path), "--rows", str(ctx.rows), "--out", str(ctx.out),
               "--threads", str(ctx.threads), "--seed", str(ctx.seed), *build_flags(config)]
        done = self.run_child(cmd)

        if done.returncode != 0:
            raise RuntimeError(f"diskann3 build exited with {done.returncode}")

        return build_fields(last_json(done.stdout), ctx.rows)

    def overhead_config(self) -> dict:
        return {}

    def overhead_params(self) -> dict:
        return {"list": 100, "beam": 4}

    def open(self, ctx: ServeContext, path: Path) -> Handle:
        self.meta(path)
        _, beam = search_params(ctx.params, ctx.k)
        threads = max(1, ctx.threads)
        self.allow(ctx.rows, threads)
        self.reserve(beam * threads)
        folder = index_folder(path)
        handle = Handle(ptr=None, dim=0, points=0, threads=threads, opened={}, folder=folder, regime=ctx.regime,
                        pq_sizes={str(folder / name): (folder / name).stat().st_size for name in PQ_FILES})
        self.open_ptr(handle, path, cache_nodes(ctx.params, ctx.regime))
        return handle

    def open_ptr(self, handle: Handle, path: Path, nodes: int) -> None:
        """Opens the index for `handle`, keeping `nodes` nodes in memory, and resets what the open read."""
        info = OpenInfo()
        prefix = str(index_folder(path) / PREFIX).encode()
        ptr = self.lib.diskann3_open(prefix, READERS[self.reader], self.meta(path)["metric"].encode(), handle.threads,
                                     nodes, ctypes.byref(info))

        if not ptr:
            raise RuntimeError(f"DiskANN couldn't open {path}: {self.error()}")

        handle.ptr, handle.dim, handle.points, handle.cache_nodes = ptr, int(info.dim), int(info.points), nodes
        handle.opened = {"open_pq_us": info.pq_us, "open_reader_us": info.reader_us,
                         "open_searcher_us": info.searcher_us}
        handle.open_bytes = int(info.bytes)
        handle.unclaimed = self.pq_bytes[str(path)] + int(info.bytes)

    def buffers(self, handle: Handle, k: int) -> tuple[np.ndarray, SearchInfo]:
        """The calling thread's id buffer and search counters for `handle`."""
        local = handle.local

        if getattr(local, "k", None) != k:
            local.ids, local.info, local.k = np.empty(k, dtype=np.int64), SearchInfo(), k

        return local.ids, local.info

    def search(self, handle: Handle, query: np.ndarray, k: int, params: dict) -> np.ndarray:
        size, beam = search_params(params, k)
        q = np.ascontiguousarray(query, dtype=np.float32)
        ids, info = self.buffers(handle, k)
        found = self.lib.diskann3_search(handle.ptr, q.ctypes.data, q.shape[0], k, size, beam, ids.ctypes.data, None,
                                         ctypes.byref(info))

        if found < 0:
            raise RuntimeError(f"DiskANN's search failed: {self.error()}")

        handle.local.open_bytes, handle.unclaimed = handle.unclaimed, 0
        return ids.copy()

    def search_batch(self, handle: Handle, queries: np.ndarray, k: int, params: dict, threads: int) -> np.ndarray:
        size, beam = search_params(params, k)
        self.reserve(beam * threads)
        qs = np.ascontiguousarray(queries, dtype=np.float32)
        out = np.empty((qs.shape[0], k), dtype=np.int64)
        rc = self.lib.diskann3_search_batch(handle.ptr, qs.ctypes.data, qs.shape[0], qs.shape[1], k, size, beam,
                                            threads, out.ctypes.data)

        if rc != 0:
            raise RuntimeError(f"DiskANN's batch search failed: {self.error()}")

        return out

    def configure(self, handle: Handle, params: dict) -> None:
        """Applies a search setting. The cached nodes are set at open, so a setting with another count opens the
        index again, as a fresh open with that setting would."""
        self.reserve(int(params.get("beam", 1)) * handle.threads)
        nodes = cache_nodes(params, handle.regime)

        if nodes != handle.cache_nodes:
            self.close(handle)
            self.open_ptr(handle, handle.folder, nodes)

    def close(self, handle: Handle) -> None:
        if handle.ptr:
            self.lib.diskann3_close(handle.ptr)
            handle.ptr = None

    def stats(self, handle: Handle) -> dict:
        """The counters of the calling thread's last search. `files_read` holds the bytes it read of each file, the
        PQ files and the index file's header too for the first search after an open. The pool reader reads through a
        handle that skips the page cache, so these counts show reads that mincore can't see."""
        info = getattr(handle.local, "info", None)

        if info is None:
            return {"stages": dict(handle.opened)}

        claimed = getattr(handle.local, "open_bytes", 0)
        out = search_stats(info, handle.opened, claimed)
        disk = int(info.bytes_requested) + (handle.open_bytes if claimed else 0)
        out["files_read"] = {str(handle.folder / DISK_FILE): disk} | (handle.pq_sizes if claimed else {})
        return out

    def resident_rerun(self, ctx: ServeContext, handle: Handle, query: np.ndarray) -> tuple[float | None, dict]:
        """The pool reader reads through a handle that skips the page cache, so a plain rerun would read every sector
        from the disk again. So one untimed search asks the kernel for each sector with F_RDADVISE as well, which
        brings them into the cache, and the timed rerun then reads them from there through the same reader. The
        sectors are dropped again after it, as the pool reader would leave them."""
        if self.reader != "pool":
            return super().resident_rerun(ctx, handle, query)

        self.lib.diskann3_prefill(handle.ptr, 1)

        try:
            self.search(handle, query, ctx.k, ctx.params)
        finally:
            self.lib.diskann3_prefill(handle.ptr, 0)

        rerun, extra = super().resident_rerun(ctx, handle, query)
        disk = str(handle.folder / DISK_FILE)
        size = Path(disk).stat().st_size
        dropped = evictor.drop_loaded([{"path": disk, "ranges": [{"offset": 0, "length": size, "loaded": True}]}])
        return rerun, extra | {"rerun_prefilled": True, "rerun_bytes_dropped": dropped}

    def warm_up(self, handle: Handle, ctx: ServeContext) -> None:
        """Mode "whole" reads every file, as the base class does. Mode "pq" drops the cached pages of the index file,
        so the searches read every sector from the disk, and keeps what the open holds, the PQ files and the header."""
        if ctx.mode != "pq":
            super().warm_up(handle, ctx)
            return

        if self.reader != "pool":
            raise Unsupported(f"DiskANN's warm mode pq needs the pool reader, whose reads skip the page cache, not "
                              f"the {self.reader} reader")

        disk = handle.folder / DISK_FILE
        evictor.drop_loaded([{"path": str(disk), "ranges": [{"offset": 0, "length": disk.stat().st_size,
                                                             "loaded": True}]}])

    def files(self, path: Path) -> list[dict]:
        return files_of(path, self.reader)

    def cold_setup(self, ctx: ServeContext) -> None:
        self.allow(ctx.rows, ctx.threads)
        _, beam = search_params(ctx.params, ctx.k)

        for path in (ctx.index, ctx.dummy):
            if path is not None:
                self.meta(path)

        self.reserve(beam * max(1, ctx.threads))
        super().cold_setup(ctx)

        if ctx.regime != "c1" and self.lib.diskann3_prepare() != 0:
            raise RuntimeError(f"couldn't make the searcher's runtime: {self.error()}")

    def warm_run(self, ctx: ServeContext, grid: list[dict], qids: np.ndarray, ids_dir: Path, passes: int,
                 warmup: int, qps_seconds: float, qps_repeats: int) -> Iterator[dict]:
        self.allow(ctx.rows, ctx.threads)

        for params in grid:
            search_params(params, ctx.k)

        return super().warm_run(ctx, grid, qids, ids_dir, passes, warmup, qps_seconds, qps_repeats)


if __name__ == "__main__":
    main(DiskANN)
