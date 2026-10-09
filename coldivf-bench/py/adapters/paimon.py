"""Apache Paimon's vector index (paimon-vindex) with IVF-RQ, an inverted file index (IVF) over RaBitQ codes.

An index folder holds index.paimon, the file the library's writer wrote, and meta.json with the build's settings and
the file's routing span, the bytes a reader keeps in memory once it has read them. The library trains its own clusters,
assigns each row to its nearest one, and has no rerank against raw vectors, so its recall is what its codes reach. A
cosine dataset of unit vectors uses the "inner_product" metric, which ranks it the same way.

A reader reads through an input object. Each round of reads hands the input all of its byte ranges at once through
pread_many, and the library's documentation asks the input to read them concurrently. A cold first query reads three
rounds. Opening reads the 64-byte header, the first search reads the centroids and the list offsets in one range, and
then every probed list. The routing span is the header plus that range, and the lists are bulk data.

The variant picks the input. By default a thread pool runs each round's preads at once, one range per pread, with up to
"read_threads" in flight. "rdadvise" asks macOS to read every range of a round ahead with F_RDADVISE, then reads them in
turn on the calling thread. "serial" reads them in turn with plain preads. Every round is timed and counted, so a cold
trial reports its rounds, ranges and bytes.
"""

from __future__ import annotations

import fcntl
import json
import os
import platform
import struct
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from adapters.base import Adapter, BuildContext, ServeContext, Unsupported, main
import schema

INDEX_FILE = "index.paimon"
META = "meta.json"
HEADER_BYTES = 64
MAGIC = b"QRVI"
BATCH_ROWS = 65536
READ_THREADS = 64
# Each list's entry in the offset table after the centroids, an offset and a length.
OFFSET_ENTRY_BYTES = 16
# fcntl's F_RDADVISE on macOS, which takes a struct radvisory of an off_t offset and an int length.
F_RDADVISE = 44
RADVISORY = struct.Struct("qi4x")
MAX_ADVISE = (1 << 31) - 1
# Rounds an input keeps for stats. A long batch run drops the oldest.
KEEP_ROUNDS = 4096
UNIT_TOLERANCE = 1e-3

BUILD_KEYS = {"bits", "nlist", "vpl", "train_rows", "batch_rows", "metric", "options"}
SEARCH_KEYS = {"nprobe", "read_threads", "memory_budget_bytes"}
VARIANTS = ("", "pool", "rdadvise", "serial")


def paimon_metric(metric: str, unit: bool) -> str:
    """Paimon's metric for a dataset metric. Inner product ranks unit vectors as cosine does."""
    if metric == "l2":
        return "l2"

    if metric == "ip" or unit:
        return "inner_product"

    return "cosine"


def unit_rows(rows: np.ndarray, sample: int = 1000, tolerance: float = UNIT_TOLERANCE) -> bool:
    """Whether evenly spaced sample rows all have length 1, within `tolerance`."""
    if len(rows) == 0:
        return False

    picks = np.unique(np.linspace(0, len(rows) - 1, min(sample, len(rows))).astype(np.int64))
    norms = np.linalg.norm(np.asarray(rows[picks], dtype=np.float64), axis=1)
    return bool(np.all(np.abs(norms - 1.0) <= tolerance))


def nlist_of(config: dict, rows: int) -> int | None:
    """The list count a build asks for. It is "nlist" as given, or the row count over "vpl" (vectors per list), or
    None for the library's own choice."""
    if config.get("nlist"):
        return int(config["nlist"])

    if config.get("vpl"):
        return max(1, round(rows / float(config["vpl"])))

    return None


def build_options(config: dict, dim: int, rows: int, metric: str) -> dict[str, str]:
    """The writer's options, all strings. Coarse assignment is exact, and "options" in the config adds or overrides
    any of them."""
    nlist = nlist_of(config, rows)
    out = {
        "index.type": "ivf_rq",
        "dimension": str(dim),
        "rq.bits": str(int(config.get("bits", 4))),
        "metric": metric,
        "ivf.coarse-assignment": "exact",
    }

    if nlist is None:
        out.update({"nlist": "auto", "expected-vector-count": str(rows)})
    else:
        out["nlist"] = str(nlist)

    out.update({str(k): str(v) for k, v in (config.get("options") or {}).items()})
    return out


def training_rows(rows: int, limit: int | None) -> np.ndarray | None:
    """Evenly spaced row ids for a training sample of `limit` rows, or None to train on every row."""
    if not limit or limit >= rows:
        return None

    return np.unique(np.linspace(0, rows - 1, int(limit)).round().astype(np.int64))


def check_build(config: dict) -> None:
    unknown = sorted(set(config) - BUILD_KEYS)

    if unknown:
        raise ValueError(f"Paimon takes no build setting {unknown}")


def check_search(params: dict) -> None:
    unknown = sorted(set(params) - SEARCH_KEYS)

    if unknown:
        raise ValueError(f"Paimon takes no search setting {unknown}")

    if "nprobe" not in params:
        raise ValueError("a Paimon search needs nprobe")


def header_fields(head: bytes) -> dict:
    """The header fields this adapter relies on, the dimension and the list count, from the file's first 64 bytes."""
    if len(head) < HEADER_BYTES or head[:4] != MAGIC:
        raise ValueError("not a Paimon IVF-RQ index file")

    version, dim, _, nlist = struct.unpack_from("<4I", head, 4)
    return {"version": version, "dim": dim, "nlist": nlist}


def routing_formula(dim: int, nlist: int) -> int:
    """Where the routing span ends by the format, i.e. the header, then nlist centroids of dim float32 values, then
    the list offset table."""
    return HEADER_BYTES + nlist * (dim * 4 + OFFSET_ENTRY_BYTES)


def routing_end(path: Path | str) -> int:
    """The end of an index's routing span, from meta.json when the build measured it, else from the header."""
    folder = Path(path)
    meta = folder / META

    if meta.exists():
        return int(json.loads(meta.read_text())["routing_end"])

    with open(folder / INDEX_FILE, "rb") as f:
        fields = header_fields(f.read(HEADER_BYTES))

    return routing_formula(fields["dim"], fields["nlist"])


def measured_routing(rounds: list[list[tuple[int, int]]]) -> int:
    """The routing span's end from the ranges of the first two rounds a reader asked for, the header at open and the
    centroids with the offsets at the first search. They must cover the file from byte 0 without a gap."""
    spans = sorted(r for rnd in rounds[:2] for r in rnd)
    end = 0

    for offset, length in spans:
        if offset > end:
            raise ValueError(f"the reader's first rounds leave a gap at byte {end}")

        end = max(end, offset + length)

    return end


def pread_full(fd: int, offset: int, length: int) -> bytes:
    """Exactly `length` bytes at `offset`. A pread may return fewer, so it reads on until it has them all."""
    chunk = os.pread(fd, length, offset)

    if len(chunk) == length:
        return chunk

    parts = [chunk]
    done = len(chunk)

    while done < length:
        more = os.pread(fd, length - done, offset + done)

        if not more:
            raise OSError(f"the file ends at {offset + done}, before {offset + length}")

        parts.append(more)
        done += len(more)

    return b"".join(parts)


def advise(fd: int, offset: int, length: int) -> None:
    """Asks macOS to read a range into the page cache ahead of use, in pieces that fit radvisory's int length."""
    while length > 0:
        piece = min(length, MAX_ADVISE)
        fcntl.fcntl(fd, F_RDADVISE, RADVISORY.pack(offset, piece))
        offset, length = offset + piece, length - piece


class RoundLog:
    """Every round an input served, as (ranges, bytes, microseconds), numbered from 0 for the input's life. Rounds can
    come from several threads at once, eg. in a batch search, so a lock guards the list."""

    def __init__(self, keep: int = KEEP_ROUNDS, ranges: bool = False):
        self.lock = threading.Lock()
        self.rounds: list[tuple[int, int, float]] = []
        self.spans: list[list[tuple[int, int]]] = []
        self.first = 0
        self.keep = keep
        self.keep_ranges = ranges

    @property
    def count(self) -> int:
        return self.first + len(self.rounds)

    def add(self, ranges: list[tuple[int, int]], us: float) -> None:
        with self.lock:
            self.rounds.append((len(ranges), sum(n for _, n in ranges), round(us, 1)))

            if self.keep_ranges:
                self.spans.append(list(ranges))

            if len(self.rounds) > self.keep:
                drop = len(self.rounds) - self.keep
                del self.rounds[:drop]
                self.first += drop

    def between(self, start: int, end: int) -> list[tuple[int, int, float]]:
        """The rounds numbered start to end that are still kept."""
        with self.lock:
            lo, hi = max(0, start - self.first), max(0, end - self.first)
            return list(self.rounds[lo:hi])


class Input:
    """A reader's input over one file. It times each round of pread_many and logs it. The read hints stay 0, which
    leaves them to the library."""

    estimated_random_read_latency_nanos = 0
    preferred_window_bytes = 0
    max_ranges_per_read = 0

    def __init__(self, path: Path | str, log: RoundLog | None = None):
        self.fd = os.open(path, os.O_RDONLY)
        self.log = log or RoundLog()

    def pread_many(self, ranges: list[tuple[int, int]]) -> list[bytes]:
        start = time.perf_counter_ns()

        try:
            chunks = self.read(ranges)
        except Exception:
            # The binding turns any error here into a failed search without its cause, so the cause goes to stderr.
            traceback.print_exc(file=sys.stderr)
            raise

        self.log.add(ranges, (time.perf_counter_ns() - start) / 1e3)
        return chunks

    def read(self, ranges: list[tuple[int, int]]) -> list[bytes]:
        raise NotImplementedError

    def close(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1


class SerialInput(Input):
    """Plain preads, one range after another."""

    def read(self, ranges: list[tuple[int, int]]) -> list[bytes]:
        return [pread_full(self.fd, offset, length) for offset, length in ranges]


class PoolInput(Input):
    """One pread per range on a thread pool, so a round's ranges are in flight together, up to the pool's size."""

    def __init__(self, path: Path | str, pool: ThreadPoolExecutor, log: RoundLog | None = None):
        super().__init__(path, log)
        self.pool = pool

    def read(self, ranges: list[tuple[int, int]]) -> list[bytes]:
        if len(ranges) < 2:
            return [pread_full(self.fd, offset, length) for offset, length in ranges]

        return list(self.pool.map(lambda r: pread_full(self.fd, r[0], r[1]), ranges))


class AdviseInput(Input):
    """F_RDADVISE for every range of a round, so the drive reads them together, then plain preads in turn."""

    def read(self, ranges: list[tuple[int, int]]) -> list[bytes]:
        for offset, length in ranges:
            advise(self.fd, offset, length)

        return [pread_full(self.fd, offset, length) for offset, length in ranges]


def round_stats(rounds: list[tuple[int, int, float]]) -> dict:
    """The stats fields of a search from its rounds."""
    nbytes = sum(r[1] for r in rounds)
    return {
        "bytes_requested": nbytes,
        "bytes_scanned": nbytes,
        "stages": {
            "read_us": round(sum(r[2] for r in rounds), 1),
            "rounds": len(rounds),
            "round_ranges": [r[0] for r in rounds],
            "round_bytes": [r[1] for r in rounds],
            "round_us": [r[2] for r in rounds],
        },
    }


@dataclass
class Handle:
    reader: Any
    input: Input
    # The log's round count after the last search, or 0 for a fresh input, so the first search's rounds include the
    # header read at open.
    mark: int = 0
    last: tuple[int, int] = (0, 0)
    searches: dict = field(default_factory=dict)


class Paimon(Adapter):
    system = "paimon"
    # search_batch runs its queries on the library's own thread pool, sized by RAYON_NUM_THREADS.
    batch_api = True
    workers = "the adapter's read pool, and Rayon's pool for batch searches"

    def __init__(self) -> None:
        super().__init__()
        self.pv: Any = None
        self.pool: ThreadPoolExecutor | None = None
        self.pool_size = 0

    def load(self) -> None:
        if self.variant not in VARIANTS:
            raise ValueError(f"Paimon's variant is one of {VARIANTS}, not {self.variant!r}")

        import paimon_vindex

        self.pv = paimon_vindex

    def version(self) -> str:
        return f"paimon-vindex {_dist_version('paimon-vindex')}"

    def wheels(self) -> dict[str, str | None]:
        return {"paimon-vindex": schema.dist_sha256("paimon-vindex")}

    def simd_level(self) -> str | None:
        # The library is Rust, whose 64-bit Arm builds always have NEON.
        return "neon" if platform.machine() == "arm64" else None

    def overhead_config(self) -> dict:
        return {"bits": 4, "nlist": 4}

    def overhead_params(self) -> dict:
        return {"nprobe": 1}

    def build(self, ctx: BuildContext) -> dict:
        config = dict(ctx.config)
        check_build(config)

        if ctx.clusters is not None:
            raise Unsupported("Paimon trains its own clusters and takes no partition")

        rows = ctx.data
        metric = config.get("metric") or paimon_metric(ctx.metric, ctx.metric == "cosine" and unit_rows(rows))
        options = build_options(config, ctx.dim, ctx.rows, metric)
        batch = int(config.get("batch_rows", BATCH_ROWS))
        sample = training_rows(ctx.rows, config.get("train_rows"))
        start = time.perf_counter()
        trainer = self.pv.VectorIndexTrainer(options)

        if sample is None:
            for lo in range(0, ctx.rows, batch):
                trainer.add_training_vectors(np.ascontiguousarray(rows[lo : lo + batch], dtype=np.float32))
        else:
            for lo in range(0, len(sample), batch):
                trainer.add_training_vectors(np.ascontiguousarray(rows[sample[lo : lo + batch]], dtype=np.float32))

        training = trainer.finish_training()
        trained = time.perf_counter()
        writer = self.pv.VectorIndexWriter(training)

        for lo in range(0, ctx.rows, batch):
            block = np.ascontiguousarray(rows[lo : lo + batch], dtype=np.float32)
            writer.add_vectors(np.arange(lo, lo + len(block), dtype=np.int64), block)

        added = time.perf_counter()
        file = ctx.out / INDEX_FILE

        with open(file, "wb") as f:
            writer.write(f)
            f.flush()
            os.fsync(f.fileno())

        writer.close()
        written = time.perf_counter()
        layout = self.layout(file, np.asarray(ctx.dataset.queries[0], dtype=np.float32))
        meta = {"file": INDEX_FILE, "options": options, "rows": ctx.rows, "dim": ctx.dim,
                "train_rows": int(len(sample)) if sample is not None else ctx.rows, **layout}
        (ctx.out / META).write_text(json.dumps(meta, indent=1))
        size = file.stat().st_size
        return {
            "cluster_seconds": trained - start,
            "encode_seconds": added - trained,
            "stored_vector_bytes": 0,
            "exact_f32_stored": False,
            "extra": {"write_seconds": written - added, "metric": metric, "nlist": layout["nlist"],
                      "rq_bits": layout["rq_bits"], "train_rows": meta["train_rows"],
                      "routing_bytes": layout["routing_end"], "list_bytes": size - layout["routing_end"],
                      "seed": "the library takes no seed"},
        }

    def layout(self, file: Path, query: np.ndarray) -> dict:
        """Opens the new file once with an input that keeps every range, runs one search that probes one list, and
        takes the routing span from the reads. It checks the span against the format."""
        log = RoundLog(ranges=True)
        source = SerialInput(file, log)
        reader = self.pv.VectorIndexReader(source)

        try:
            reader.search(query, self.pv.SearchParams.ivf(top_k=1, nprobe=1))
            meta = reader.metadata()
        finally:
            reader.close()
            source.close()

        end = measured_routing(log.spans)
        formula = routing_formula(meta.dimension, meta.nlist)

        if end != formula:
            raise ValueError(f"the reader's routing span ends at {end}, but the format says {formula}")

        return {"nlist": meta.nlist, "rq_bits": meta.rq_bits, "metric": meta.metric, "index_type": meta.index_type,
                "total_vectors": meta.total_vectors, "routing_end": end}

    def files(self, path: Path) -> list[dict]:
        """The index file's routing span and lists, and the meta file. The routing span and the meta are also
        "loaded", since the reader keeps the span in memory once it has read it, and the adapter reads the meta once."""
        folder = Path(path)
        file = folder / INDEX_FILE
        size = file.stat().st_size
        end = min(size, routing_end(folder))
        ranges = [{"offset": 0, "length": end, "kind": "routing", "loaded": True}]

        if size > end:
            ranges.append({"offset": end, "length": size - end, "kind": "bulk"})

        out = [{"path": str(file), "bytes": size, "ranges": ranges}]
        meta = folder / META

        if meta.exists():
            n = meta.stat().st_size
            out.append({"path": str(meta), "bytes": n,
                        "ranges": [{"offset": 0, "length": n, "kind": "routing", "loaded": True}]})

        return out

    def pool_for(self, threads: int) -> ThreadPoolExecutor:
        """The shared read pool, made again when a setting asks for another size. Its threads run at the QoS class a
        new thread starts with, as every library's own threads do, and only the timed thread asks for the
        user-interactive class."""
        if self.pool is None or self.pool_size != threads:
            if self.pool is not None:
                self.pool.shutdown(wait=False)

            self.pool = ThreadPoolExecutor(threads, thread_name_prefix="pread")
            self.pool_size = threads

        return self.pool

    def prestart(self, threads: int) -> None:
        """Starts every thread of the read pool, so no trial waits for one to start."""
        pool = self.pool_for(threads)
        gate = threading.Barrier(threads + 1)
        futures = [pool.submit(gate.wait) for _ in range(threads)]
        gate.wait()

        for f in futures:
            f.result()

    def make_input(self, file: Path, params: dict) -> Input:
        if self.variant == "rdadvise":
            return AdviseInput(file)

        if self.variant == "serial":
            return SerialInput(file)

        return PoolInput(file, self.pool_for(int(params.get("read_threads", READ_THREADS))))

    def configure(self, handle: Handle, params: dict) -> None:
        check_search(params)

        if isinstance(handle.input, PoolInput):
            handle.input.pool = self.pool_for(int(params.get("read_threads", READ_THREADS)))

    def open(self, ctx: ServeContext, path: Path) -> Handle:
        source = self.make_input(Path(path) / INDEX_FILE, ctx.params)
        options = {}

        if ctx.params.get("memory_budget_bytes") is not None:
            options["memory_budget_bytes"] = int(ctx.params["memory_budget_bytes"])

        return Handle(reader=self.pv.VectorIndexReader(source, **options), input=source)

    def close(self, handle: Handle) -> None:
        handle.reader.close()
        handle.input.close()

    def params_of(self, handle: Handle, k: int, params: dict) -> Any:
        key = (k, int(params["nprobe"]))

        if key not in handle.searches:
            handle.searches[key] = self.pv.SearchParams.ivf(top_k=k, nprobe=key[1])

        return handle.searches[key]

    def search(self, handle: Handle, query: np.ndarray, k: int, params: dict) -> np.ndarray:
        start = handle.mark
        ids, _ = handle.reader.search(query, self.params_of(handle, k, params))
        handle.mark = handle.input.log.count
        handle.last = (start, handle.mark)
        return ids

    def search_batch(self, handle: Handle, queries: np.ndarray, k: int, params: dict, threads: int) -> np.ndarray:
        ids, _ = handle.reader.search_batch(np.ascontiguousarray(queries, dtype=np.float32),
                                            self.params_of(handle, k, params))
        handle.mark = handle.input.log.count
        return ids

    def stats(self, handle: Handle) -> dict:
        """The rounds of the last search. A fresh handle's first search counts the header read at open too."""
        return round_stats(handle.input.log.between(*handle.last))

    def cold_setup(self, ctx: ServeContext) -> None:
        check_search(ctx.params)

        if self.variant in ("", "pool"):
            self.prestart(int(ctx.params.get("read_threads", READ_THREADS)))

        super().cold_setup(ctx)


def _dist_version(name: str) -> str | None:
    try:
        import importlib.metadata as metadata

        return metadata.version(name)
    except Exception:
        return None


if __name__ == "__main__":
    main(Paimon)
