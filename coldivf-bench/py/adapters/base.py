"""The base class of every system's adapter, and the adapter's end of the wire.

An adapter is a module in this folder that subclasses `Adapter` and ends with `main(MyAdapter)`. The driver runs it as
`<python> -m adapters.<name> <subcommand> ...` with this folder's parent on PYTHONPATH. Each subcommand reports through
JSON lines, one object per line with an "event" field, on the stdout the process started with. Commands from the
driver come as JSON lines on stdin. The base class moves everything else written to stdout, eg. a library's log, to
stderr, so it can't break the wire. README.md describes every subcommand, event and field.

An in-process adapter implements `open` and `search`, plus `build`, and the base class does the timing, the counters,
the cold protocol and the warm passes. An adapter that drives an outside program overrides `run_build`,
`cold_trial` and `warm_run` instead.

For the write benchmark an adapter also sets its role and fills in the write hooks, `writer_open`, `insert`, `delete`,
`sync`, `durable_point`, `maintain` and the rest, and the base class runs the `write` and `maintain` subcommands
around them. A hook a system lacks raises `Unsupported`, so a system with no write hook is rebuild-only.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import data  # noqa: E402
import evictor  # noqa: E402
import rusage  # noqa: E402
import schema  # noqa: E402
import writes  # noqa: E402

REGIMES = ("c0", "c1", "intent")
CPU_STATES = ("prewarm", "idle", "none")

# The roles of the write benchmark. "full" inserts and deletes through the system's own calls, "inserts" has no delete
# call, "rebuild" has no write call and takes the stream as a fresh build of the new vectors, and "exact" is the flat
# scan, whose writes the harness makes.
WRITE_ROLES = writes.ROLES
# How far a write gets: lost if the process dies ("memory"), safe from kill -9 but not from a power loss ("process"),
# or safe from a power loss too ("power").
LEVELS = writes.LEVELS
SYNC_KINDS = ("incremental", "whole")

# Environment variables that set the thread count of common numeric libraries.
THREAD_ENV = (
    "OMP_NUM_THREADS", "RAYON_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

PREWARM_S = 0.15
IDLE_S = 0.2


class Unsupported(Exception):
    """Raised by a hook for a subcommand, regime or setting the system doesn't offer. The adapter then reports
    "unsupported" and exits with status 0."""


def thread_env(threads: int) -> dict[str, str]:
    return {k: str(threads) for k in THREAD_ENV}


def now_ns() -> int:
    return time.perf_counter_ns()


def percentile(values: np.ndarray | list[float], p: float) -> float | None:
    """The nearest-rank percentile `p`, from 0 to 100. It is the smallest value that at least p% of the values are at
    or below."""
    v = np.sort(np.asarray(values, dtype=np.float64))

    if v.size == 0:
        return None

    rank = int(np.ceil(p * v.size / 100.0 - 1e-9))
    return float(v[min(max(rank, 1), v.size) - 1])


def latency_stats(times_us: np.ndarray) -> dict[str, float | None]:
    return {
        "p50_us": percentile(times_us, 50),
        "p90_us": percentile(times_us, 90),
        "p99_us": percentile(times_us, 99),
        "mean_us": float(np.mean(times_us)) if len(times_us) else None,
    }


def median_of_passes(passes: list[dict[str, float | None]]) -> dict[str, float | None]:
    """Each statistic's median over the passes."""
    out = {}

    for key in passes[0] if passes else {}:
        values = [p[key] for p in passes if p[key] is not None]
        out[key] = float(np.median(values)) if values else None

    return out


def spin(threads: int, seconds: float) -> None:
    """Keeps `threads` cores busy for `seconds`, the calling thread among them, so they run at full clock afterwards.
    Each thread hashes a buffer in a loop, which releases Python's global interpreter lock (GIL)."""
    block = b"\0" * 65536
    end = time.perf_counter() + seconds

    def work() -> None:
        rusage.set_interactive()

        while time.perf_counter() < end:
            hashlib.sha256(block).digest()

    helpers = [threading.Thread(target=work, daemon=True) for _ in range(max(0, threads - 1))]

    for t in helpers:
        t.start()

    work()

    for t in helpers:
        t.join()


def wait_until(deadline_ns: int, spin: bool) -> None:
    """Waits until `deadline_ns` on the perf_counter clock. Asleep it wakes 5 ms early and spins the rest, since a
    sleep on a busy machine can end a few milliseconds late."""
    while (left := deadline_ns - now_ns()) > 0:
        if not spin and left > 5_000_000:
            time.sleep((left - 5_000_000) / 1e9)


def cpu_state(state: str, threads: int) -> None:
    """The step every system runs right before its timer. "prewarm" spins as many cores as the system uses, "idle"
    sleeps, so the timed call starts on a core that wakes from idle, and "none" does nothing."""
    if state == "prewarm":
        spin(threads, PREWARM_S)
    elif state == "idle":
        time.sleep(IDLE_S)
    elif state != "none":
        raise ValueError(f"unknown cpu state {state}")


class Wire:
    """The adapter's end of the wire. Events go out on the stdout the process started with, and file descriptor 1 then
    points at stderr. Commands come in on stdin."""

    def __init__(self) -> None:
        sys.stdout.flush()
        self._out = os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8")
        os.dup2(2, 1)
        self._in = sys.stdin

    def send(self, event: str, **fields: Any) -> None:
        self._out.write(schema.dumps(schema.clean({"event": event, **fields})) + "\n")
        self._out.flush()

    def fileno(self) -> int:
        return self._out.fileno()

    def receive(self) -> dict | None:
        """The next command, or None at the end of stdin."""
        while True:
            raw = self._in.readline()

            if not raw:
                return None

            raw = raw.strip()

            if raw:
                return json.loads(raw)


@dataclass
class BuildContext:
    """What a build gets. `out` is an empty folder for the index. `data` is rows 0 to `rows` of the base, mapped, so a
    build reads it in chunks rather than copying it."""

    config: dict
    data_path: Path
    rows: int
    out: Path
    clusters: Path | None
    threads: int
    seed: int
    metric: str
    variant: str
    dataset: data.Dataset

    @property
    def data(self) -> np.ndarray:
        return self.dataset.prefix(self.rows)

    @property
    def dim(self) -> int:
        return self.dataset.dim


@dataclass
class ServeContext:
    """What a warm run or a cold trial gets. `params` holds the search settings, and `queries` the query vectors by
    query id. `warmup` holds the vectors of the queries a cold process searches before READY, which no trial of its cell
    times, and `warmup_ids` their ids."""

    index: Path
    dataset: data.Dataset
    rows: int
    k: int
    metric: str
    threads: int
    variant: str
    params: dict = field(default_factory=dict)
    queries: dict[int, np.ndarray] = field(default_factory=dict)
    regime: str = "c0"
    cpu_state: str = "prewarm"
    delay_ms: float = 0.0
    dummy: Path | None = None
    split: str = "test"
    mode: str = "whole"
    handle: Any = None
    warmup: list[np.ndarray] = field(default_factory=list)
    warmup_ids: list[int] = field(default_factory=list)
    # The base rows a warm run's first setting also searches as queries, for the findability check, or None.
    probes: np.ndarray | None = None


@dataclass
class WriteContext:
    """What a write session or a maintenance run gets. `index` is a clone of a state's folder, which the session may
    change. `rows` is the build's row count, so its ids are 0 to rows, and every insert's id is its row. `params` holds
    the write entry's settings. In the policy pass `policy` names the run kind it starts, at `maintain_threads` threads
    with `maintain_params`, the suite's maintenance settings."""

    index: Path
    dataset: data.Dataset
    rows: int
    metric: str
    threads: int
    variant: str
    mode: str = "stream"
    pass_name: str = "joint"
    params: dict = field(default_factory=dict)
    policy: str | None = None
    maintain_threads: int = 1
    maintain_params: dict = field(default_factory=dict)


def read_except(path: Path | str, skip: list[tuple[int, int]], chunk: int = 1 << 24) -> int:
    """Reads a file through with plain reads, apart from the byte spans (offset, end) in `skip`, and returns the bytes
    it read."""
    done = 0

    with open(path, "rb") as f:
        size = os.fstat(f.fileno()).st_size
        at = 0

        for start, end in sorted(skip) + [(size, size)]:
            while at < min(start, size):
                block = os.pread(f.fileno(), min(chunk, start - at), at)

                if not block:
                    return done

                at, done = at + len(block), done + len(block)

            at = max(at, end)

    return done


def logical_writes() -> int:
    """The bytes this process has asked to write, from proc_pid_rusage, whether or not they reached the disk yet."""
    raw = rusage.rusage_v6()
    return raw["logical_writes"] if raw else 0


class OpTimer:
    """The calls of one op of a write session, eg. its inserts. It times each call with the clock a search is timed
    with, and reads proc_pid_rusage and the machine's swap counts around each one, outside the timer, so the ops a
    session interleaves each get their own counters."""

    def __init__(self, op: str, batch: int, durability: str | None):
        self.op, self.batch, self.durability = op, batch, durability
        self.times: list[float] = []
        self.first_us: float | None = None
        self.rows = 0
        self.measured = False
        self.used = rusage.Delta()
        self.logical = 0
        self.vm = {"swapins": 0, "compressions": 0}
        self.footprint_before: int | None = None
        self.footprint_after: int | None = None
        self.unsupported: str | None = None
        self.extra: dict = {}
        # The clock at the start and the end of the last call, in nanoseconds.
        self.last = (0, 0)

    def call(self, fn: Any, rows: int = 0, timed: bool = True, keep: Any = None) -> Any:
        """Runs `fn`, one call of `rows` rows, and times it unless it is a warm-up call. `keep` may look at what the
        call returned and drop its sample, eg. a resize that had nothing to do. The first call counts as `first_us`,
        warm-up or not."""
        before, logical, vm = rusage.Usage.now(), logical_writes(), rusage.vm_counts()
        t0 = now_ns()
        out = fn()
        t1 = now_ns()
        after, logical_after, vm_after = rusage.Usage.now(), logical_writes(), rusage.vm_counts()
        self.last = (t0, t1)

        if keep is not None and not keep(out):
            return out

        if self.first_us is None:
            self.first_us = (t1 - t0) / 1e3

        if timed:
            self.add((t1 - t0) / 1e3, rows)
            self.measure(before, after, logical_after - logical, rusage.vm_delta(vm, vm_after))

        return out

    def add(self, us: float, rows: int) -> None:
        """One timed call of `rows` rows that took `us` microseconds."""
        if self.first_us is None:
            self.first_us = us

        self.times.append(us)
        self.rows += rows

    def measure(self, before: rusage.Usage, after: rusage.Usage, logical: int, vm: dict) -> None:
        if self.footprint_before is None:
            self.footprint_before = before.footprint

        self.footprint_after = after.footprint
        self.used = self.used + (after - before)
        self.logical += max(0, logical)
        self.measured = True

        for k in self.vm:
            self.vm[k] += vm.get(k, 0)

    @property
    def seconds(self) -> float:
        return float(np.sum(self.times)) / 1e6

    def fields(self) -> dict:
        """The op's write fields. Rows per second count only inserts and deletes, the calls that write rows."""
        t = np.asarray(self.times, dtype=np.float64)
        rate = self.rows / self.seconds if self.op in ("insert", "delete") and self.rows and self.seconds else None
        out = {
            "op": self.op, "batch": self.batch, "calls": len(t), "rows": self.rows, "durability": self.durability,
            "first_us": self.first_us, "max_us": float(t.max()) if len(t) else None, "rows_per_s": rate,
            **latency_stats(t),
        }

        if self.measured:
            out.update(cpu_ms=self.used.cpu_ms, cpu_qos_ms=self.used.qos_ms, logical_bytes_written=self.logical,
                       disk_bytes_written=self.used.bytes_written, footprint_before=self.footprint_before,
                       footprint_after=self.footprint_after,
                       lifetime_max_footprint=rusage.Usage.now().lifetime_max_footprint,
                       swapins_delta=self.vm["swapins"], compressions_delta=self.vm["compressions"])

        if self.unsupported is not None:
            out["unsupported"] = self.unsupported

        return out | {"extra": dict(self.extra)}


def maintain_fields(action: str, threads: int, params: dict, seconds: float, before: rusage.Usage,
                    after: rusage.Usage, report: dict | None) -> dict:
    """A maintenance run's fields from what the hook returned. Its "rows_live", "durability" and "data_file" go to
    their own places, and the rest of the library's report to extra."""
    report = dict(report or {})
    rows_live, durability, data_file = (report.pop(k, None) for k in ("rows_live", "durability", "data_file"))
    used = after - before
    return {
        "action": action, "threads": threads, "params": params, "seconds": seconds, "cpu_seconds": used.cpu_ns / 1e9,
        "footprint_before": before.footprint, "peak_footprint": after.lifetime_max_footprint,
        "bytes_read": used.bytes_read, "bytes_written": used.bytes_written, "rows_live": rows_live,
        "durability": durability, "extra": {"report": report, "data_file": str(data_file) if data_file else None},
    }


def check_ids(got: Any, ids: np.ndarray) -> None:
    """Raises unless a library that numbers its inserts itself gave them the stream's ids, i.e. their rows."""
    if got is None:
        return

    got = np.asarray(got, dtype=np.int64).ravel()

    if not np.array_equal(got, ids):
        raise RuntimeError(f"the library numbered the rows {int(ids[0])} to {int(ids[-1])} as {got[:3].tolist()} and "
                           f"on, not by their rows")


class Adapter:
    """One system. Subclasses set the class attributes and implement the hooks they need. Hooks raise `Unsupported`
    for what the system can't do."""

    # The system's name, as the results file and every line name it.
    system = ""
    # How it is called, eg. "python" for a Python binding or "cli" for an outside program.
    binding = "python"
    # The cold regimes it can run.
    regimes: tuple[str, ...] = REGIMES
    # Whether `search_batch` is a native batch call that releases Python's lock. Then QPS at several threads uses it.
    batch_api = False
    # Without a batch call, QPS at several threads runs "threads" in this process, or "processes", one per worker,
    # which suits mapped indexes searched by a per-query Python loop.
    qps_mode = "threads"
    # The threads the system runs besides the one that calls it, eg. "OpenMP's pool", or "" for none. Only the calling
    # thread is moved to the user-interactive QoS class, so these run at the class they start with, the default.
    workers = ""
    # Whether an open reads every file of the index into the system's own memory, so the default files() marks every
    # range "loaded".
    loads_at_open = False
    # The system's role in the write benchmark, one of WRITE_ROLES. A system without write hooks rebuilds.
    write_role = "rebuild"
    # The level a write call reaches by itself and the level `sync` reaches, and the level of the durable point where
    # it differs from the sync's, eg. {"call": "process", "sync": "power"}.
    durability: dict[str, str] = {"call": "memory", "sync": "power"}
    # "incremental" when a sync writes only what changed, so a stream syncs after every call and a latency run times a
    # sync after each of its durable inserts. "whole" when it rewrites the index, so a stream saves once at its end and
    # a latency run times `saves` durable points instead.
    sync_kind = "whole"
    # The actions `maintain` takes, eg. ("fold", "auto", "major", "job"). A compaction is one of them.
    maintenance_actions: tuple[str, ...] = ()

    def __init__(self) -> None:
        self.children: set[subprocess.Popen] = set()
        # The suite's variant label, set before `load`, so a hook can act on it.
        self.variant = ""
        self.wire: Wire | None = None

    def load(self) -> None:
        """Imports the library. It runs once per process, after the thread environment is set."""

    def version(self) -> str | None:
        """The library's version."""
        return None

    def wheels(self) -> dict[str, str | None]:
        """Hashes of the packages or binaries the system runs, eg. `schema.dist_sha256("faiss-cpu")`."""
        return {}

    def simd_level(self) -> str | None:
        """The SIMD (single instruction, multiple data) level the library uses, eg. "NEON"."""
        return None

    def thread_env(self, threads: int) -> dict[str, str]:
        """Extra environment that sets this library's threads, eg. its own variable. The base sets the common ones."""
        return {}

    def build(self, ctx: BuildContext) -> dict:
        """Builds an index of ctx.data into ctx.out, with id i for row i. It returns build fields it knows, eg.
        cluster_seconds, encode_seconds, stored_vector_bytes and exact_f32_stored, and an "extra" dict."""
        raise Unsupported(f"{self.system} has no build")

    def overhead_config(self) -> dict:
        """The build config of the 1K-vector index that measures the per-call cost."""
        return {}

    def overhead_params(self) -> dict:
        """The search settings of that measurement."""
        return {}

    def open(self, ctx: ServeContext, path: Path) -> Any:
        """Opens the index at `path` with ctx.params, and returns a handle. In a cold trial this is the first timed
        call, so it must be the first call that touches the index."""
        raise Unsupported(f"{self.system} has no open")

    def search(self, handle: Any, query: np.ndarray, k: int, params: dict) -> np.ndarray:
        """The ids of the k nearest rows to one query, nearest first, padded with -1."""
        raise Unsupported(f"{self.system} has no search")

    def search_batch(self, handle: Any, queries: np.ndarray, k: int, params: dict, threads: int) -> np.ndarray:
        """The ids for a batch of queries, one row each, through the library's own batch call on `threads` threads."""
        raise Unsupported(f"{self.system} has no batch search")

    def configure(self, handle: Any, params: dict) -> None:
        """Applies search settings to an open handle, eg. between the settings of a warm grid."""

    def close(self, handle: Any) -> None:
        """Releases a handle."""

    def warm_up(self, handle: Any, ctx: ServeContext) -> None:
        """Brings the index into memory for a warm run. Mode "whole" reads every file of the index by default, apart
        from the ranges files() marks "loaded", which the system holds in its own memory. Other modes, eg. "t1", are
        system-specific."""
        if ctx.mode != "whole":
            raise Unsupported(f"{self.system} has no warm mode {ctx.mode}")

        for spec in self.files(ctx.index):
            # A file can be listed before it exists, eg. a log the system writes on first use.
            if not os.path.isfile(spec["path"]):
                continue

            loaded = [(int(r["offset"]), int(r["offset"]) + int(r["length"])) for r in spec.get("ranges") or []
                      if r.get("loaded")]
            read_except(spec["path"], loaded)

    def stats(self, handle: Any) -> dict:
        """Counters of the last search, eg. dist_evals, bytes_scanned, bytes_requested or "stages", a dict of stage
        times. Empty by default."""
        return {}

    def self_evict(self, handle: Any) -> None:
        """Drops the cached pages of the index through the system's own mappings, for C1 where the driver's eviction
        doesn't reach them."""
        raise Unsupported(f"{self.system} can't evict its own mappings")

    def files(self, path: Path) -> list[dict]:
        """The index's files, each with its byte ranges marked "bulk" or "routing". Routing data is what every query
        reads and a system keeps in RAM in C1, eg. a coarse quantizer. A range the system copies into its own memory at
        open or at its first use, and never reads from the file again, also has "loaded" set, so its cached pages don't
        count as memory the system needs. By default every file is one bulk range."""
        path = Path(path)
        paths = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
        loaded = {"loaded": True} if self.loads_at_open else {}
        out = []

        for p in paths:
            size = p.stat().st_size
            out.append({"path": str(p), "bytes": size,
                        "ranges": [{"offset": 0, "length": size, "kind": "bulk"} | loaded]})

        return out

    def writer_open(self, ctx: WriteContext, path: Path) -> Any:
        """Opens the index at `path` for writes and returns a handle, eg. loading a faiss index whose searches map it.
        A system with no write call keeps this default, which makes it rebuild-only."""
        raise Unsupported(f"{self.system} has no write call, so it takes the stream as a fresh build")

    def insert(self, w: Any, ids: np.ndarray, rows: np.ndarray) -> np.ndarray | None:
        """One insert call of `rows`, one float32 row per id of `ids`, int64. The batch is len(ids), and a batch of 1
        uses the library's single-item call where it has one. A library that numbers its inserts itself returns the ids
        it gave them, which a stream session checks against `ids`. Any other returns None."""
        raise Unsupported(f"{self.system} has no insert")

    def delete(self, w: Any, ids: np.ndarray) -> None:
        """One delete call of `ids`. An inserts-only system keeps this default, so its latency runs record the deletes
        as unsupported and the driver gives it only the inserts-only ladder."""
        raise Unsupported(f"{self.system} has no delete")

    def sync(self, w: Any) -> dict | None:
        """Brings every change so far to the level durability["sync"] names, and returns what it did, eg. the files it
        synced, for the line's extra."""
        raise Unsupported(f"{self.system} has no sync")

    def durable_point(self, w: Any) -> dict | None:
        """The system's own end of a session, i.e. its sync, flush, save or close. A stream session times it once at
        its end, and a latency run times it `saves` times when `sync_kind` is "whole". It is the sync by default. One
        that closes the library, eg. AsterVec's close, opens it again into the same handle."""
        return self.sync(w)

    def resize(self, w: Any, capacity: int) -> bool:
        """Makes room for ids up to `capacity` before a round's inserts, for a library that needs it, eg.
        RaBitQ-Library's HNSW. It returns whether it did anything, and the session times it as an op of its own when it
        did. It does nothing by default."""
        return False

    def maintain(self, w: Any, action: str, threads: int, params: dict) -> dict | None:
        """Runs one action of `maintenance_actions` on `threads` threads with the suite's `params`, and returns what the
        library reports, eg. ColdIVF's RunReport as a dict. Its "rows_live", the vectors the index holds after,
        "durability", the level the maintained index is at, and "data_file", a file whose hash the driver compares
        between actions, go to the line's fields, and the rest to its extra."""
        raise Unsupported(f"{self.system} has no maintenance action {action}")

    def due(self, w: Any, params: dict) -> bool:
        """Whether the library's own policy calls for a maintenance run now, with the suite's `params`, for the policy
        pass, eg. ColdIVF's Status.due."""
        raise Unsupported(f"{self.system} has no maintenance policy")

    def pending(self, w: Any) -> dict:
        """The library's own counts of unmaintained changes, eg. ColdIVF's Status or LanceDB's fragments and versions,
        for the extra of a session's last line. Nothing by default."""
        return {}

    def writer_close(self, w: Any) -> None:
        """Closes the writer after its durable point."""

    def seal_state(self, ctx: WriteContext, path: Path) -> dict | None:
        """Leaves a state's folder as every later open will find it, once its session has closed. eg. a store that
        writes at every open takes one more open and close and then stores.py's seal, as its build does. It isn't
        timed, and does nothing by default."""
        return None

    def write_session(self, ctx: WriteContext, args: argparse.Namespace, wire: Wire) -> None:
        """Runs the `write` subcommand. It opens the writer, makes the session's calls through the hooks, and sends
        `opened`, an `op` event per op with its write fields, a `maintain` event per run of the policy pass, and
        `done`. A stream session ends with the durable point, closes the writer and seals the state. A latency run
        leaves its clone as it is, since the driver throws it away."""
        stream = writes.read_stream(args.stream)
        stream.check()
        rusage.set_interactive()
        gc.collect()
        gc.disable()
        start = rusage.Usage.now()
        t0 = now_ns()
        w = self.writer_open(ctx, ctx.index)
        opened = {"open_us": (now_ns() - t0) / 1e3, "open_footprint": rusage.Usage.now().footprint}
        wire.send("opened", pid=os.getpid(), footprint_before_open=start.footprint, **opened)
        before = writes.folder_bytes(ctx.index)

        if ctx.mode == "latency":
            ops, last = self._latency_calls(ctx, w, stream, args), None
            done: dict = {}
        else:
            ops, last, done = self._stream_calls(ctx, w, stream, args, wire)

        after = writes.folder_bytes(ctx.index)
        self.writer_close(w)
        # The calls change the folder from its size at the open to its size before the durable point, which the
        # durable point changes in turn.
        point = done.pop("bytes_before_point", after)
        sent = False

        for op in ops:
            if not op.times and op.unsupported is None:
                continue

            span = (point, after) if op is last else (before, point)
            fields = op.fields() | {"mode": ctx.mode, "threads": ctx.threads, "index_bytes_before": span[0],
                                    "index_bytes_after": span[1]}

            if not sent:
                fields.update(opened)
                sent = True

            wire.send("op", fields=fields)

        if ctx.mode == "stream":
            done["sealed"] = self.seal_state(ctx, ctx.index)

        wire.send("done", **done)

    def _latency_calls(self, ctx: WriteContext, w: Any, stream: writes.Stream,
                       args: argparse.Namespace) -> list[OpTimer]:
        """A latency run: single inserts of reserved rows and single deletes of base ids, each timed after warm-up
        calls, then the cost of making one write durable. That is an insert and a sync after it, timed together and the
        sync alone, where the sync writes only what changed, or else `saves` durable points."""
        plan = writes.latency_plan(stream, args.round, ctx.pass_name != "inserts", args.reserve, args.warmup,
                                   args.calls, args.durable_calls, args.seed)
        call, sync = self.durability.get("call"), self.durability.get("sync")
        grow = OpTimer("resize", 0, call)
        held = stream.base_rows + sum(len(r.inserts) for r in stream.rounds[: args.round])
        grow.call(lambda: self.resize(w, held + len(plan.inserts)), keep=bool)
        ins, dels = OpTimer("insert", 1, call), OpTimer("delete", 1, call)

        for i in range(plan.warmup + plan.calls):
            ids = plan.inserts[i : i + 1]
            rows = writes.vectors(ctx.dataset, ids)
            ins.call(lambda: self.insert(w, ids, rows), 1, timed=i >= plan.warmup)

        try:
            for i in range(len(plan.deletes)):
                ids = plan.deletes[i : i + 1]
                dels.call(lambda: self.delete(w, ids), 1, timed=i >= plan.warmup)
        except Unsupported as e:
            dels.unsupported = str(e)

        if self.sync_kind != "incremental":
            point = OpTimer("durable_point", 0, self.durability.get("durable_point", sync))

            for _ in range(args.saves):
                point.call(lambda: self.durable_point(w))

            return [grow, ins, dels, point]

        # Each durable insert times the insert and the sync after it as one call, with the counters of both, and the
        # sync alone apart, from a clock read between them.
        both, alone = OpTimer("insert_sync", 1, sync), OpTimer("sync", 1, sync)
        marks = [0]

        def pair(ids: np.ndarray, rows: np.ndarray) -> Any:
            self.insert(w, ids, rows)
            marks[0] = now_ns()
            return self.sync(w)

        for i in range(plan.durable):
            at = plan.warmup + plan.calls + i
            ids = plan.inserts[at : at + 1]
            rows = writes.vectors(ctx.dataset, ids)
            both.call(lambda: pair(ids, rows), 1)
            alone.add((both.last[1] - marks[0]) / 1e3, 1)

        return [grow, ins, dels, both, alone]

    def _stream_calls(self, ctx: WriteContext, w: Any, stream: writes.Stream, args: argparse.Namespace,
                      wire: Wire) -> tuple[list[OpTimer], OpTimer, dict]:
        """A stream session: the calls of rounds --from-round + 1 to --round in batches, a sync after each call where
        the sync writes only what changed, in the policy pass a maintenance run whenever one is due, and the durable
        point at the end. Each insert passes the next ids and the vectors of the stream's rows. It returns the ops, the
        durable point among them, and what the done event carries."""
        rounds = writes.stream_calls(stream, ctx.pass_name, args.from_round, args.round, args.batch)
        call, sync = self.durability.get("call"), self.durability.get("sync")
        ins, dels = OpTimer("insert", args.batch, call), OpTimer("delete", args.batch, call)
        syncs, grow = OpTimer("sync", args.batch, sync), OpTimer("resize", 0, call)
        capacity = stream.base_rows + sum(len(r.inserts) for r in stream.rounds[: args.from_round])
        runs, status_us, maintain_s = 0, 0.0, 0.0

        for calls in rounds:
            capacity += sum(len(c.ids) for c in calls if c.op == "insert")
            grow.call(lambda: self.resize(w, capacity), keep=bool)

            for j, c in enumerate(calls):
                if c.op == "insert":
                    rows = writes.vectors(ctx.dataset, c.rows)
                    check_ids(ins.call(lambda: self.insert(w, c.ids, rows), len(c.ids)), c.ids)
                else:
                    dels.call(lambda: self.delete(w, c.ids), len(c.ids))

                if self.sync_kind == "incremental":
                    syncs.call(lambda: self.sync(w), len(c.ids))

                if ctx.policy:
                    a = now_ns()
                    due = self.due(w, ctx.maintain_params)
                    status_us += (now_ns() - a) / 1e3

                    if due:
                        runs += 1
                        fields = self._policy_run(ctx, w, runs, c.round, j)
                        maintain_s += fields["seconds"]
                        wire.send("maintain", fields=fields)

        before_point = writes.folder_bytes(ctx.index)
        pending = self.pending(w)
        point = OpTimer("durable_point", args.batch, self.durability.get("durable_point", sync))
        rows = ins.rows + dels.rows
        point.call(lambda: self.durable_point(w), rows)
        busy = ins.seconds + dels.seconds + syncs.seconds + grow.seconds
        point.extra.update(
            pending=pending, rows=rows, runs=runs, status_us=status_us, maintain_s=maintain_s,
            rows_per_s=rows / busy if rows and busy else None,
            rows_per_s_durable=rows / (busy + point.seconds) if rows else None,
            rows_per_s_with_runs=rows / (busy + point.seconds + maintain_s) if rows and ctx.policy else None,
        )
        return [grow, ins, dels, syncs, point], point, {"bytes_before_point": before_point, "runs": runs}

    def _policy_run(self, ctx: WriteContext, w: Any, run: int, round_: int, call: int) -> dict:
        """One maintenance run of the policy pass, made at once when the library's policy calls for it."""
        pending = self.pending(w)
        before, start = rusage.Usage.now(), time.perf_counter()
        report = self.maintain(w, ctx.policy, ctx.maintain_threads, ctx.maintain_params)
        seconds, after = time.perf_counter() - start, rusage.Usage.now()
        fields = maintain_fields(ctx.policy, ctx.maintain_threads, ctx.maintain_params, seconds, before, after, report)
        fields["extra"].update(run=run, round=round_, after_call=call, pending_before=pending)
        return fields

    def maintain_session(self, ctx: WriteContext, action: str, params: dict, wire: Wire) -> None:
        """Runs the `maintain` subcommand: one action on a clone of a state, in a fresh process, so its lifetime peak
        footprint is the action's. It then closes the writer, seals the state and sends `maintain`."""
        if action not in self.maintenance_actions:
            raise Unsupported(f"{self.system} has no maintenance action {action}")

        t0 = now_ns()
        w = self.writer_open(ctx, ctx.index)
        open_us = (now_ns() - t0) / 1e3
        before, start = rusage.Usage.settled(), time.perf_counter()
        report = self.maintain(w, action, ctx.threads, params)
        seconds, after = time.perf_counter() - start, rusage.Usage.settled()
        pending = self.pending(w)
        self.writer_close(w)
        fields = maintain_fields(action, ctx.threads, params, seconds, before, after, report)
        fields["open_us"] = open_us
        fields["extra"].update(pending=pending, sealed=self.seal_state(ctx, ctx.index))
        wire.send("maintain", fields=fields)

    def run_child(self, cmd: list[str], env: dict | None = None, timeout: float | None = None,
                  capture: bool = True) -> subprocess.CompletedProcess:
        """Runs an outside program. Its stderr goes to this process's stderr, and its stdout is captured. A SIGTERM to
        this process ends it too."""
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE if capture else sys.stderr, stderr=sys.stderr,
                                stdin=subprocess.DEVNULL)
        self.children.add(proc)

        try:
            out, _ = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            raise
        finally:
            self.children.discard(proc)

        return subprocess.CompletedProcess(cmd, proc.returncode, out or b"", b"")

    def run_measured(self, cmd: list[str], env: dict | None = None) -> tuple[subprocess.CompletedProcess, rusage.Usage]:
        """Runs an outside program with its stdout on this process's stderr, as run_child does without capture, and
        also returns the program's counters as it exited, eg. its lifetime peak footprint."""
        proc = subprocess.Popen(cmd, env=env, stdout=sys.stderr, stderr=sys.stderr, stdin=subprocess.DEVNULL)
        self.children.add(proc)

        try:
            used = rusage.Usage.at_exit(proc.pid)
            proc.wait()
        except BaseException:
            proc.kill()
            proc.wait()
            raise
        finally:
            self.children.discard(proc)

        return subprocess.CompletedProcess(cmd, proc.returncode, b"", b""), used

    def exec_program(self, cmd: list[str], env: dict | None = None) -> None:
        """Replaces this process with a program that speaks the wire itself, on the adapter's stdin and its original
        stdout. The program keeps this pid and process group. Call it from `cold_setup`, before READY."""
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(self.wire.fileno(), 1)
        os.execve(cmd[0], cmd, env if env is not None else dict(os.environ))

    def _terminate(self, signum: int, frame: Any) -> None:
        for proc in list(self.children):
            try:
                proc.kill()
            except OSError:
                pass

        os._exit(128 + signum)

    def run_build(self, ctx: BuildContext) -> dict:
        """Runs `build` and measures it. Fields `build` returns win over the measured ones."""
        before = rusage.Usage.settled()
        start = time.perf_counter()
        fields = self.build(ctx) or {}
        seconds = time.perf_counter() - start
        after = rusage.Usage.settled()
        used = after - before
        out = {
            "seconds": seconds,
            "cpu_seconds": used.cpu_ns / 1e9,
            "threads": ctx.threads,
            "seed": ctx.seed,
            "peak_footprint": after.lifetime_max_footprint,
            "bytes_read": used.bytes_read,
            "bytes_written": used.bytes_written,
        }
        out.update(fields)
        return out

    def warmup_query(self, ctx: ServeContext) -> np.ndarray:
        """The query a cold process searches before READY. The driver picks one that no trial of the cell searches, so
        whatever the warm-up leaves in a cache can't answer a timed query. Without one it is the cell's first query."""
        return ctx.warmup[0] if ctx.warmup else next(iter(ctx.queries.values()))

    def cold_setup(self, ctx: ServeContext) -> None:
        """Runs before READY. C0 and intent search the dummy index once, so code and thread pools are warm. C1 opens
        the index and runs one warm-up query, and keeps the handle."""
        first = self.warmup_query(ctx)

        if ctx.regime == "c1":
            ctx.handle = self.open(ctx, ctx.index)
            self.search(ctx.handle, first, ctx.k, ctx.params)
        elif ctx.dummy is not None:
            handle = self.open(ctx, ctx.dummy)
            self.search(handle, first, ctx.k, ctx.params)
            self.close(handle)

    def cold_teardown(self, ctx: ServeContext) -> None:
        if ctx.handle is not None:
            self.close(ctx.handle)
            ctx.handle = None

    def cold_trial(self, ctx: ServeContext, go: dict) -> dict:
        """One timed trial for a GO command. It returns the trial's cold fields."""
        if go.get("self_evict"):
            self.self_evict(ctx.handle)

        queries = [ctx.queries[int(q)] for q in go["queries"]]
        first, rest = queries[0], queries[1:]
        cpu_state(ctx.cpu_state, ctx.threads)
        before = rusage.Usage.settled()
        out: dict[str, Any] = {}

        if ctx.regime == "c1":
            handle = ctx.handle
            t0 = now_ns()
            ids = self.search(handle, first, ctx.k, ctx.params)
            t2 = now_ns()
            out.update(open_us=None, search_us=(t2 - t0) / 1e3, total_us=(t2 - t0) / 1e3)
        elif ctx.regime == "intent":
            # The query text arrives at t0, and the system may open the index at once. The vector is ready delay_ms
            # later, and the timer runs from then until the ids are in hand.
            t0 = now_ns()
            handle = self.open(ctx, ctx.index)
            t1 = now_ns()
            ready = t0 + int(ctx.delay_ms * 1e6)
            wait_until(ready, spin=ctx.cpu_state == "prewarm")
            t_search = now_ns()
            ids = self.search(handle, first, ctx.k, ctx.params)
            t2 = now_ns()
            out.update(open_us=(t1 - t0) / 1e3, search_us=(t2 - t_search) / 1e3, total_us=(t2 - ready) / 1e3)
        else:
            t0 = now_ns()
            handle = self.open(ctx, ctx.index)
            t1 = now_ns()
            ids = self.search(handle, first, ctx.k, ctx.params)
            t2 = now_ns()
            out.update(open_us=(t1 - t0) / 1e3, search_us=(t2 - t1) / 1e3, total_us=(t2 - t0) / 1e3)

        after = rusage.Usage.settled()
        stats = self.stats(handle) or {}
        next_us = []

        for q in rest:
            a = now_ns()
            self.search(handle, q, ctx.k, ctx.params)
            next_us.append((now_ns() - a) / 1e3)

        rerun, rerun_extra = self.resident_rerun(ctx, handle, first)

        if ctx.regime != "c1":
            self.close(handle)

        used = after - before
        out.update(
            next_us=next_us,
            resident_rerun_us=rerun,
            ids=[int(i) for i in np.asarray(ids).ravel()[: ctx.k]],
            stages=stats.get("stages"),
            bytes_requested=stats.get("bytes_requested"),
            disk_bytes=used.bytes_read,
            pageins=used.pageins,
            footprint_before=before.footprint,
            footprint_delta=used.footprint,
            lifetime_max_footprint=after.lifetime_max_footprint,
            cpu_ms=used.cpu_ms,
            cpu_qos_ms=used.qos_ms,
            p_share=used.p_share,
            ghz=used.ghz,
            energy_nj=used.energy_nj,
            qos=rusage.qos_name(),
            extra={k: v for k, v in stats.items() if k not in ("stages", "bytes_requested")} | rerun_extra,
        )
        return out

    def resident_rerun(self, ctx: ServeContext, handle: Any, query: np.ndarray) -> tuple[float | None, dict]:
        """Times the trial's first query again after the others, with the pages it read in memory, for the split of a
        cold time into compute and waiting on reads. It returns the time in microseconds and anything to add to the
        trial's extra. A system whose reads skip the page cache overrides it."""
        a = now_ns()
        self.search(handle, query, ctx.k, ctx.params)
        return (now_ns() - a) / 1e3, {}

    def warm_run(self, ctx: ServeContext, grid: list[dict], qids: np.ndarray, ids_dir: Path, passes: int,
                 warmup: int, qps_seconds: float, qps_repeats: int) -> Iterator[dict]:
        """Runs every setting of `grid` on one open index and yields one result per setting. A result holds "params",
        the warm "fields" and, at 1 thread, "ids_file", an .npy of the first pass's ids, one row per query of qids. With
        ctx.probes the first setting's result also holds "probes_file", the ids of a search for each probe row."""
        queries = np.stack([ctx.queries[int(q)] for q in qids])
        all_queries = list(ctx.queries.values())
        ctx.params = dict(grid[0]) if grid else {}
        before_open = rusage.Usage.now().footprint
        handle = self.open(ctx, ctx.index)
        self.warm_up(handle, ctx)

        try:
            for i, params in enumerate(grid):
                self.configure(handle, params)

                for j in range(warmup):
                    self.search(handle, all_queries[j % len(all_queries)], ctx.k, params)

                # What the system copied into its own memory needs no second copy in the page cache while it is timed.
                dropped = self._drop_loaded(ctx)

                if ctx.threads <= 1:
                    fields, ids = self._latency(handle, ctx, params, queries, passes)
                    path = ids_dir / f"setting-{i}.npy"
                    np.save(path, ids)
                    result = {"params": params, "fields": fields, "ids_file": str(path)}
                else:
                    fields = self._qps(handle, ctx, params, queries, qps_seconds, qps_repeats)
                    result = {"params": params, "fields": fields, "ids_file": None}

                if i == 0 and ctx.probes is not None:
                    result["probes_file"] = str(self.probe(handle, ctx, params, ids_dir))

                # A run with worker processes reports its workers' figures instead.
                fields.setdefault("footprint_before_open", before_open)
                fields.setdefault("footprint_since_open", fields["footprint"] - before_open
                                  if fields.get("footprint") is not None else None)
                fields["loaded_bytes_dropped"] = dropped
                yield result
        finally:
            self.close(handle)

    def probe(self, handle: Any, ctx: ServeContext, params: dict, ids_dir: Path) -> Path:
        """Searches each row of ctx.probes as a query with `params`, for the findability check, after the timed passes,
        and saves the ids as probes.npy in `ids_dir`, one row per probe. An adapter that overrides warm_run calls it
        too, or leaves the probes out."""
        found = np.full((len(ctx.probes), ctx.k), -1, dtype=np.int64)

        for j, row in enumerate(np.asarray(ctx.probes).tolist()):
            query = np.asarray(ctx.dataset.base[row], dtype=np.float32)
            ids = np.asarray(self.search(handle, query, ctx.k, params)).ravel()[: ctx.k]
            found[j, : len(ids)] = ids

        path = ids_dir / "probes.npy"
        np.save(path, found)
        return path

    def _drop_loaded(self, ctx: ServeContext) -> int | None:
        """Drops the cached pages of the ranges files() marks "loaded", and returns the bytes that were cached there, or
        None when that failed."""
        try:
            return evictor.drop_loaded(self.files(ctx.index))
        except OSError:
            return None

    def _latency(self, handle: Any, ctx: ServeContext, params: dict, queries: np.ndarray,
                 passes: int) -> tuple[dict, np.ndarray]:
        ids = np.full((len(queries), ctx.k), -1, dtype=np.int64)
        evals: dict[str, list[float]] = {"dist_evals": [], "bytes_scanned": []}
        vm_before, before = rusage.vm_counts(), rusage.Usage.settled()
        stats = []

        for p in range(passes):
            times = np.empty(len(queries))

            for j, q in enumerate(queries):
                a = now_ns()
                found = self.search(handle, q, ctx.k, params)
                times[j] = (now_ns() - a) / 1e3

                if p == 0:
                    found = np.asarray(found).ravel()[: ctx.k]
                    ids[j, : len(found)] = found
                    counters = self.stats(handle) or {}

                    for key in evals:
                        if counters.get(key) is not None:
                            evals[key].append(float(counters[key]))

            stats.append(latency_stats(times))

        after, vm_after = rusage.Usage.settled(), rusage.vm_counts()
        used, vm = after - before, rusage.vm_delta(vm_before, vm_after)
        fields = median_of_passes(stats)
        fields.update(
            queries=len(queries),
            passes=passes,
            batch_api=False,
            processes=1,
            qps=1e6 / fields["mean_us"] if fields.get("mean_us") else None,
            cpu_ms_per_query=used.cpu_ms / max(1, len(queries) * passes),
            footprint=after.footprint,
            swapins_delta=vm["swapins"],
            compressions_delta=vm["compressions"],
            dist_evals=float(np.mean(evals["dist_evals"])) if evals["dist_evals"] else None,
            bytes_scanned=float(np.mean(evals["bytes_scanned"])) if evals["bytes_scanned"] else None,
        )
        fields.update(self._memory(ctx, after.footprint))
        return fields, ids

    def _qps(self, handle: Any, ctx: ServeContext, params: dict, queries: np.ndarray, seconds: float,
             repeats: int) -> dict:
        processes = not self.batch_api and self.qps_mode == "processes"
        vm_before, before = rusage.vm_counts(), rusage.Usage.settled()
        rates, total, runs = [], 0, []

        for _ in range(repeats):
            if self.batch_api:
                count, elapsed = self._qps_batch(handle, ctx, params, queries, seconds)
            elif processes:
                run = self._qps_processes(ctx, params, seconds)
                count, elapsed = run["queries"], run["seconds"]
                runs.append(run)
            else:
                count, elapsed = self._qps_threads(handle, ctx, params, queries, seconds)

            rates.append(count / elapsed if elapsed > 0 else 0.0)
            total += count

        after, vm_after = rusage.Usage.settled(), rusage.vm_counts()
        used, vm = after - before, rusage.vm_delta(vm_before, vm_after)
        fields = {
            "queries": len(queries),
            "passes": repeats,
            "batch_api": self.batch_api,
            "processes": ctx.threads if processes else 1,
            "qps": float(np.median(rates)) if rates else None,
            "cpu_ms_per_query": used.cpu_ms / max(1, total),
            "footprint": after.footprint,
            "swapins_delta": vm["swapins"],
            "compressions_delta": vm["compressions"],
        }

        if runs:
            # The workers did the work, each in a process of its own, so their CPU time and memory are what count.
            # This process only waited, holding an index it didn't search.
            busiest = max(runs, key=lambda r: r["footprint"])
            fields.update(cpu_ms_per_query=sum(r["cpu_ns"] for r in runs) / 1e6 / max(1, total),
                          footprint=busiest["footprint"], footprint_before_open=busiest["footprint_before_open"],
                          footprint_since_open=busiest["footprint"] - busiest["footprint_before_open"],
                          worker_footprints=busiest["footprints"], parent_footprint=after.footprint)

        fields.update(self._memory(ctx, fields["footprint"]))
        return fields

    def _qps_batch(self, handle: Any, ctx: ServeContext, params: dict, queries: np.ndarray,
                   seconds: float) -> tuple[int, float]:
        start, count = time.perf_counter(), 0

        while time.perf_counter() - start < seconds:
            self.search_batch(handle, queries, ctx.k, params, ctx.threads)
            count += len(queries)

        return count, time.perf_counter() - start

    def _qps_threads(self, handle: Any, ctx: ServeContext, params: dict, queries: np.ndarray,
                     seconds: float) -> tuple[int, float]:
        start = time.perf_counter()
        end = start + seconds

        def worker(offset: int) -> int:
            rusage.set_interactive()
            count, j = 0, offset

            while time.perf_counter() < end:
                self.search(handle, queries[j % len(queries)], ctx.k, params)
                count, j = count + 1, j + 1

            return count

        offsets = [i * len(queries) // ctx.threads for i in range(ctx.threads)]

        with ThreadPoolExecutor(ctx.threads) as pool:
            counts = list(pool.map(worker, offsets))

        return sum(counts), time.perf_counter() - start

    def _qps_processes(self, ctx: ServeContext, params: dict, seconds: float) -> dict:
        """One QPS run on one worker process per thread. It returns the queries the workers ran, the longest of their
        timed loops in seconds, the CPU time of those loops in nanoseconds, and the sum of their footprints at the end
        of their loops and before they opened the index, with each worker's footprint."""
        module = type(self).__module__
        spec = getattr(sys.modules.get("__main__"), "__spec__", None)

        # Run with -m, the adapter's module is __main__, and its real name is in the spec.
        if module == "__main__" and spec is not None:
            module = spec.name
        cmd = [sys.executable, "-m", module, "qps-worker", "--index", str(ctx.index), "--data",
               str(ctx.dataset.base_path), "--rows", str(ctx.rows), "--k", str(ctx.k), "--metric", ctx.metric,
               "--variant", ctx.variant, "--split", ctx.split, "--params", json.dumps(params), "--seconds",
               str(seconds), "--threads", "1"]
        env = os.environ | thread_env(1)
        workers = [subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env)
                   for _ in range(ctx.threads)]

        for w in workers:
            self.children.add(w)

        try:
            for w in workers:
                _expect(w, "ready")

            for w in workers:
                w.stdin.write(json.dumps({"cmd": "go"}) + "\n")
                w.stdin.flush()

            counts = [_expect(w, "count") for w in workers]
        finally:
            for w in workers:
                w.wait(timeout=60)
                self.children.discard(w)

        return {
            "queries": sum(c["queries"] for c in counts),
            "seconds": max(c["seconds"] for c in counts),
            "cpu_ns": sum(int(c.get("cpu_ns") or 0) for c in counts),
            "footprint": sum(int(c.get("footprint") or 0) for c in counts),
            "footprint_before_open": sum(int(c.get("footprint_before_open") or 0) for c in counts),
            "footprints": [c.get("footprint") for c in counts],
        }

    def _memory(self, ctx: ServeContext, footprint: int | None) -> dict:
        """The resident bytes of the index's files, apart from the ranges files() marks "loaded", which the system
        holds in its own memory and so counts in its footprint already. The cached bytes of those ranges go apart."""
        try:
            resident, loaded = evictor.resident_split(self.files(ctx.index))
        except OSError:
            resident = loaded = None

        return {
            "resident_file_bytes": resident,
            "memory_total": footprint + resident if resident is not None and footprint is not None else None,
            "resident_loaded_bytes": loaded,
        }

    def qps_worker(self, ctx: ServeContext, wire: Wire, seconds: float) -> None:
        before_open = rusage.Usage.now().footprint
        handle = self.open(ctx, ctx.index)
        queries = list(ctx.queries.values())
        self.search(handle, queries[0], ctx.k, ctx.params)
        wire.send("ready", pid=os.getpid())
        wire.receive()
        rusage.set_interactive()
        before = rusage.Usage.settled()
        start, count = time.perf_counter(), 0

        while time.perf_counter() - start < seconds:
            self.search(handle, queries[count % len(queries)], ctx.k, ctx.params)
            count += 1

        elapsed = time.perf_counter() - start
        after = rusage.Usage.settled()
        wire.send("count", queries=count, seconds=elapsed, cpu_ns=(after - before).cpu_ns, footprint=after.footprint,
                  footprint_before_open=before_open, lifetime_max_footprint=after.lifetime_max_footprint)
        self.close(handle)

    def run_overhead(self, ctx: BuildContext, ids: int = 1000, calls: int = 1000) -> dict:
        """The per-call cost of a search on an index of the base's first `ids` rows."""
        with tempfile.TemporaryDirectory(prefix=f"overhead-{self.system}-") as tmp:
            ctx.out, ctx.rows, ctx.config = Path(tmp), min(ids, ctx.dataset.count), self.overhead_config()
            self.build(ctx)
            serve = ServeContext(index=ctx.out, dataset=ctx.dataset, rows=ctx.rows, k=10, metric=ctx.metric,
                                 threads=1, variant=ctx.variant, params=self.overhead_params())
            handle = self.open(serve, ctx.out)
            q = np.asarray(ctx.dataset.queries[0], dtype=np.float32)

            for _ in range(min(calls, 100)):
                self.search(handle, q, 10, serve.params)

            times = np.empty(calls)

            for i in range(calls):
                a = now_ns()
                self.search(handle, q, 10, serve.params)
                times[i] = (now_ns() - a) / 1e3

            self.close(handle)

        return {"overhead_us": percentile(times, 50), "calls": calls, "index_rows": ctx.rows}

    def supported_regimes(self) -> tuple[str, ...]:
        """The cold regimes this system runs. It is `regimes` by default, and a hook may narrow it by variant."""
        return self.regimes

    def info(self) -> dict:
        return {
            "system": self.system,
            "system_version": self.version(),
            "binding": self.binding,
            "simd_level": self.simd_level(),
            "regimes": list(self.supported_regimes()),
            "batch_api": self.batch_api,
            "qps_mode": self.qps_mode,
            "workers": self.workers,
            "worker_qos": "default",
            "wheel_sha256": self.wheels(),
            "venv": sys.prefix,
            "python": sys.executable,
            "write_role": self.write_role,
            "durability": dict(self.durability),
            "sync_kind": self.sync_kind,
            "maintenance_actions": list(self.maintenance_actions),
        }


def _expect(proc: subprocess.Popen, event: str) -> dict:
    for raw in proc.stdout:
        try:
            msg = json.loads(raw)
        except ValueError:
            continue

        if msg.get("event") == event:
            return msg

        if msg.get("event") == "error":
            raise RuntimeError(msg.get("message"))

    raise RuntimeError(f"a worker ended before sending {event}")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="adapter")
    sub = p.add_subparsers(dest="command", required=True)

    def common(s: argparse.ArgumentParser) -> None:
        s.add_argument("--data", required=True, help="the base .fbin")
        s.add_argument("--rows", type=int, required=True, help="the prefix of the base, rows 0 to N")
        s.add_argument("--metric", default="")
        s.add_argument("--threads", type=int, default=1)
        s.add_argument("--variant", default="")
        s.add_argument("--k", type=int, default=10)

    b = sub.add_parser("build")
    common(b)
    b.add_argument("--config", default="{}")
    b.add_argument("--out", required=True)
    b.add_argument("--clusters", default=None)
    b.add_argument("--seed", type=int, default=1)

    w = sub.add_parser("warm")
    common(w)
    w.add_argument("--index", required=True)
    w.add_argument("--grid", required=True, help="a JSON list of search settings")
    w.add_argument("--split", default="tune")
    w.add_argument("--mode", default="whole")
    w.add_argument("--ids-dir", required=True)
    w.add_argument("--passes", type=int, default=3)
    w.add_argument("--warmup", type=int, default=1000)
    w.add_argument("--qps-seconds", type=float, default=10.0)
    w.add_argument("--qps-repeats", type=int, default=3)
    w.add_argument("--probes", default=None, help="an .npy of base rows to search as queries, for findability")

    c = sub.add_parser("cold")
    common(c)
    c.add_argument("--index", required=True)
    c.add_argument("--dummy", default=None)
    c.add_argument("--params", default="{}")
    c.add_argument("--queries", required=True, help="comma-separated query ids")
    c.add_argument("--regime", default="c0", choices=REGIMES)
    c.add_argument("--cpu-state", default="prewarm", choices=CPU_STATES)
    c.add_argument("--delay-ms", type=float, default=0.0)
    c.add_argument("--split", default="test")
    c.add_argument("--warmup-queries", default="", help="comma-separated query ids for the searches before READY")

    f = sub.add_parser("files")
    f.add_argument("--index", required=True)
    f.add_argument("--variant", default="")

    o = sub.add_parser("overhead")
    common(o)

    i = sub.add_parser("info")
    i.add_argument("--variant", default="")

    q = sub.add_parser("qps-worker")
    common(q)
    q.add_argument("--index", required=True)
    q.add_argument("--params", default="{}")
    q.add_argument("--split", default="tune")
    q.add_argument("--seconds", type=float, default=10.0)

    wr = sub.add_parser("write")
    common(wr)
    wr.add_argument("--index", required=True, help="a clone of the state the session starts from")
    wr.add_argument("--stream", required=True, help="the stream file")
    wr.add_argument("--mode", default="stream", choices=writes.MODES)
    wr.add_argument("--pass", dest="pass_name", default="joint", choices=writes.PASSES)
    wr.add_argument("--round", type=int, required=True,
                    help="the round a stream session ends at, or the round of a latency run's state")
    wr.add_argument("--from-round", type=int, default=0, help="the round of the state a stream session starts from")
    wr.add_argument("--batch", type=int, default=1000, help="rows per call of a stream session")
    wr.add_argument("--calls", type=int, default=1000, help="timed calls of each op of a latency run")
    wr.add_argument("--warmup", type=int, default=100, help="untimed calls before them")
    wr.add_argument("--durable-calls", type=int, default=200, help="inserts each followed by a sync")
    wr.add_argument("--saves", type=int, default=5, help="durable points timed where a sync rewrites the index")
    wr.add_argument("--reserve", type=int, default=0, help="the first reserved row a latency run inserts")
    wr.add_argument("--seed", type=int, default=2, help="the seed of a latency run's deletes")
    wr.add_argument("--params", default="{}", help="the write entry's settings, as JSON")
    wr.add_argument("--policy", default=None, help="the run kind of the policy pass")
    wr.add_argument("--maintain-params", default="{}")
    wr.add_argument("--maintain-threads", type=int, default=1)

    m = sub.add_parser("maintain")
    common(m)
    m.add_argument("--index", required=True, help="a clone of the state the action maintains")
    m.add_argument("--action", required=True)
    m.add_argument("--params", default="{}")
    return p


def _queries(dataset: data.Dataset, ids: list[int]) -> dict[int, np.ndarray]:
    return {int(q): np.array(dataset.queries[int(q)], dtype=np.float32) for q in ids}


def _serve(args: argparse.Namespace, dataset: data.Dataset) -> ServeContext:
    return ServeContext(index=Path(args.index), dataset=dataset, rows=args.rows, k=args.k,
                        metric=args.metric or dataset.metric, threads=args.threads, variant=args.variant)


def _cold(adapter: Adapter, args: argparse.Namespace, wire: Wire, dataset: data.Dataset) -> None:
    if args.regime not in adapter.supported_regimes():
        raise Unsupported(f"{adapter.system} has no {args.regime} regime")

    ctx = _serve(args, dataset)
    ctx.params = json.loads(args.params)
    ctx.regime, ctx.cpu_state, ctx.delay_ms, ctx.split = args.regime, args.cpu_state, args.delay_ms, args.split
    ctx.dummy = Path(args.dummy) if args.dummy else None
    ctx.queries = _queries(dataset, [int(q) for q in args.queries.split(",") if q.strip()])
    ctx.warmup_ids = [int(q) for q in args.warmup_queries.split(",") if q.strip()]
    ctx.warmup = list(_queries(dataset, ctx.warmup_ids).values())
    rusage.set_interactive()
    adapter.cold_setup(ctx)
    gc.collect()
    gc.disable()
    wire.send("ready", pid=os.getpid(), regime=ctx.regime, **adapter.info())

    try:
        while (cmd := wire.receive()) is not None and cmd.get("cmd") == "go":
            fields = adapter.cold_trial(ctx, cmd)
            wire.send("trial", trial=cmd.get("trial"), fields=fields)

            if ctx.regime != "c1":
                break
    finally:
        adapter.cold_teardown(ctx)


def _warm(adapter: Adapter, args: argparse.Namespace, wire: Wire, dataset: data.Dataset) -> None:
    ctx = _serve(args, dataset)
    ctx.split, ctx.mode = args.split, args.mode
    grid = json.loads(args.grid)
    qids = data.split_ids(dataset.queries.shape[0], args.split)
    ctx.queries = _queries(dataset, sorted(set(qids.tolist()) | set(range(min(args.warmup, len(dataset.queries))))))
    ids_dir = Path(args.ids_dir)
    ids_dir.mkdir(parents=True, exist_ok=True)
    ctx.probes = np.load(args.probes) if args.probes else None
    rusage.set_interactive()

    for i, result in enumerate(adapter.warm_run(ctx, grid, qids, ids_dir, args.passes, args.warmup,
                                                args.qps_seconds, args.qps_repeats)):
        fields = dict(result["fields"])
        fields.setdefault("mode", ctx.mode)
        fields.setdefault("threads", ctx.threads)
        fields.setdefault("k", ctx.k)
        wire.send("setting", index=i, params=result["params"], fields=fields, ids_file=result.get("ids_file"),
                  qids=None if result.get("ids_file") is None else qids.tolist(),
                  probes_file=result.get("probes_file"))

    wire.send("done")


def main(cls: type[Adapter], argv: list[str] | None = None) -> None:
    """The entry point of an adapter module: `if __name__ == "__main__": main(MyAdapter)`."""
    args = _parser().parse_args(argv)
    wire = Wire()
    adapter = cls()
    adapter.wire = wire
    signal.signal(signal.SIGTERM, adapter._terminate)

    try:
        threads = getattr(args, "threads", 1)
        adapter.variant = getattr(args, "variant", "")
        os.environ.update(thread_env(threads) | adapter.thread_env(threads))
        adapter.load()
        dataset = data.Dataset.from_base(args.data) if getattr(args, "data", None) else None
        run(adapter, args, wire, dataset)
    except Unsupported as e:
        wire.send("unsupported", message=str(e))
    except Exception as e:
        wire.send("error", message=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
        sys.exit(1)


def run(adapter: Adapter, args: argparse.Namespace, wire: Wire, dataset: data.Dataset | None) -> None:
    if args.command == "info":
        wire.send("info", **adapter.info())
    elif args.command == "files":
        wire.send("files", files=adapter.files(Path(args.index)))
    elif args.command == "build":
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        ctx = BuildContext(config=json.loads(args.config), data_path=Path(args.data), rows=args.rows, out=out,
                           clusters=Path(args.clusters) if args.clusters else None, threads=args.threads,
                           seed=args.seed, metric=args.metric or dataset.metric, variant=args.variant,
                           dataset=dataset)
        wire.send("build", fields=adapter.run_build(ctx), info=adapter.info())
    elif args.command == "overhead":
        ctx = BuildContext(config={}, data_path=Path(args.data), rows=args.rows, out=Path("."), clusters=None,
                           threads=args.threads, seed=1, metric=args.metric or dataset.metric, variant=args.variant,
                           dataset=dataset)
        wire.send("overhead", fields=adapter.run_overhead(ctx))
    elif args.command == "cold":
        _cold(adapter, args, wire, dataset)
    elif args.command == "warm":
        _warm(adapter, args, wire, dataset)
    elif args.command == "qps-worker":
        ctx = _serve(args, dataset)
        ctx.params = json.loads(args.params)
        ctx.queries = _queries(dataset, data.split_ids(dataset.queries.shape[0], args.split).tolist())
        adapter.qps_worker(ctx, wire, args.seconds)
    elif args.command == "write":
        ctx = _write_ctx(args, dataset)
        ctx.mode, ctx.pass_name, ctx.policy = args.mode, args.pass_name, args.policy
        ctx.maintain_params, ctx.maintain_threads = json.loads(args.maintain_params), args.maintain_threads
        adapter.write_session(ctx, args, wire)
    elif args.command == "maintain":
        adapter.maintain_session(_write_ctx(args, dataset), args.action, json.loads(args.params), wire)


def _write_ctx(args: argparse.Namespace, dataset: data.Dataset) -> WriteContext:
    return WriteContext(index=Path(args.index), dataset=dataset, rows=args.rows, metric=args.metric or dataset.metric,
                        threads=args.threads, variant=args.variant,
                        params=json.loads(getattr(args, "params", None) or "{}"))
