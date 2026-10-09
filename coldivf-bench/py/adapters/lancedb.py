"""LanceDB through lancedb and, when the environment has it, pylance (the lance package).

An index folder holds a LanceDB database with one table, "t", and meta.json with the build's settings. The table has
each base row's number in "id" and its vector in "vector", a fixed-size list of float32. The rows go in as one stream of
batches, so the table keeps a few large data files, and nothing compacts it.

The index is IVF_RQ, an inverted file index (IVF) over RaBitQ codes, or IVF_PQ over product quantization (PQ) codes.
lancedb's create_index makes it from a config object, which passes the bits as given. With shared clusters, pylance's
create_index takes their centroids as the partitions. A cosine dataset of unit vectors uses Lance's "dot" distance,
which ranks it the same way at less cost.

With pylance a search runs through its scanner, which takes approx_mode and query_parallelism. Without it, eg. where an
older lancedb is installed alone, a search runs through lancedb's query builder. Every search sets the minimum and the
maximum nprobes to the same value. LANCE_CPU_THREADS sizes Lance's compute pool to the job's threads.

A cold trial times the open and the first query. With pylance the open is pylance's own dataset open, as a user who
searches through its scanner opens the table once. {"open": "lancedb"} opens through lancedb's connect and open_table,
then to_lance, which opens the dataset a second time. Every line records the path in extra.open_path. In C1
(disk-resident steady state) the process keeps one session whose index cache is about the smallest that keeps the
index's model, its centroids and metadata, loaded, so nearly every partition comes from the files again on every query.
analyze_plan runs only in runs of its own, once per warm setting after the timed passes and in cold cells whose params
hold {"analyze": true}.
"""

from __future__ import annotations

import json
import os
import platform
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from adapters.base import Adapter, BuildContext, ServeContext, Unsupported, main
import data
import schema

TABLE = "t"
META = "meta.json"
BATCH_ROWS = 65536
# Lance trains its clusters on sample_rate rows per partition, 256 by default. The rate is capped so the training
# sample stays near this many rows. IVF_PQ also builds at most rows / sample_rate partitions.
SAMPLE_ROWS = 1_000_000
DEFAULT_SAMPLE_RATE = 256
DEFAULT_SUB_VECTORS = 96
# How many queries of a warm split run once more through analyze_plan after each setting.
ANALYZE_QUERIES = 8
# A cosine dataset counts as unit length when every sampled row's length is this close to 1.
UNIT_TOLERANCE = 1e-3
# C1's index cache grows one partition at a time, up to C1_TRIES times, until C1_CHECKS queries in a row keep the
# index loaded. After C1_STEPS steps each step doubles, so a cache that needs much more is still found. Once the index
# stays, the cache gets C1_MARGIN partitions more, so a query whose partitions are larger than most doesn't push it
# out.
C1_TRIES = 16
C1_STEPS = 8
C1_CHECKS = 6
C1_MARGIN = 1
# The most partitions a query probes when C1 measures the bytes one partition takes in the cache.
C1_SAMPLE_PROBES = 16

BUILD_KEYS = {"index", "bits", "num_sub_vectors", "partitions", "vpl", "sample_rate", "max_iterations", "distance",
              "batch_rows"}
SEARCH_KEYS = {"nprobes", "refine_factor", "approx_mode", "query_parallelism", "open", "analyze"}
PYLANCE_KEYS = {"approx_mode", "query_parallelism"}
OPEN_MODES = ("lancedb", "lance")


def open_mode(params: dict, pylance: bool) -> str:
    """How a search setting opens the table. With pylance it is "lance" by default, pylance's dataset open alone, since
    a user who searches with pylance's scanner opens once. "lancedb" opens through lancedb's connect and open_table
    first, then to_lance, which opens the dataset a second time. Without pylance it is always "lancedb"."""
    return str(params.get("open") or ("lance" if pylance else "lancedb"))

# Units of the metrics analyze_plan prints. Counts use powers of 1000, sizes powers of 1024, and times become
# microseconds.
COUNT_UNITS = {"": 1.0, "K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}
SIZE_UNITS = {"B": 1.0, "KB": 1024.0, "MB": 1024.0 ** 2, "GB": 1024.0 ** 3, "TB": 1024.0 ** 4}
TIME_UNITS = {"ns": 1e-3, "µs": 1.0, "us": 1.0, "ms": 1e3, "s": 1e6}
_VALUE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*([A-Za-zµ%]*)\s*$")
_OP = re.compile(r"^(?P<indent> *)(?P<op>[A-Za-z_][A-Za-z0-9_]*)")
_ELAPSED = re.compile(r"\belapsed=(\d+(?:\.\d+)?\s*[a-zµ]+)")
_PROJECTION = re.compile(r"\bprojection=\[([^\]]*)\]")
# The plan totals and the per-role figures a summary keeps.
TOTALS = ("bytes_read", "iops", "requests", "parts_loaded", "partitions_searched", "index_comparisons",
          "index_cache_hits", "index_cache_misses", "indices_loaded")
ROLE_KEYS = ("bytes_read", "iops", "requests", "parts_loaded", "rows_scanned", "elapsed_us")


def distance_type(metric: str, unit: bool) -> str:
    """Lance's distance for a dataset metric. Inner product ranks unit vectors as cosine does, and costs less."""
    if metric == "l2":
        return "l2"

    if metric == "ip" or unit:
        return "dot"

    return "cosine"


def unit_rows(rows: np.ndarray, sample: int = 1000, tolerance: float = UNIT_TOLERANCE) -> bool:
    """Whether evenly spaced sample rows all have length 1, within `tolerance`."""
    if len(rows) == 0:
        return False

    picks = np.unique(np.linspace(0, len(rows) - 1, min(sample, len(rows))).astype(np.int64))
    norms = np.linalg.norm(np.asarray(rows[picks], dtype=np.float64), axis=1)
    return bool(np.all(np.abs(norms - 1.0) <= tolerance))


def partitions(config: dict, rows: int) -> int | None:
    """The partition count a build asks for. It is "partitions" as given, or the row count over "vpl" (vectors per
    list), or None for Lance's own default."""
    if config.get("partitions"):
        return int(config["partitions"])

    if config.get("vpl"):
        return max(1, round(rows / float(config["vpl"])))

    return None


def sample_rate(config: dict, parts: int | None, rows: int) -> int:
    """Rows per partition in the training sample. A rate in the config wins. Otherwise it is Lance's 256, lowered so
    partitions × rate stays at or below the row count and SAMPLE_ROWS. IVF_PQ then builds every partition it is asked
    for, since it builds at most rows / rate."""
    if config.get("sample_rate"):
        return int(config["sample_rate"])

    if not parts:
        return DEFAULT_SAMPLE_RATE

    return max(1, min(DEFAULT_SAMPLE_RATE, min(rows, SAMPLE_ROWS) // parts))


def index_kwargs(config: dict, distance: str, parts: int | None, rate: int) -> tuple[str, dict]:
    """The index kind, "ivf_rq" or "ivf_pq", and the arguments of its lancedb config object."""
    kind = str(config.get("index", "ivf_rq")).lower()

    if kind not in ("ivf_rq", "ivf_pq"):
        raise ValueError(f"LanceDB builds ivf_rq or ivf_pq here, not {kind}")

    out: dict[str, Any] = {"distance_type": distance, "num_partitions": parts, "sample_rate": rate,
                           "max_iterations": int(config.get("max_iterations", 50))}

    if kind == "ivf_rq":
        out["num_bits"] = int(config.get("bits", 1))
    else:
        out["num_bits"] = int(config.get("bits", 8))
        out["num_sub_vectors"] = int(config.get("num_sub_vectors", DEFAULT_SUB_VECTORS))

    return kind, out


def check_build(config: dict) -> None:
    unknown = sorted(set(config) - BUILD_KEYS)

    if unknown:
        raise ValueError(f"LanceDB takes no build setting {unknown}")


def check_search(params: dict, pylance: bool) -> None:
    """Rejects settings this adapter doesn't know, and pylance's settings where only lancedb is installed."""
    unknown = sorted(set(params) - SEARCH_KEYS)

    if unknown:
        raise ValueError(f"LanceDB takes no search setting {unknown}")

    if "nprobes" not in params:
        raise ValueError("a LanceDB search needs nprobes")

    mode = open_mode(params, pylance)

    if mode not in OPEN_MODES:
        raise ValueError(f"open is lancedb or lance, not {mode}")

    if not pylance:
        given = sorted(k for k in PYLANCE_KEYS if params.get(k) is not None)

        if given or mode == "lance":
            what = ", ".join(given) if given else "open lance"
            raise Unsupported(f"{what} needs pylance, which this environment lacks")


def nearest(query: np.ndarray, k: int, params: dict) -> dict:
    """pylance's nearest argument for a search. query_parallelism is -1 unless set, which runs the partitions on the
    compute pool, since the default runs them one at a time."""
    n = int(params["nprobes"])
    out = {"column": "vector", "q": query, "k": k, "minimum_nprobes": n, "maximum_nprobes": n,
           "query_parallelism": int(params.get("query_parallelism", -1))}

    if params.get("refine_factor"):
        out["refine_factor"] = int(params["refine_factor"])

    if params.get("approx_mode") is not None:
        out["approx_mode"] = str(params["approx_mode"])

    return out


def kind_of(relative: str) -> str:
    """Whether a file of an index folder is routing data, which every query reads and C1 keeps, or bulk data. Routing
    is meta.json, the manifests and index.idx, which holds the index's centroids. Bulk is the partitions in
    auxiliary.idx, the data files and the transaction files."""
    parts = Path(relative).parts
    name = parts[-1] if parts else ""

    if name == META or name.endswith(".manifest") or name == "latest_version_hint.json" or "_versions" in parts:
        return "routing"

    if "_indices" in parts and name == "index.idx":
        return "routing"

    return "bulk"


def parse_value(text: str, key: str = "") -> float | None:
    """A metric value of analyze_plan as a number. Times become microseconds, eg. "1.97ms" is 1970. Sizes become bytes,
    eg. "126.1 KB", and counts plain numbers, eg. "25.95 K" is 25950. "B" is bytes for a key that ends in "bytes", and
    otherwise billions. Anything else, eg. "100% (10/10)", is None."""
    m = _VALUE.match(text)

    if not m:
        return None

    number, unit = float(m.group(1)), m.group(2)

    if unit in TIME_UNITS:
        return number * TIME_UNITS[unit]

    if unit in SIZE_UNITS and (unit != "B" or key.endswith("bytes")):
        return number * SIZE_UNITS[unit]

    if unit in COUNT_UNITS:
        return number * COUNT_UNITS[unit]

    return None


def role_of(op: str, projection: str | None) -> str | None:
    """What a plan stage does in a vector search. "ranking" orders the partitions, "partitions" searches the probed
    ones, "refine" reads the candidates' vectors, and "ids" reads the id column."""
    if op == "ANNIvfPartition":
        return "ranking"

    if op == "ANNSubIndex":
        return "partitions"

    if projection is not None and op.startswith("Lance"):
        columns = {c.strip() for c in projection.split(",")}

        if "vector" in columns:
            return "refine"

        if "id" in columns:
            return "ids"

    return None


def parse_plan(text: str) -> list[dict]:
    """The stages of analyze_plan's output, one per operator line, with its depth, elapsed time, role and numeric
    metrics."""
    stages = []

    for line in text.splitlines():
        m = _OP.match(line)

        if not m or not line.strip():
            continue

        head, metrics = line, {}
        at = line.find("metrics=[")

        if at >= 0:
            head = line[:at]
            body = line[at + len("metrics=["):]
            body = body[: body.rfind("]")] if "]" in body else body

            for item in body.split(", "):
                key, sep, value = item.partition("=")

                if sep and (number := parse_value(value, key.strip())) is not None:
                    metrics[key.strip()] = number

        elapsed = _ELAPSED.search(head)
        projection = _PROJECTION.search(head)
        op = m.group("op")
        stages.append({
            "op": op,
            "depth": len(m.group("indent")) // 2,
            "elapsed_us": parse_value(elapsed.group(1)) if elapsed else None,
            "role": role_of(op, projection.group(1) if projection else None),
            "metrics": metrics,
        })

    return stages


def plan_summary(stages: list[dict]) -> dict:
    """Totals over a plan's stages, the elapsed time of its first stage, and per role the figures of its stages."""
    out: dict[str, Any] = {key: sum(s["metrics"].get(key, 0.0) for s in stages) for key in TOTALS}
    out["elapsed_us"] = stages[0]["elapsed_us"] if stages else None
    roles: dict[str, dict] = {}

    for s in stages:
        if s["role"] is None:
            continue

        into = roles.setdefault(s["role"], dict.fromkeys(ROLE_KEYS, 0.0))

        for key in ROLE_KEYS:
            value = s["elapsed_us"] if key == "elapsed_us" else s["metrics"].get(key)
            into[key] += value or 0.0

    out["by_role"] = roles
    return out


def mean_summary(summaries: list[dict]) -> dict:
    """The mean of plan summaries, leaf by leaf, with the number of plans it covers."""
    if not summaries:
        return {}

    def mean(values: list[Any]) -> Any:
        if all(isinstance(v, dict) for v in values):
            keys = sorted({k for v in values for k in v})
            return {k: mean([v[k] for v in values if k in v]) for k in keys}

        numbers = [float(v) for v in values if isinstance(v, (int, float))]
        return sum(numbers) / len(numbers) if numbers else None

    return mean(summaries) | {"plans": len(summaries)}


def pad(ids: np.ndarray, k: int) -> np.ndarray:
    """The first k ids, padded with -1."""
    out = np.full(k, -1, dtype=np.int64)
    ids = np.asarray(ids, dtype=np.int64)[:k]
    out[: len(ids)] = ids
    return out


def read_meta(path: Path | str) -> dict:
    return json.loads((Path(path) / META).read_text())


@dataclass
class Handle:
    """One open table. `dataset` is pylance's view of it, and `table` lancedb's."""

    path: Path
    meta: dict
    db: Any = None
    table: Any = None
    dataset: Any = None
    session: Any = None
    # Lance's read counters when the last stats were taken, or when the open started.
    io_mark: tuple[int, int] = (0, 0)
    # The stages of the last analyze run.
    plan: list | None = None


class LanceDB(Adapter):
    system = "lancedb"
    # A search releases Python's lock, so queries per second (QPS) at several threads come from threads over one table.
    qps_mode = "threads"
    workers = "Lance's compute and IO pools"

    def __init__(self) -> None:
        super().__init__()
        self.ldb: Any = None
        self.lance: Any = None
        self.pa: Any = None
        self.metas: dict[str, dict] = {}
        # The index cache capacity of C1's session, set before READY, and how c1_capacity found it.
        self.c1_cache: int | None = None
        self.c1: dict = {}
        self.last: Handle | None = None

    def load(self) -> None:
        import lancedb
        import pyarrow

        self.ldb, self.pa = lancedb, pyarrow

        try:
            import lance
        except ImportError:
            lance = None

        self.lance = lance

    @property
    def pylance(self) -> bool:
        return self.lance is not None

    def version(self) -> str:
        text = f"lancedb {self.ldb.__version__}"
        return text + (f", pylance {self.lance.__version__}" if self.pylance else "")

    def wheels(self) -> dict[str, str | None]:
        names = ("lancedb", "pylance", "pyarrow") if self.pylance else ("lancedb", "pyarrow")
        return {name: schema.dist_sha256(name) for name in names}

    def simd_level(self) -> str | None:
        if self.pylance:
            return str(self.lance.simd_info().get("tier"))

        # Lance's kernels for 64-bit Arm use NEON, which every such chip has.
        return "neon" if platform.machine() == "arm64" else None

    def thread_env(self, threads: int) -> dict[str, str]:
        return {"LANCE_CPU_THREADS": str(threads)}

    def overhead_config(self) -> dict:
        return {"index": "ivf_rq", "bits": 1, "partitions": 4}

    def overhead_params(self) -> dict:
        return {"nprobes": 1}

    def build(self, ctx: BuildContext) -> dict:
        config = dict(ctx.config)
        check_build(config)
        rows = ctx.data
        distance = config.get("distance") or distance_type(ctx.metric, ctx.metric == "cosine" and unit_rows(rows))
        centroids = data.clusters(ctx.clusters)[0] if ctx.clusters is not None else None
        parts = len(centroids) if centroids is not None else partitions(config, ctx.rows)
        rate = sample_rate(config, parts, ctx.rows)
        kind, kwargs = index_kwargs(config, distance, parts, rate)

        if centroids is not None and not self.pylance:
            raise Unsupported("importing clusters needs pylance's create_index")

        db = self.ldb.connect(str(ctx.out))
        start = time.perf_counter()
        table = db.create_table(TABLE, self.batches(rows, int(config.get("batch_rows", BATCH_ROWS))),
                                schema=self.table_schema(ctx.dim))
        ingested = time.perf_counter()

        if centroids is not None:
            self.index_from_centroids(table, kind, kwargs, np.asarray(centroids, dtype=np.float32))
        else:
            table.create_index("vector", config=self.index_config(kind, kwargs))

        indexed = time.perf_counter()
        built = self.partitions_built(ctx.out)
        meta = {"table": TABLE, "distance": distance, "index": kind, "config": kwargs, "rows": ctx.rows,
                "dim": ctx.dim, "partitions": built, "centroids": "imported" if centroids is not None else "trained",
                "lancedb": self.ldb.__version__, "pylance": self.lance.__version__ if self.pylance else None}
        (ctx.out / META).write_text(json.dumps(meta, indent=1))
        return {
            "cluster_seconds": None,
            "encode_seconds": None,
            "stored_vector_bytes": ctx.rows * ctx.dim * 4,
            "exact_f32_stored": True,
            "extra": {"ingest_seconds": ingested - start, "index_seconds": indexed - ingested, "distance": distance,
                      "index": kind, "partitions": built, "partitions_asked": parts, "sample_rate": rate,
                      "centroids": meta["centroids"]},
        }

    def table_schema(self, dim: int) -> Any:
        pa = self.pa
        return pa.schema([pa.field("id", pa.int64()), pa.field("vector", pa.list_(pa.float32(), dim))])

    def batches(self, rows: np.ndarray, size: int) -> Any:
        """The rows as one stream of record batches, so the table is written in one pass with large data files."""
        pa = self.pa
        dim = rows.shape[1]
        table_schema = self.table_schema(dim)

        def gen() -> Iterator[Any]:
            for lo in range(0, len(rows), size):
                block = np.ascontiguousarray(rows[lo : lo + size], dtype=np.float32)
                vectors = pa.FixedSizeListArray.from_arrays(pa.array(block.reshape(-1)), dim)
                ids = pa.array(np.arange(lo, lo + len(block), dtype=np.int64))
                yield pa.RecordBatch.from_arrays([ids, vectors], schema=table_schema)

        return pa.RecordBatchReader.from_batches(table_schema, gen())

    def index_config(self, kind: str, kwargs: dict) -> Any:
        from lancedb.index import IvfPq, IvfRq

        return (IvfRq if kind == "ivf_rq" else IvfPq)(**kwargs)

    def index_from_centroids(self, table: Any, kind: str, kwargs: dict, centroids: np.ndarray) -> None:
        """Builds the index on given centroids through pylance, which trains only the codes."""
        args = {"metric": kwargs["distance_type"], "ivf_centroids": centroids, "num_bits": kwargs["num_bits"],
                "sample_rate": kwargs["sample_rate"], "max_iterations": kwargs["max_iterations"]}

        if kind == "ivf_pq":
            args["num_sub_vectors"] = kwargs["num_sub_vectors"]

        table.to_lance().create_index("vector", kind.upper(), **args)

    def partitions_built(self, path: Path) -> int | None:
        """The index's partition count, from pylance's index statistics of the latest version, or None without
        pylance."""
        if not self.pylance:
            return None

        # Statistics without the centroids, which only make them bigger.
        os.environ["LANCE_INCLUDE_VECTOR_CENTROIDS"] = "false"

        try:
            stats = self.lance.dataset(str(Path(path) / f"{TABLE}.lance")).stats.index_stats("vector_idx")
            return int(stats["indices"][0]["num_partitions"])
        except (KeyError, IndexError, TypeError, ValueError, RuntimeError, OSError):
            return None

    def meta(self, path: Path | str) -> dict:
        """The build settings of an index, read once. Cold runs read them before READY, so a trial never does."""
        key = str(path)

        if key not in self.metas:
            self.metas[key] = read_meta(path)

        return self.metas[key]

    def files(self, path: Path) -> list[dict]:
        """Every file of the index folder. The routing files are also "loaded", since Lance reads them into its own
        caches, the manifests at open and index.idx at the first query, and doesn't read them from the file again."""
        root = Path(path)
        out = []

        for p in sorted(x for x in root.rglob("*") if x.is_file()):
            size = p.stat().st_size
            kind = kind_of(str(p.relative_to(root)))
            span = {"offset": 0, "length": size, "kind": kind} | ({"loaded": True} if kind == "routing" else {})
            out.append({"path": str(p), "bytes": size, "ranges": [span]})

        return out

    def configure(self, handle: Handle, params: dict) -> None:
        check_search(params, self.pylance)

    def open(self, ctx: ServeContext, path: Path) -> Handle:
        session = None

        if ctx.regime == "c1" and self.c1_cache is not None:
            session = self.new_session(self.c1_cache)

        handle = self.open_with(Path(path), ctx.params, session)
        self.last = handle
        return handle

    def open_with(self, path: Path, params: dict, session: Any = None) -> Handle:
        handle = Handle(path=path, meta=self.meta(path), session=session)
        mode = open_mode(params, self.pylance)

        if self.pylance:
            handle.io_mark = self.io_counts()
            options = {"session": session} if session is not None else {}

            if mode == "lance":
                handle.dataset = self.lance.dataset(str(path / f"{TABLE}.lance"), **options)
            else:
                handle.db = self.ldb.connect(str(path))
                handle.table = handle.db.open_table(TABLE)
                handle.dataset = handle.table.to_lance(**options)
        else:
            handle.db = self.ldb.connect(str(path), **({"session": session} if session is not None else {}))
            handle.table = handle.db.open_table(TABLE)

        return handle

    def close(self, handle: Handle) -> None:
        handle.dataset = handle.table = handle.db = None

    def io_counts(self) -> tuple[int, int]:
        """pylance's counts of bytes read and read operations so far in this process."""
        return int(self.lance.bytes_read_counter()), int(self.lance.iops_counter())

    def builder(self, handle: Handle, query: np.ndarray, k: int, params: dict) -> Any:
        """lancedb's query for a search. It selects _distance too, since leaving it out logs a warning per query."""
        n = int(params["nprobes"])
        q = handle.table.search(query, vector_column_name="vector").distance_type(handle.meta["distance"])
        q = q.minimum_nprobes(n).maximum_nprobes(n).select(["id", "_distance"]).limit(k)
        return q.refine_factor(int(params["refine_factor"])) if params.get("refine_factor") else q

    def search(self, handle: Handle, query: np.ndarray, k: int, params: dict) -> np.ndarray:
        if params.get("analyze"):
            handle.plan = parse_plan(self.analyze_text(handle, query, k, params))
            return pad(np.empty(0), k)

        if self.pylance:
            table = handle.dataset.to_table(nearest=nearest(query, k, params), columns=["id"],
                                            disable_scoring_autoprojection=True)
        else:
            table = self.builder(handle, query, k, params).to_arrow()

        return pad(table.column("id").to_numpy(), k)

    def analyze_text(self, handle: Handle, query: np.ndarray, k: int, params: dict) -> str:
        if self.pylance:
            scanner = handle.dataset.scanner(nearest=nearest(query, k, params), columns=["id"],
                                             disable_scoring_autoprojection=True)
            return scanner.analyze_plan()

        return self.builder(handle, query, k, params).analyze_plan()

    def stats(self, handle: Handle) -> dict:
        """pylance's bytes and read operations since the last stats or the open, and the last analyze run's plan.
        lancedb has a Lance of its own, so the reads of its connect and open_table aren't in pylance's counts."""
        out: dict[str, Any] = {}

        if self.pylance:
            now = self.io_counts()
            out = {"bytes_requested": now[0] - handle.io_mark[0], "iops": now[1] - handle.io_mark[1]}
            handle.io_mark = now

        if handle.plan is not None:
            out["analyze"] = plan_summary(handle.plan) | {"stages": handle.plan}
            handle.plan = None

        return out

    def cache_bytes(self, handle: Handle) -> int | None:
        """The bytes in a handle's index cache, or for lancedb alone in its whole session."""
        if handle.session is None:
            return None

        return int(handle.session.index_cache_size_bytes() if self.pylance else handle.session.size_bytes)

    def new_session(self, capacity: int | None = None) -> Any:
        make = self.lance.Session if self.pylance else self.ldb.Session
        return make(index_cache_size_bytes=capacity) if capacity else make()

    def cached_after(self, ctx: ServeContext, params: dict, query: np.ndarray) -> int:
        """The bytes a fresh session's index cache holds after one search with `params`."""
        handle = self.open_with(Path(ctx.index), params, self.new_session())
        self.search(handle, query, ctx.k, params)
        held = int(self.cache_bytes(handle) or 0)
        self.close(handle)
        return held

    def keeps_index(self, ctx: ServeContext, capacity: int, queries: list[np.ndarray]) -> bool:
        """Whether a session whose index cache holds `capacity` bytes keeps the index loaded over `queries`. The first
        query loads it into the empty cache, and the later ones must find it there, as analyze_plan's indices_loaded
        shows."""
        handle = self.open_with(Path(ctx.index), ctx.params, self.new_session(capacity))
        loaded = []

        for q in queries:
            self.search(handle, q, ctx.k, dict(ctx.params, analyze=True))
            loaded.append(plan_summary(handle.plan)["indices_loaded"])

        self.close(handle)
        return not any(loaded[1:])

    def c1_capacity(self, ctx: ServeContext) -> dict:
        """The index cache that C1's session keeps, about the smallest that keeps the index's model loaded, i.e. its
        centroids and metadata. Its searches run the warm-up queries, which no trial searches.

        One query probing one partition fills a fresh cache with the model plus one partition, and one probing more
        partitions shows the bytes of a partition. The capacity starts at the model plus one partition and grows one
        partition at a time until queries stop loading the index again, then gets C1_MARGIN partitions more. A cache
        that size can still keep a few partitions between queries, so each trial reports how many it can hold."""
        queries = list(ctx.warmup) or list(ctx.queries.values())
        checks = (queries * C1_CHECKS)[:C1_CHECKS]
        probes = max(2, min(int(ctx.params["nprobes"]), C1_SAMPLE_PROBES))
        plain = dict(ctx.params, refine_factor=None, analyze=False)
        one = self.cached_after(ctx, dict(plain, nprobes=1), checks[0])
        part = max(1, (self.cached_after(ctx, dict(plain, nprobes=probes), checks[0]) - one) // (probes - 1))
        model = max(1, one - part)
        capacity, step, kept = one, part, False

        for tries in range(1, C1_TRIES + 1):
            if self.keeps_index(ctx, capacity, checks):
                kept = True
                break

            capacity += step

            if tries >= C1_STEPS:
                step *= 2

        capacity += C1_MARGIN * part if kept else 0
        return {"capacity": capacity, "start": one, "model_bytes": model, "partition_bytes": part, "kept": kept,
                "tries": tries}

    def cold_setup(self, ctx: ServeContext) -> None:
        check_search(ctx.params, self.pylance)
        self.meta(ctx.index)

        if ctx.dummy is not None:
            self.meta(ctx.dummy)

        if ctx.regime == "c1":
            self.c1 = self.c1_capacity(ctx)
            self.c1_cache = self.c1["capacity"]

        super().cold_setup(ctx)

    def cold_trial(self, ctx: ServeContext, go: dict) -> dict:
        if ctx.regime == "c1":
            # The counters start at this trial, not at the warm-up query or the last trial's later queries.
            self.stats(ctx.handle)

        out = super().cold_trial(ctx, go)
        out["extra"]["open_path"] = open_mode(ctx.params, self.pylance)

        if ctx.params.get("analyze"):
            # An analyze run returns a plan, not ids, so its trial has no recall.
            out["ids"] = None

        if ctx.regime == "c1":
            held = self.cache_bytes(ctx.handle)
            c1 = self.c1
            out["extra"] |= {
                "index_cache_capacity": c1["capacity"], "index_cache_start": c1["start"],
                "index_cache_keeps_index": c1["kept"], "index_cache_bytes": held,
                "index_cache_model_bytes": c1["model_bytes"], "index_cache_partition_bytes": c1["partition_bytes"],
                # How many partitions the cache can hold beside the model, and how many it holds after this trial.
                "partitions_cached_max": round((c1["capacity"] - c1["model_bytes"]) / c1["partition_bytes"], 1),
                "partitions_cached": round(max(0, (held or 0) - c1["model_bytes"]) / c1["partition_bytes"], 1),
            }

        return out

    def warm_run(self, ctx: ServeContext, grid: list[dict], qids: np.ndarray, ids_dir: Path, passes: int,
                 warmup: int, qps_seconds: float, qps_repeats: int) -> Iterator[dict]:
        """The base class's warm run, plus one analyze run per setting over the first queries of the split, after the
        setting's timed passes. Its index comparisons become dist_evals."""
        self.meta(ctx.index)
        sample = [ctx.queries[int(q)] for q in qids[:ANALYZE_QUERIES]]

        for result in super().warm_run(ctx, grid, qids, ids_dir, passes, warmup, qps_seconds, qps_repeats):
            params = dict(result["params"], analyze=True)
            plans = []

            for q in sample:
                self.search(self.last, q, ctx.k, params)
                plans.append(plan_summary(self.last.plan))
                self.last.plan = None

            summary = mean_summary(plans)
            fields = result["fields"]
            fields["extra"] = (fields.get("extra") or {}) | {"analyze": summary,
                                                             "open_path": open_mode(result["params"], self.pylance)}

            if fields.get("dist_evals") is None and summary.get("index_comparisons"):
                fields["dist_evals"] = summary["index_comparisons"]

            yield result


if __name__ == "__main__":
    main(LanceDB)
