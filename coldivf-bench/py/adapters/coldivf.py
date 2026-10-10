"""ColdIVF through its release binaries build_index, warm and cold, from $COLDIVF_BIN or target/mine/release.

The binaries take a dataset by name and find its files through $COLDIVF_DATA. For warm, a tune or test split becomes a
dataset of its own under data/splits/, with a link to the base, the split's queries, and the split's ground truth with
its distances, so warm sees the split's queries in order. warm counts ties when the distances are there, which gives
the same recall the driver computes for every other system.

cold searches the dataset itself, and takes each trial's queries by their ids. A C0 or intent trial is one
`cold --one` run, a fresh process that first searches the cell's dummy index once, if it has one. It maps the index
once, drops its pages through that mapping, waits for anything that reads them back, and checks them right before its
timer. It times Index::open, prepare and the search of the first query, searches the next queries without dropping
anything, then the first query again with its pages cached. In intent the vector arrives delay_ms after the timer
starts, and the binary times the search from then. A C1 cell runs one `cold --one --c1` process. It opens the index,
searches the cell's warm-up query, which loads the cluster tables it then keeps, and runs each trial from a line of its
input. The binary prints each query's ids, so the driver scores ColdIVF with the same recall code as every other
system. A trial whose own check finds cached pages comes back rejected. The binaries time each search inside their own
process, so their per-call cost is 0.
"""

from __future__ import annotations

import json
import os
import platform
import struct
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from adapters.base import Adapter, BuildContext, ServeContext, Unsupported, main
import data
import schema

REPO = Path(__file__).resolve().parents[3]
BIN = Path(os.environ.get("COLDIVF_BIN") or REPO / "target" / "mine" / "release")

BUILD_FLAGS = {
    "vectors_per_list": "--vectors-per-list", "parents": "--parents", "order": "--order", "anchor": "--anchor",
    "page_size": "--page-size", "clustering": "--clustering",
}
SEARCH_FLAGS = {"nprobe": "--nprobe", "rerank": "--rerank", "reach": "--reach"}

# Build settings that only some builds of build_index take, with the text that shows a binary takes it. The binaries
# ignore flags they don't know, so a setting a binary lacks would build the default without a word.
NEWER_BUILD_FLAGS = {"anchor": b"--anchor must"}

# The text of an error message that only a cold binary of this harness holds, one that checks each index file's vector
# count, waits for the vector as the other systems do and reports its footprints. An older binary would search other
# queries than the trial's, or wait for the vector another way, without a word.
COLD_MARKER = b"that the manifest or the dataset gives it"

# The share of a C1 trial's bulk pages that may still be cached when its timer starts, as the driver allows.
C1_CACHED = 0.005

# The index file's header holds a table of 7 sections at byte 80, each an offset, a length, a kind and a checksum.
SECTIONS = ("header", "rotation", "centers", "extents", "codes", "rows", "ids")
ROUTING = {"header", "rotation", "centers", "extents"}
HEADER_BYTES = 324
MAGIC = b"COLDIVF\0"


def index_file(path: Path | str) -> Path:
    """The .coldivf file of an index, given the file or a folder that holds exactly one."""
    path = Path(path)

    if path.is_file():
        return path

    found = sorted(path.glob("*.coldivf"))

    if len(found) != 1:
        raise FileNotFoundError(f"{path} should hold one .coldivf file, not {len(found)}")

    return found[0]


def sections(path: Path | str) -> dict[str, tuple[int, int]]:
    """Each section's offset and length, from the header of the index file at `path`."""
    with open(path, "rb") as f:
        head = f.read(HEADER_BYTES)

    if len(head) < HEADER_BYTES or head[:8] != MAGIC or struct.unpack_from("<I", head, 8)[0] != 1:
        raise ValueError(f"{path} isn't a ColdIVF index file of version 1")

    out = {}

    for i, name in enumerate(SECTIONS):
        offset, length, _, _ = struct.unpack_from("<QQII", head, 80 + 24 * i)
        out[name] = (offset, length)

    return out


def header_count(path: Path | str) -> int:
    """The vector count in an index file's header."""
    with open(path, "rb") as f:
        head = f.read(32)

    return struct.unpack_from("<Q", head, 24)[0]


def flags(params: dict, table: dict[str, str], what: str) -> list[str]:
    """Command-line flags for settings, which must all be ones the binary takes."""
    unknown = sorted(set(params) - set(table))

    if unknown:
        raise Unsupported(f"the ColdIVF binaries take no {what} {unknown}")

    out = []

    for key, value in params.items():
        if value is not None:
            out += [table[key], str(value).lower() if isinstance(value, bool) else str(value)]

    return out


def _fresh(target: Path, source: Path) -> bool:
    return target.exists() and target.stat().st_mtime_ns >= source.stat().st_mtime_ns


def truth_distances(ds: data.Dataset, queries: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Each truth id's distance from its query, as the binaries measure it. That is the squared Euclidean distance for a
    dataset whose name ends in -euclidean, and minus the inner product for any other, whose rows have unit length. It
    is computed in float64 from the base rows, one query's rows at a time."""
    l2 = ds.name.endswith("-euclidean")
    out = np.empty(ids.shape, dtype=np.float32)

    for i, row in enumerate(np.asarray(ids)):
        if np.any(row < 0):
            raise ValueError(f"the truth of query row {i} names a row below 0")

        rows = np.asarray(ds.base[row], dtype=np.float64)
        q = np.asarray(queries[i], dtype=np.float64)
        out[i] = ((rows - q) ** 2).sum(axis=1) if l2 else -(rows @ q)

    return out


def split_dataset(ds: data.Dataset, n: int, split: str) -> tuple[Path, str]:
    """The folder and name of a dataset that holds only one split's queries, for binaries that take a dataset name. Its
    base is a link to the real base, so nothing large is copied. Its truth has a distances file beside it, which makes
    the binaries count ties, as the driver does for every other system."""
    if split == "all":
        return ds.root, ds.name

    name = f"{split}.{ds.name}"
    root = ds.derived / "splits"
    root.mkdir(parents=True, exist_ok=True)
    qids = data.split_ids(ds.queries.shape[0], split)
    link = root / f"{name}.base.fbin"

    if not (link.is_symlink() and os.readlink(link) == str(ds.base_path)):
        tmp = root / f".{name}.{os.getpid()}.link"
        tmp.unlink(missing_ok=True)
        os.symlink(ds.base_path, tmp)
        os.replace(tmp, link)

    queries = root / f"{name}.query.fbin"

    if not _fresh(queries, ds.query_path):
        data.write_fbin(queries, np.asarray(ds.queries[qids]))

    truth = ds.truth(n)
    target = root / f"{name}.gt.ibin" if n >= ds.count else ds.derived / f"{name}-{n}.gt.ibin"

    if not _fresh(target, truth.path):
        data.write_ibin(target, np.asarray(truth.ids[qids]))

    dists = data.dist_path(target)

    if not _fresh(dists, target):
        data.write_fbin(dists, truth_distances(ds, np.asarray(ds.queries[qids]), np.asarray(truth.ids[qids])))

    return root, name


def near_ties(recall: data.Recall, qids: np.ndarray) -> int:
    """How many queries of `qids` have a tie at rank 10 between rows that differ. The driver counts a result within two
    float32 units of the true 10th distance as a hit, and the binaries only one at most as far as it, so at such a
    query the binary's recall with ties can come out lower than the driver's."""
    truth, base = recall.truth, recall.dataset.base

    if truth.k <= 10:
        return 0

    tied = np.asarray(qids)[~recall.tie_free(qids, 10)]
    return sum(1 for q in tied.tolist() if not np.array_equal(base[truth.ids[q, 9]], base[truth.ids[q, 10]]))


def offers(binary: Path, marker: bytes) -> bool:
    """Whether a binary holds `marker`, eg. the text of an option's error message, which shows it takes that option."""
    try:
        return marker in binary.read_bytes()
    except OSError:
        return False


def _crate_version() -> str:
    try:
        return tomllib.loads((REPO / "Cargo.toml").read_text())["workspace"]["package"]["version"]
    except (OSError, KeyError, ValueError):
        return "unknown"


class ColdIVF(Adapter):
    system = "coldivf"
    binding = "cli"
    regimes = ("c0", "c1", "intent")
    # A cold search reads in F_RDADVISE waves from the calling thread, so it runs no read threads.
    workers = ""

    def __init__(self) -> None:
        super().__init__()
        self.file: Path | None = None
        # The C1 process of a C1 cell.
        self.server: subprocess.Popen | None = None

    def binary(self, name: str) -> Path:
        path = BIN / name

        if not path.exists():
            raise FileNotFoundError(f"{path} is missing. Build it with cargo build --release -p coldivf-bench "
                                    f"--target-dir target/mine")

        return path

    def version(self) -> str:
        cold = BIN / "cold"
        tag = schema.file_sha256(cold)[:12] if cold.exists() else "missing"
        return f"{_crate_version()}+{tag}"

    def wheels(self) -> dict[str, str | None]:
        return {name: schema.file_sha256(BIN / name) if (BIN / name).exists() else None
                for name in ("build_index", "warm", "cold")}

    def simd_level(self) -> str:
        # The binaries pick their kernels at run time, which is NEON with dot products on Apple silicon.
        return "neon" if platform.machine() == "arm64" else "runtime"

    def env(self, root: Path) -> dict:
        return os.environ | {"COLDIVF_DATA": str(root)}

    def run_build(self, ctx: BuildContext) -> dict:
        if ctx.clusters is not None:
            raise Unsupported("build_index can't import clusters yet")

        out = ctx.out / "index.coldivf"
        binary = self.binary("build_index")

        for key, marker in NEWER_BUILD_FLAGS.items():
            if key in ctx.config and not offers(binary, marker):
                raise Unsupported(f"this build_index takes no {key}")

        with tempfile.TemporaryDirectory(prefix="coldivf-build-") as tmp:
            results = Path(tmp) / "build.jsonl"
            cmd = [str(binary), "--dataset", ctx.dataset.name, "--n", str(ctx.rows), "--out",
                   str(out), "--threads", str(ctx.threads), "--seed", str(ctx.seed), "--results", str(results),
                   *flags(ctx.config, BUILD_FLAGS, "build setting")]
            done = self.run_child(cmd, env=self.env(ctx.dataset.root), capture=False)

            if done.returncode != 0:
                raise RuntimeError(f"build_index exited with {done.returncode}")

            line = schema.read(results)[-1]

        return {
            "seconds": line.get("seconds"),
            "cpu_seconds": line.get("cpu_seconds"),
            "threads": line.get("threads"),
            "seed": line.get("seed"),
            "peak_footprint": line.get("peak_footprint"),
            "bytes_read": line.get("disk_read"),
            "bytes_written": line.get("disk_written"),
            "stored_vector_bytes": ctx.rows * ctx.dim,
            "exact_f32_stored": False,
            "extra": {k: line.get(k) for k in ("commit", "dirty", "built", "index_crc", "index_epoch", "input_bytes",
                                               "clustering", "vectors_per_list", "parents", "order", "page_size")},
        }

    def files(self, path: Path) -> list[dict]:
        file = index_file(path)
        size = file.stat().st_size
        ranges = [{"offset": off, "length": length, "kind": "routing" if name in ROUTING else "bulk"}
                  for name, (off, length) in sections(file).items() if length > 0]
        out = [{"path": str(file), "bytes": size, "ranges": ranges}]

        # The log of inserts and the writers' lock sit beside the file, and are watched even before they exist.
        for suffix, kind in ((".delta", "bulk"), (".lock", "routing")):
            side = Path(f"{file}{suffix}")
            n = side.stat().st_size if side.exists() else 0
            ranges = [{"offset": 0, "length": n, "kind": kind}] if n else []
            out.append({"path": str(side), "bytes": n, "ranges": ranges})

        return out

    def warm_run(self, ctx: ServeContext, grid: list[dict], qids: np.ndarray, ids_dir: Path, passes: int,
                 warmup: int, qps_seconds: float, qps_repeats: int) -> Iterator[dict]:
        # A warm binary with passes, threads and the t1 mode names them in an error message.
        full = offers(self.binary("warm"), b"--mode t1")

        if ctx.threads > 1 and not full:
            raise Unsupported("this warm binary runs at 1 thread")

        if ctx.mode not in (("whole", "t1") if full else ("whole",)):
            raise Unsupported(f"this warm binary has no mode {ctx.mode}")

        root, name = split_dataset(ctx.dataset, ctx.rows, ctx.split)
        file = index_file(ctx.index)
        recall = data.Recall(ctx.dataset, ctx.dataset.truth(ctx.rows))
        free = {k: bool(np.all(recall.tie_free(qids, k))) for k in (1, 10)}
        near = near_ties(recall, qids)
        more = ["--mode", ctx.mode, "--passes", str(passes if ctx.threads <= 1 else qps_repeats)] if full else []

        if full and ctx.threads > 1:
            more += ["--threads", str(ctx.threads), "--seconds", str(qps_seconds)]

        for params in grid:
            with tempfile.TemporaryDirectory(prefix="coldivf-warm-") as tmp:
                results = Path(tmp) / "warm.jsonl"
                cmd = [str(self.binary("warm")), "--index", str(file), "--dataset", name, "--n", str(ctx.rows),
                       "--k", str(ctx.k), "--queries", str(len(qids)), "--results", str(results), *more,
                       *flags(params, SEARCH_FLAGS, "search setting")]
                done, used = self.run_measured(cmd, env=self.env(root))

                if done.returncode != 0:
                    raise RuntimeError(f"warm exited with {done.returncode}")

                line = schema.read(results)[-1]

            fields = self.warm_fields(ctx, line, free)
            fields["extra"]["near_ties"] = near
            # The binary's whole process, at its peak, which is all of the warm run's memory besides the page cache.
            footprint = used.lifetime_max_footprint or None
            fields.update(self._memory(ctx, footprint) | {"footprint": footprint})
            fields["extra"]["footprint_source"] = "the warm binary's whole process, at its peak"
            yield {"params": params, "fields": fields, "ids_file": None}

    def warm_fields(self, ctx: ServeContext, line: dict, free: dict[int, bool]) -> dict:
        """The warm fields from one line of the warm binary. With the distances file beside the split's truth, the
        binary counts ties as the driver does for other systems, and says so in "ties". Without it, eg. from an older
        binary, its recall counts ids alone. That equals recall with ties wherever the truth has no tie at that rank,
        so then a recall field is filled only where no query of the split has one."""
        ties = bool(line.get("ties"))
        at10 = line.get("recall10", line.get("recall") if ctx.k == 10 else None)
        ids10 = line.get("recall") if ctx.k == 10 else (None if ties else at10)
        mean = line.get("mean_us")
        fields = {
            "p50_us": line.get("p50_us"),
            "p90_us": line.get("p90_us"),
            "p99_us": line.get("p99_us"),
            "mean_us": mean,
            "queries": line.get("queries"),
            "passes": line.get("passes", 1),
            "batch_api": False,
            "processes": 1,
            "qps": line.get("qps") or (1e6 / mean if mean else None),
            "recall10_ids_only": ids10,
            "recall10": at10 if ties or free[10] else None,
            "recall1": line.get("recall1") if ties or free[1] else None,
            "recall100": line.get("recall100") if ties else None,
            "recall10_p05": line.get("recall10_p05") if ties or free[10] else None,
            "recall10_p10": line.get("recall10_p10") if ties or free[10] else None,
            # The binary's share counts ids alone, even with ties.
            "share_ge_0_9": line.get("share_at_0_9") if free[10] else None,
            "dist_evals": line.get("scanned"),
        }
        binary = {k: line.get(k) for k in ("recall", "recall1", "recall10", "recall100", "recall10_p05",
                                           "recall10_p10", "share_at_0_9", "ties", "resident_share",
                                           "rows_cached_pages", "threads", "seconds") if k in line}
        fields["extra"] = {"recall_source": "binary, with ties" if ties else "binary, ids only", "binary": binary,
                           "tie_free": free, "index_crc": line.get("index_crc"), "commit": line.get("commit"),
                           "dirty": line.get("dirty")}
        return fields

    def cold_binary(self) -> Path:
        """The cold binary, which must be one of this harness."""
        cold = self.binary("cold")

        if not offers(cold, COLD_MARKER):
            raise RuntimeError(f"{cold} is older than this adapter. Build it again with cargo build --release -p "
                               f"coldivf-bench --target-dir target/mine")

        return cold

    def cold_command(self, ctx: ServeContext) -> list[str]:
        """What every cold process of a cell starts with."""
        cmd = [str(self.cold_binary()), "--one", "--index", str(self.file), "--dataset", ctx.dataset.name, "--n",
               str(ctx.rows), "--k", str(ctx.k), "--mode", "wave", "--variant",
               "idle" if ctx.cpu_state == "idle" else "prewarm", *flags(ctx.params, SEARCH_FLAGS, "search setting")]

        # The warm-up query is one that no trial of the cell searches.
        if ctx.warmup_ids:
            cmd += ["--warmup-query", str(ctx.warmup_ids[0])]

        return cmd

    def cold_setup(self, ctx: ServeContext) -> None:
        cold = self.cold_binary()

        with open(cold, "rb") as f:
            f.read()

        self.file = index_file(ctx.index)

        if ctx.regime == "c1":
            self.server = self.start_server(ctx)
        else:
            # A failed start loads the binary and its libraries before the timed process starts.
            subprocess.run([str(cold)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def start_server(self, ctx: ServeContext) -> subprocess.Popen:
        """Starts the cell's C1 process and waits until it has warmed its index."""
        proc = subprocess.Popen(self.cold_command(ctx) + ["--c1"], env=self.env(ctx.dataset.root),
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr)
        self.children.add(proc)
        ready = self.server_line(proc)

        if ready.get("event") != "ready":
            raise RuntimeError(f"cold --c1 said {ready} instead of that it is ready")

        return proc

    def server_line(self, proc: subprocess.Popen) -> dict:
        """The next JSON line of the C1 process `proc`."""
        while True:
            raw = proc.stdout.readline()

            if not raw:
                raise RuntimeError(f"cold --c1 ended with status {proc.wait()}")

            if raw.startswith(b"{"):
                return json.loads(raw)

    def cold_teardown(self, ctx: ServeContext) -> None:
        proc, self.server = self.server, None

        if proc is None:
            return

        try:
            proc.stdin.write(b"stop\n")
            proc.stdin.close()
        except OSError:
            pass

        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

        proc.stdout.close()
        self.children.discard(proc)

    def cold_trial(self, ctx: ServeContext, go: dict) -> dict:
        queries = [int(q) for q in go["queries"]]
        trial, ids = str(int(go.get("trial") or 0)), ",".join(str(q) for q in queries)

        if ctx.regime == "c1":
            self.server.stdin.write(f"--trial {trial} --queries {ids}\n".encode())
            self.server.stdin.flush()
            lines = [self.server_line(self.server) for _ in queries]
        else:
            delay = ctx.delay_ms if ctx.regime == "intent" else 0.0
            cmd = self.cold_command(ctx) + ["--trial", trial, "--queries", ids, "--gap-ms", str(delay)]

            if ctx.dummy is not None:
                cmd += ["--dummy", str(index_file(ctx.dummy))]

            done = self.run_child(cmd, env=self.env(ctx.dataset.root), timeout=600)

            if done.returncode != 0:
                raise RuntimeError(f"cold exited with {done.returncode}")

            lines = [json.loads(x) for x in done.stdout.decode().splitlines() if x.startswith("{")]

        return self.trial_fields(ctx, queries, sorted(lines, key=lambda x: x["seq"]))

    def trial_fields(self, ctx: ServeContext, queries: list[int], lines: list[dict]) -> dict:
        """The cold fields of one trial from the binary's lines, one per query."""
        first = lines[0]
        # C1 times the search alone. Intent times from the vector's arrival, so the open time past the delay counts,
        # as for every other system.
        before = None if ctx.regime == "c1" else first["open_us"] + first.get("read_us", 0) + first["prepare_us"]
        total = first["from_vector_us"] if ctx.regime == "intent" else first["total_us"]
        stage_keys = ("ctx_us", "read_us", "prepare_us", "gap_us", "late_us", "tables_us", "buffers_us", "route_us",
                      "submit_us", "wave_us", "cluster_wave_us", "scan_us", "rows_us", "rerank_us",
                      "post_embedding_us", "cluster_bytes", "cluster_ranges", "row_bytes", "row_ranges",
                      "header_bytes", "table_bytes")
        cached, checked = int(first.get("pages_resident_at_start") or 0), int(first.get("pages_checked") or 0)
        ids = first.get("ids")
        out: dict[str, Any] = {
            "open_us": before,
            "search_us": first["search_us"],
            "total_us": total,
            "next_us": [x["search_us"] for x in lines[1:]],
            "resident_rerun_us": first.get("resident_rerun_us"),
            "ids": [int(i) for i in ids] if isinstance(ids, list) else None,
            "stages": {k: first[k] for k in stage_keys if first.get(k) is not None},
            "disk_bytes": first.get("disk_bytes"),
            "pageins": first.get("pageins"),
            # A C1 process's footprint before the search holds the state it keeps, eg. its cluster tables.
            "footprint_before": first.get("footprint_before"),
            "footprint_delta": int(first["footprint_delta"]) if first.get("footprint_delta") is not None else None,
            "lifetime_max_footprint": first.get("peak_footprint"),
            "p_share": first.get("p_share"),
            "ghz": first.get("ghz"),
            "energy_nj": first.get("energy_nj"),
            "qos": first.get("qos"),
            # The binary counts ties as the driver does, which recomputes both from the ids.
            "recall10_ids_only": first.get("recall") if ctx.k == 10 else None,
            "recall10": first.get("recall10"),
            "extra": {k: first.get(k) for k in ("pages_resident_at_start", "pages_checked", "syscall_open_us",
                                                "open_p_share", "open_ghz", "index_crc", "index_epoch", "load_1",
                                                "rerun_disk_bytes", "dummy_us", "warmup_query")}
            | {"next_recall10": [x.get("recall10") for x in lines[1:]],
               "next_recall_ids_only": [x.get("recall") for x in lines[1:]], "recall_source": "binary, with ties",
               "dummy_searched": first.get("dummy") is not None, "rerun_measured": True},
        }

        if cached > (C1_CACHED * checked if ctx.regime == "c1" else 0):
            out.update(status="rejected", error=f"the binary found {cached} of {checked} pages cached at its timer")

        return out

    def run_overhead(self, ctx: BuildContext, ids: int = 1000, calls: int = 1000) -> dict:
        # The binaries time each search inside their own process, around the library's call, so no binding cost
        # enters their latencies.
        return {"overhead_us": 0.0, "calls": 0, "index_rows": 0,
                "note": "the binaries time each search around the library's own call"}


if __name__ == "__main__":
    main(ColdIVF)
