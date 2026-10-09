"""ColdIVF through its release binaries build_index, warm and cold, from $COLDIVF_BIN or target/mine/release.

The binaries take a dataset by name and find its files through $COLDIVF_DATA. A tune or test split becomes a dataset
of its own under data/splits/, with a link to the base, the split's queries, and the split's ground truth with its
distances, so the binaries see the split's queries in order. The binaries print no ids. warm counts ties when the
distances are there, which gives the same recall the driver computes for every other system. cold reports recall by
ids alone, so a trial's recall10 is filled only for a query whose truth has no tie at rank 10, where the two are equal.

Each cold trial runs `cold --one`, a fresh process that maps the index once, drops its pages through that mapping,
waits for anything that reads them back, and checks them right before its timer. It times Index::open, prepare and the
search of the first query, then searches the next queries without dropping anything. A trial whose own check finds a
cached page comes back rejected. C1, a dummy search in the timed process and a rerun of the first query need modes the
binaries don't have yet. The binaries time each search inside their own process, so their per-call cost is 0.
"""

from __future__ import annotations

import json
import os
import platform
import struct
import subprocess
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
    regimes = ("c0", "intent")
    # A cold search reads in F_RDADVISE waves from the calling thread, so it runs no read threads.
    workers = ""

    def __init__(self) -> None:
        super().__init__()
        self.positions: dict[int, int] = {}
        self.split: tuple[Path, str] | None = None
        self.file: Path | None = None
        self.tie_free: dict[int, bool] = {}

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

    def _tie_free(self, ds: data.Dataset, n: int, qids: list[int]) -> dict[int, bool]:
        recall = data.Recall(ds, ds.truth(n))
        free = recall.tie_free(np.asarray(qids), 10)
        return {int(q): bool(f) for q, f in zip(qids, free)}

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

    def cold_setup(self, ctx: ServeContext) -> None:
        with open(self.binary("cold"), "rb") as f:
            f.read()

        self.split = split_dataset(ctx.dataset, ctx.rows, ctx.split)
        qids = data.split_ids(ctx.dataset.queries.shape[0], ctx.split)
        self.positions = {int(q): i for i, q in enumerate(qids)}
        self.file = index_file(ctx.index)
        self.tie_free = self._tie_free(ctx.dataset, ctx.rows, list(ctx.queries)[:1])
        # A failed start loads the binary and its libraries, the way a dummy search warms an in-process library.
        subprocess.run([str(self.binary("cold"))], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def cold_teardown(self, ctx: ServeContext) -> None:
        pass

    def cold_trial(self, ctx: ServeContext, go: dict) -> dict:
        root, name = self.split
        queries = [int(q) for q in go["queries"]]
        positions = [self.positions[q] for q in queries]
        size = len(self.positions)

        if any((positions[0] + j) % size != p for j, p in enumerate(positions)):
            raise ValueError("a trial's queries must follow each other in the split")

        cmd = [str(self.binary("cold")), "--one", "--index", str(self.file), "--dataset", name, "--n", str(ctx.rows),
               "--k", str(ctx.k), "--first", str(positions[0]), "--trial", "0", "--per-trial", str(len(queries)),
               "--mode", "wave", "--variant", "idle" if ctx.cpu_state == "idle" else "prewarm",
               "--gap-ms", str(int(round(ctx.delay_ms))), *flags(ctx.params, SEARCH_FLAGS, "search setting")]
        done = self.run_child(cmd, env=self.env(root), timeout=600)

        if done.returncode != 0:
            raise RuntimeError(f"cold exited with {done.returncode}")

        lines = sorted((json.loads(x) for x in done.stdout.decode().splitlines() if x.startswith("{")),
                       key=lambda x: x["seq"])
        return self.trial_fields(ctx, queries, lines)

    def trial_fields(self, ctx: ServeContext, queries: list[int], lines: list[dict]) -> dict:
        """The cold fields of one trial from the binary's lines, one per query."""
        first = lines[0]
        before = first["open_us"] + first.get("read_us", 0) + first["prepare_us"]
        total = first["total_us"]
        intent: dict[str, Any] = {}

        if ctx.regime == "intent":
            # The vector is ready delay_ms after the open starts, so open time past that counts, as for every other
            # system. The binary starts its gap only once prepare is done, and settling its counters stretches the gap
            # by a few milliseconds. So its search starts later than the vector would let it, and the cluster tables
            # that prepare starts reading get that much longer in the background.
            delay_us = ctx.delay_ms * 1e3
            total = max(0.0, before - delay_us) + first["search_us"]
            late = before + first.get("gap_us", 0.0) - max(before, delay_us)
            intent = {"intent_search_late_us": late, "intent_comparable": late <= 0}

        stage_keys = ("ctx_us", "read_us", "prepare_us", "gap_us", "tables_us", "buffers_us", "route_us", "submit_us",
                      "wave_us", "cluster_wave_us", "scan_us", "rows_us", "rerank_us", "post_embedding_us",
                      "cluster_bytes", "cluster_ranges", "row_bytes", "row_ranges")
        cached = int(first.get("pages_resident_at_start") or 0)
        q0 = queries[0]
        out: dict[str, Any] = {
            "open_us": before,
            "search_us": first["search_us"],
            "total_us": total,
            "next_us": [x["search_us"] for x in lines[1:]],
            "resident_rerun_us": None,
            "stages": {k: first[k] for k in stage_keys if k in first},
            "disk_bytes": first.get("disk_bytes"),
            "pageins": first.get("pageins"),
            "footprint_delta": int(first["footprint_delta"]) if first.get("footprint_delta") is not None else None,
            "p_share": first.get("p_share"),
            "ghz": first.get("ghz"),
            "energy_nj": first.get("energy_nj"),
            "qos": first.get("qos"),
            "recall10_ids_only": first.get("recall") if ctx.k == 10 else None,
            "recall10": first.get("recall") if ctx.k == 10 and self.tie_free.get(q0) else None,
            "extra": {k: first.get(k) for k in ("pages_resident_at_start", "syscall_open_us", "open_p_share",
                                                "open_ghz", "index_crc", "index_epoch", "load_1")}
            | {"next_recall_ids_only": [x.get("recall") for x in lines[1:]], "recall_source": "binary, ids only",
               # The binary searches no dummy index in its timed process and doesn't rerun the first query.
               "dummy_searched": False, "rerun_measured": False} | intent,
        }

        if isinstance(first.get("ids"), list):
            out["ids"] = first["ids"]

        if cached:
            out.update(status="rejected", error=f"the binary found {cached} pages cached at its timer")

        return out

    def run_overhead(self, ctx: BuildContext, ids: int = 1000, calls: int = 1000) -> dict:
        # The binaries time each search inside their own process, around the library's call, so no binding cost
        # enters their latencies.
        return {"overhead_us": 0.0, "calls": 0, "index_rows": 0,
                "note": "the binaries time each search around the library's own call"}


if __name__ == "__main__":
    main(ColdIVF)
