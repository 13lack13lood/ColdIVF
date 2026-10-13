"""ColdIVF through its Python module, coldivf, in this process, as every other system's adapter runs its library, so its
times hold the module's per-call cost as theirs hold their bindings'. adapters/coldivf.py runs the release binaries
instead, which time each search inside their own process, and suits comparing ColdIVF with itself.

Build the module into the adapter's Python with `maturin develop --release` in coldivf-python. The adapter builds
indexes with the build_index binary, as adapters/coldivf.py does, so both search the same files, and opens and searches
them through the module. A cold trial times Index() and prepare() as the open and then the first search, as the cold
binary times Index::open, prepare and the search. A warm run starts the module's warm-up and waits for it, as the warm
binary's whole mode does.

Writes go through coldivf.Writer: `insert` and `delete` for a batch of 1, `insert_many` and `delete_many` for larger
ones, and `sync` as the sync and the durable point. Each change is one append to the log beside the index, so it
survives the process dying, and `sync` makes it survive a power loss with F_FULLFSYNC. The maintenance actions are the
writer's runs "fold", "auto" and "major", and "job", which runs `begin_compact`, `Job.run` throttled and `Job.finish` on
the adapter's thread. Every run and every status call takes the suite's params, eg. the build's vectors_per_list, and
its threads explicitly, since the module's threads=0 takes RAYON_NUM_THREADS, which the driver sets to the session's
threads. The first run replaces the index file and its log with the folder `index.coldivf.d`, which then holds each
generation's data file and log, the root slots, the record of runs and the activity word. The module still opens the
index by the file's path, though no file is left there.
"""

from __future__ import annotations

import importlib.metadata
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from adapters.base import Adapter, BuildContext, ServeContext, Unsupported, WriteContext, main
from adapters.coldivf import ROUTING, ColdIVF, sections
from adapters.inproc import as_ids
import schema

# The search settings the module takes, and the value of each a setting leaves out. A search needs nprobe and rerank.
SEARCH_KEYS = ("nprobe", "rerank", "reach")

# How long a warm run waits for the warm-up before it marks the index warm anyway, as the warm binary does.
WARM_TIMEOUT_S = 60.0

# What a RunReport and a Status hold, as the module names them.
REPORT_FIELDS = ("kind", "regions_due", "reclustered", "moved", "reencoded", "copied", "count", "lists", "regions",
                 "distances", "bytes_read", "bytes_written", "cpu", "wall", "indicator")
STATUS_FIELDS = ("pending", "deleted", "regions_due", "major_due", "due", "deferred", "bytes_written_24h",
                 "reencoded_24h", "cpu_24h")

# The small files of an index's folder that every open or run reads, besides its generations.
FOLDER_FILES = ("root.0", "root.1", "runs", "activity")


@dataclass
class Handle:
    index: Any
    nprobe: int
    rerank: int
    reach: float | None


@dataclass
class WriteHandle:
    """A write session's writer and its index's path, with the maintenance settings and threads that its status calls
    take, those of the session's last run or else of the policy pass."""

    writer: Any
    path: Path
    params: dict = field(default_factory=dict)
    threads: int = 1


def search_settings(params: dict) -> tuple[int, int, float | None]:
    unknown = sorted(set(params) - set(SEARCH_KEYS))

    if unknown:
        raise Unsupported(f"the coldivf module takes no search setting {unknown}")

    missing = [k for k in ("nprobe", "rerank") if params.get(k) is None]

    if missing:
        raise ValueError(f"a ColdIVF search needs {missing}")

    reach = params.get("reach")
    return int(params["nprobe"]), int(params["rerank"]), None if reach is None else float(reach)


def index_path(path: Path | str) -> Path:
    """The path that names an index, given that path or a folder that holds one index. The index is a .coldivf file
    until its first maintenance run, and from then on the folder `<name>.coldivf.d` beside where the file was, which the
    module opens by the same path."""
    path = Path(path)

    if not path.is_dir():
        return path

    names = {p.name for p in path.glob("*.coldivf") if p.is_file()}
    names |= {p.name[: -len(".d")] for p in path.glob("*.coldivf.d") if p.is_dir()}

    if len(names) != 1:
        raise FileNotFoundError(f"{path} should hold one ColdIVF index, not {len(names)}")

    return path / names.pop()


def side_file(path: Path, kind: str) -> dict:
    """A small file of the index as one range, listed even before it exists, eg. a log a writer makes on first use."""
    n = path.stat().st_size if path.exists() else 0
    return {"path": str(path), "bytes": n, "ranges": [{"offset": 0, "length": n, "kind": kind}] if n else []}


def fields_of(value: Any, names: tuple[str, ...]) -> dict:
    return {name: getattr(value, name) for name in names}


class ColdIVFPython(Adapter):
    system = "coldivf"
    binding = "python"
    # C1 needs the module to drop the cached pages of its own mapping, which it can't yet.
    regimes = ("c0", "intent")
    # A cold search reads in F_RDADVISE waves from the calling thread, so it runs no read threads.
    workers = ""
    write_role = "full"
    durability = {"call": "process", "sync": "power"}
    sync_kind = "incremental"
    maintenance_actions = ("fold", "auto", "major", "job")

    # The builds are those of the binaries' adapter, so both adapters search the same files.
    binary = ColdIVF.binary
    env = ColdIVF.env
    run_build = ColdIVF.run_build

    def __init__(self) -> None:
        super().__init__()
        self.lib: Any = None

    def load(self) -> None:
        import coldivf

        self.lib = coldivf

    def module_file(self) -> Path:
        """The module's compiled file. maturin installs it in a package of the same name."""
        inner = getattr(self.lib, "coldivf", self.lib)
        return Path(inner.__file__)

    def version(self) -> str | None:
        try:
            return f"coldivf {importlib.metadata.version('coldivf')}"
        except importlib.metadata.PackageNotFoundError:
            return None

    def wheels(self) -> dict[str, str | None]:
        build_index = self.binary("build_index")
        return {"coldivf": schema.file_sha256(self.module_file()), "build_index": schema.file_sha256(build_index)}

    def simd_level(self) -> str:
        # The library picks its kernels at run time, which is NEON with dot products on Apple silicon.
        return "neon" if platform.machine() == "arm64" else "runtime"

    def build(self, ctx: BuildContext) -> dict:
        # The index of the per-call cost's measurement, built as every other.
        return ColdIVF.run_build(self, ctx)

    def overhead_params(self) -> dict:
        return {"nprobe": 2, "rerank": 10}

    def data_file(self, index: Path) -> Path:
        """The data file that holds the index now, under the index's path as given. The module returns it with
        symbolic links resolved, eg. /private/var for /var."""
        found = Path(self.lib.data_file(str(index)))
        return index if found.name == index.name else Path(f"{index}.d") / found.name

    def files(self, path: Path) -> list[dict]:
        """The current data file, with its header, rotation, centers and extents as routing and its codes, rows and
        ids as bulk, then its log and the writers' lock. Once maintenance made the index's folder, also its root slots,
        its record of runs, its activity word and anything else there, eg. a job's resume file."""
        index = index_path(path)
        data = self.data_file(index)
        ranges = [{"offset": off, "length": length, "kind": "routing" if name in ROUTING else "bulk"}
                  for name, (off, length) in sections(data).items() if length > 0]
        out = [{"path": str(data), "bytes": data.stat().st_size, "ranges": ranges}]

        if data == index:
            return out + [side_file(Path(f"{index}.delta"), "bulk"), side_file(Path(f"{index}.lock"), "routing")]

        folder, log = data.parent, data.with_suffix(".log")
        out += [side_file(log, "bulk"), side_file(Path(f"{index}.lock"), "routing")]
        out += [side_file(folder / name, "routing") for name in FOLDER_FILES]
        listed = {log.name, data.name, *FOLDER_FILES}
        return out + [side_file(p, "bulk") for p in sorted(folder.iterdir()) if p.is_file() and p.name not in listed]

    def open(self, ctx: ServeContext, path: Path) -> Handle:
        nprobe, rerank, reach = search_settings(ctx.params)
        index = self.lib.Index(str(index_path(path)), max_k=ctx.k, max_nprobe=nprobe, max_rerank=max(rerank, ctx.k))
        index.prepare()
        return Handle(index, nprobe, rerank, reach)

    def configure(self, handle: Handle, params: dict) -> None:
        handle.nprobe, handle.rerank, handle.reach = search_settings(params)

    def search(self, handle: Handle, query: np.ndarray, k: int, params: dict) -> np.ndarray:
        ids, _ = handle.index.search(query, k, nprobe=handle.nprobe, rerank=handle.rerank, reach=handle.reach)
        return as_ids(ids, k)

    def close(self, handle: Handle) -> None:
        handle.index = None

    def warm_up(self, handle: Handle, ctx: ServeContext) -> None:
        # The whole file read through the cache, then the module's warm-up, as the warm binary's whole mode does.
        super().warm_up(handle, ctx)

        if not handle.index.warm_up(wait=True, timeout=WARM_TIMEOUT_S):
            # The hits are the same either way, and the first searches fault their pages in.
            handle.index.set_warm(True)

    def writer_open(self, ctx: WriteContext, path: Path) -> WriteHandle:
        index = index_path(path)
        return WriteHandle(self.lib.Writer(str(index)), index, dict(ctx.maintain_params), ctx.maintain_threads)

    def insert(self, w: WriteHandle, ids: np.ndarray, rows: np.ndarray) -> None:
        if len(ids) == 1:
            w.writer.insert(int(ids[0]), rows[0])
        else:
            w.writer.insert_many(ids, rows)

    def delete(self, w: WriteHandle, ids: np.ndarray) -> None:
        if len(ids) == 1:
            w.writer.delete(int(ids[0]))
        else:
            w.writer.delete_many(ids)

    def sync(self, w: WriteHandle) -> dict:
        w.writer.sync()
        return {"call": "Writer.sync", "level": "F_FULLFSYNC of the log"}

    def maintain(self, w: WriteHandle, action: str, threads: int, params: dict) -> dict:
        """Runs `action` with the suite's params and `threads`, and returns its RunReport's fields. A run commits the
        next generation with every file fully synced, so it is at the power level when it returns. The job's report
        also holds how long its begin, run and finish took."""
        if action not in self.maintenance_actions:
            raise Unsupported(f"the coldivf module has no maintenance action {action}")

        w.params, w.threads = dict(params), threads
        settings = dict(params) | {"threads": threads}
        steps = {}

        if action == "job":
            t0 = time.perf_counter()
            job = w.writer.begin_compact(**settings)

            if job is None:
                raise RuntimeError("begin_compact found no run due, so the job has nothing to run")

            t1 = time.perf_counter()
            job.run()
            t2 = time.perf_counter()
            report = job.finish()
            steps = {"begin_s": t1 - t0, "run_s": t2 - t1, "finish_s": time.perf_counter() - t2}
        else:
            report = w.writer.maintain(action, **settings)

        return fields_of(report, REPORT_FIELDS) | steps | {"rows_live": report.count, "durability": "power",
                                                           "data_file": self.data_file(w.path)}

    def due(self, w: WriteHandle, params: dict) -> bool:
        return bool(w.writer.status(**(dict(params) | {"threads": w.threads})).due)

    def pending(self, w: WriteHandle) -> dict:
        return fields_of(w.writer.status(**(w.params | {"threads": w.threads})), STATUS_FIELDS)

    def writer_close(self, w: WriteHandle) -> None:
        w.writer.close()


if __name__ == "__main__":
    main(ColdIVFPython)
