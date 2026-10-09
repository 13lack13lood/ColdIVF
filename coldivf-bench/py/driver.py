"""Runs a benchmark suite through each system's adapter and appends the results.

    python driver.py SUITE.toml [--only build,warm,cold] [--systems a,b] [--gate enforce|warn|off] [--dry-run]

A batch checks the timing gate and the disk floor, runs the drift references, builds what isn't built, measures each
system's per-call cost and runs the warm grids, then the cold trials in rounds of randomly ordered cells, runs the
references again and writes a batch summary.
Every adapter is a child process of this one, started with the suite's interpreter and never in a new session, so a
memory cap on this process group covers it. Lines go to <results>/<phase>/<system>.jsonl, and a rerun skips every run
key that is already done. README.md describes the suite file.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import random
import re
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

PY = Path(__file__).resolve().parent
sys.path.insert(0, str(PY))

import data  # noqa: E402
import evictor  # noqa: E402
import rusage  # noqa: E402
import schema  # noqa: E402

REPO = PY.parents[1]
GIB = 1 << 30

SUITE_KEYS = {"phase", "dataset", "n", "index_root", "results_root", "seed", "k", "truth", "gate", "refs", "system",
              "max_attempts", "wait_s"}
GATE_KEYS = {"mode", "cpu_idle_pct", "top_samples", "top_interval_s", "disk_floor_gb", "max_wait_s", "recheck_s",
             "require_ac", "require_caffeinate", "busy_names", "agents", "invalid_vm", "locks"}
REFS_KEYS = {"mode", "wave_trials", "wave_ranges", "wave_page", "wave_file", "wave_size_mb", "coldivf_index",
             "coldivf_dataset", "coldivf_n", "coldivf_params", "coldivf_trials", "drift_pct"}
SYSTEM_KEYS = {"name", "adapter", "python", "variant", "env", "build", "warm", "cold"}
BUILD_KEYS = {"config_id", "config", "seeds", "threads", "disk_gb", "timeout_s", "index", "clusters", "n"}
WARM_KEYS = {"config_id", "seed", "split", "threads", "mode", "grid", "settings", "passes", "warmup", "k",
             "qps_seconds", "qps_repeats", "timeout_s"}
COLD_KEYS = {"config_id", "seed", "regime", "params", "trials", "split", "cpu_state", "next", "delay_ms", "threads",
             "group", "dummy", "dummy_n", "k", "self_evict", "fresh_clone", "timeout_s", "ready_timeout_s"}

DEFAULT_GATE = {
    "mode": "enforce", "cpu_idle_pct": 85.0, "top_samples": 3, "top_interval_s": 10, "disk_floor_gb": 40.0,
    "max_wait_s": 3600, "recheck_s": 300, "require_ac": True, "require_caffeinate": True,
    "busy_names": ["cargo", "rustc"],
    # The folder where the machine's job gate keeps its lock files. Empty means $COLDIVF_BENCH_LOCKS.
    "locks": "",
    # The machine counters whose change during a timed part makes the point invalid.
    "invalid_vm": ["swapins", "compressions"],
    "agents": {"falcon": ["crowdstrike", "Falcon.app"], "fortidlp": ["FortiDLP"], "ava": ["uk.ava.reveal", "/Ava/"],
               "jamf": ["JAMF", "Jamf"]},
}
DEFAULT_REFS = {
    "mode": "off", "wave_trials": 30, "wave_ranges": 64, "wave_page": 16384,
    "wave_file": "data/bench.noindex/wave-1g.bin",
    "wave_size_mb": 1024, "coldivf_index": "", "coldivf_dataset": "", "coldivf_n": 0, "coldivf_params": {},
    "coldivf_trials": 30, "drift_pct": 10.0,
}

# A temporary file of a build, named <file>.<pid>.<count>.<kind>, which a killed build leaves behind.
STALE = re.compile(r"\.(\d+)\.(\d+)\.(spill|groups|tmp)$")

# How many warm-up queries a cold process gets, eg. for LanceDB's search for the size of its index cache.
WARMUP_QUERIES = 8
# The rows of the index an overhead subcommand builds, and how long it may take.
OVERHEAD_ROWS = 1000
OVERHEAD_TIMEOUT_S = 900.0
# What the batch line keeps of each system's info event, eg. which threads it runs and at which QoS class.
INFO_KEYS = ("system", "system_version", "binding", "regimes", "batch_api", "qps_mode", "workers", "worker_qos")
# How often a build's folder is measured, in seconds, so files that live a fraction of a second are still seen.
DISK_SAMPLE_S = 0.02


class SuiteError(ValueError):
    pass


def _check_keys(table: dict, allowed: set[str], where: str) -> None:
    unknown = sorted(set(table) - allowed)

    if unknown:
        raise SuiteError(f"{where} has unknown keys {unknown}")


def _repo_path(value: str | None) -> Path | None:
    if not value:
        return None

    path = Path(os.path.expanduser(os.path.expandvars(value)))
    return path if path.is_absolute() else REPO / path


def _slug(text: str) -> str:
    """`text` with every run of characters other than letters, digits, "_", "." and "-" made one "_"."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def expand_grid(grid: dict | None, settings: list | None) -> list[dict]:
    """Every setting of a warm entry. They are the product of the grid's lists in the order given, then the explicit
    settings."""
    out: list[dict] = []

    if grid:
        keys = list(grid)
        values = [v if isinstance(v, list) else [v] for v in grid.values()]
        out += [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    out += [dict(s) for s in settings or []]
    return out


@dataclass
class Build:
    system: System
    config_id: str
    config: dict
    seed: int
    n: int
    threads: int = 10
    disk_gb: float = 0.0
    timeout_s: float = 7200.0
    prebuilt: Path | None = None
    clusters: Path | None = None
    out: Path | None = None

    @property
    def index(self) -> Path:
        return self.prebuilt if self.prebuilt is not None else self.out


@dataclass
class Warm:
    system: System
    build: Build
    split: str
    threads: int
    mode: str
    settings: list[dict]
    passes: int = 3
    warmup: int = 1000
    k: int = 10
    qps_seconds: float = 10.0
    qps_repeats: int = 3
    timeout_s: float = 7200.0


@dataclass
class Cell:
    system: System
    build: Build
    regime: str
    params: dict
    trials: int
    split: str = "test"
    cpu_state: str = "prewarm"
    next: int = 9
    delay_ms: float = 0.0
    threads: int = 1
    group: str = "main"
    dummy: Build | Path | None = None
    k: int = 10
    self_evict: bool = False
    # Runs each trial on a new copy-on-write clone of the index, a file no cache has seen.
    fresh_clone: bool = False
    timeout_s: float = 600.0
    ready_timeout_s: float = 600.0
    # Why the cell can't run in this batch, the status its one line takes, and whether that line is written.
    skip: str | None = None
    skip_status: str = "failed"
    reported: bool = False
    # The index's part of the cell's run keys, taken once before any process of the cell opens the index.
    index_key: str | None = None

    @property
    def label(self) -> str:
        return ":".join(x for x in (self.system.name, self.system.variant, self.build.config_id, self.regime) if x)

    @property
    def dummy_path(self) -> Path | None:
        return self.dummy.index if isinstance(self.dummy, Build) else self.dummy


@dataclass
class System:
    name: str
    adapter: str
    python: str
    variant: str = ""
    env: dict = field(default_factory=dict)
    builds: list[Build] = field(default_factory=list)
    warm: list[Warm] = field(default_factory=list)
    cold: list[Cell] = field(default_factory=list)

    @property
    def module(self) -> str:
        return self.adapter if "." in self.adapter else f"adapters.{self.adapter}"


@dataclass
class Suite:
    path: Path
    sha256: str
    phase: str
    dataset: str
    n: int
    index_root: Path
    results_root: Path
    seed: int
    k: int
    truth: Path | None
    gate: dict
    refs: dict
    systems: list[System]
    max_attempts: int = 3
    wait_s: float = 1.0

    @property
    def builds(self) -> list[Build]:
        out: list[Build] = []

        for s in self.systems:
            out += s.builds

            for c in s.cold:
                if isinstance(c.dummy, Build) and c.dummy not in out:
                    out.append(c.dummy)

        return out

    @property
    def warm(self) -> list[Warm]:
        return [w for s in self.systems for w in s.warm]

    @property
    def cells(self) -> list[Cell]:
        return [c for s in self.systems for c in s.cold]


def _find_build(system: System, config_id: str, seed: int | None, where: str) -> Build:
    for b in system.builds:
        if b.config_id == config_id and (seed is None or b.seed == seed):
            return b

    raise SuiteError(f"{where} names build {config_id} (seed {seed}), which {system.name} doesn't have")


def load_suite(path: Path | str) -> Suite:
    """Reads and checks a suite file. Unknown keys are errors, so a typo never runs the default."""
    path = Path(path)
    raw = path.read_bytes()
    doc = tomllib.loads(raw.decode())
    _check_keys(doc, SUITE_KEYS, "the suite")

    for key in ("phase", "dataset", "n"):
        if key not in doc:
            raise SuiteError(f"the suite needs {key}")

    gate = DEFAULT_GATE | doc.get("gate", {})
    _check_keys(gate, GATE_KEYS, "[gate]")
    refs = DEFAULT_REFS | doc.get("refs", {})
    _check_keys(refs, REFS_KEYS, "[refs]")
    phase, n, k = str(doc["phase"]), int(doc["n"]), int(doc.get("k", 10))
    # A folder whose name ends in .noindex keeps Spotlight from reading the first pages of every new file.
    index_root = _repo_path(doc.get("index_root", f"data/bench.noindex/{phase}"))
    suite = Suite(path=path, sha256=hashlib.sha256(raw).hexdigest(), phase=phase, dataset=str(doc["dataset"]),
                  n=n, index_root=index_root, results_root=_repo_path(doc.get("results_root", "results/bench")),
                  seed=int(doc.get("seed", 1)), k=k, truth=_repo_path(doc.get("truth")), gate=gate, refs=refs,
                  systems=[], max_attempts=int(doc.get("max_attempts", 3)), wait_s=float(doc.get("wait_s", 1.0)))
    names = set()

    for entry in doc.get("system", []):
        _check_keys(entry, SYSTEM_KEYS, "a [[system]]")
        name, variant = str(entry["name"]), str(entry.get("variant", ""))

        if (name, variant) in names:
            raise SuiteError(f"system {name} with variant {variant!r} appears twice")

        names.add((name, variant))
        python = os.path.expanduser(os.path.expandvars(entry.get("python") or sys.executable))
        system = System(name=name, adapter=str(entry.get("adapter", name)), python=python, variant=variant,
                        env={str(a): str(b) for a, b in entry.get("env", {}).items()})
        suite.systems.append(system)

        for b in entry.get("build", []):
            _check_keys(b, BUILD_KEYS, f"a build of {name}")
            bn = int(b.get("n", n))

            for seed in b.get("seeds", [1]):
                build = Build(system=system, config_id=str(b["config_id"]), config=dict(b.get("config", {})),
                              seed=int(seed), n=bn, threads=int(b.get("threads", 10)),
                              disk_gb=float(b.get("disk_gb", 0)), timeout_s=float(b.get("timeout_s", 7200)),
                              prebuilt=_repo_path(b.get("index")), clusters=_repo_path(b.get("clusters")))
                folder = _slug(name + (f"-{variant}" if variant else ""))
                build.out = index_root / folder / _slug(f"{build.config_id}-n{bn}-s{build.seed}")
                system.builds.append(build)

        for w in entry.get("warm", []):
            _check_keys(w, WARM_KEYS, f"a warm entry of {name}")
            build = _find_build(system, str(w["config_id"]), w.get("seed"), f"a warm entry of {name}")
            settings = expand_grid(w.get("grid"), w.get("settings"))

            if not settings:
                raise SuiteError(f"a warm entry of {name} has no settings")

            system.warm.append(Warm(system=system, build=build, split=str(w.get("split", "tune")),
                                    threads=int(w.get("threads", 1)), mode=str(w.get("mode", "whole")),
                                    settings=settings, passes=int(w.get("passes", 3)),
                                    warmup=int(w.get("warmup", 1000)), k=int(w.get("k", k)),
                                    qps_seconds=float(w.get("qps_seconds", 10)),
                                    qps_repeats=int(w.get("qps_repeats", 3)),
                                    timeout_s=float(w.get("timeout_s", 7200))))

        for c in entry.get("cold", []):
            _check_keys(c, COLD_KEYS, f"a cold entry of {name}")
            build = _find_build(system, str(c["config_id"]), c.get("seed"), f"a cold entry of {name}")
            regime = str(c.get("regime", "c0"))

            if regime not in ("c0", "c1", "intent"):
                raise SuiteError(f"a cold entry of {name} has regime {regime}, not c0, c1 or intent")

            dummy: Build | Path | None = _repo_path(c.get("dummy"))

            if dummy is None and int(c.get("dummy_n", 0)) > 0:
                dummy = Build(system=system, config_id=build.config_id, config=build.config, seed=build.seed,
                              n=int(c["dummy_n"]), threads=build.threads, disk_gb=0.0, timeout_s=build.timeout_s)
                dummy.out = build.out.parent / f"{build.out.name}-dummy{dummy.n}"

            system.cold.append(Cell(system=system, build=build, regime=regime, params=dict(c.get("params", {})),
                                    trials=int(c.get("trials", 30)), split=str(c.get("split", "test")),
                                    cpu_state=str(c.get("cpu_state", "prewarm")),
                                    next=int(c.get("next", 9 if regime != "c1" else 0)),
                                    delay_ms=float(c.get("delay_ms", 0)), threads=int(c.get("threads", 1)),
                                    group=str(c.get("group", "main")), dummy=dummy, k=int(c.get("k", k)),
                                    self_evict=bool(c.get("self_evict", False)),
                                    fresh_clone=bool(c.get("fresh_clone", False)),
                                    timeout_s=float(c.get("timeout_s", 600)),
                                    ready_timeout_s=float(c.get("ready_timeout_s", 600))))

    _check_groups(suite)
    return suite


def _check_groups(suite: Suite) -> None:
    """A C1 process keeps parts of its index cached, which a C0 trial of the same index in the same group would drop,
    so a group never mixes the two on one index."""
    for group in {c.group for c in suite.cells}:
        cells = [c for c in suite.cells if c.group == group]
        c1 = {str(c.build.index) for c in cells if c.regime == "c1"}
        other = {str(c.build.index) for c in cells if c.regime != "c1"}

        if c1 & other:
            raise SuiteError(f"group {group} runs C1 and another regime on the same index {sorted(c1 & other)[0]}")


def parse_top(text: str) -> list[float]:
    """The idle percentages of each sample of `top -l`, eg. from "CPU usage: 3.1% user, 5.2% sys, 91.6% idle"."""
    return [float(m) for m in re.findall(r"CPU usage:.*?([\d.]+)% idle", text)]


def cpu_idle(samples: int, interval: float) -> float | None:
    """The mean idle CPU share over `top`'s samples after the first, whose figure covers the time before it ran."""
    try:
        out = subprocess.run(["top", "-l", str(max(2, samples)), "-s", str(max(1, round(interval))), "-n", "0"],
                             capture_output=True, text=True, timeout=samples * interval + 60).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None

    idle = parse_top(out)[1:]
    return float(np.mean(idle)) if idle else None


def processes() -> list[tuple[int, int, str]]:
    """Every process as (pid, parent pid, command line)."""
    try:
        out = subprocess.run(["ps", "-axo", "pid=,ppid=,args="], capture_output=True, text=True, errors="replace",
                             timeout=30).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []

    rows = []

    for line in out.splitlines():
        parts = line.split(None, 2)

        if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
            rows.append((int(parts[0]), int(parts[1]), parts[2]))

    return rows


def ancestors(rows: list[tuple[int, int, str]], pid: int) -> set[int]:
    parent = {p: pp for p, pp, _ in rows}
    out, at = set(), pid

    while at in parent and at not in out and at > 1:
        out.add(at)
        at = parent[at]

    return out


def caffeinated(rows: list[tuple[int, int, str]]) -> bool:
    """Whether a caffeinate process holds off idle sleep, eg. `caffeinate -i` or `caffeinate -di`."""
    for _, _, args in rows:
        words = args.split()

        if words and os.path.basename(words[0]) == "caffeinate":
            if any(w.startswith("-") and not w.startswith("--") and "i" in w for w in words[1:]):
                return True

    return False


def busy(rows: list[tuple[int, int, str]], names: list[str], mine: set[int]) -> list[str]:
    """Processes of the listed names that aren't this process's own ancestors."""
    out = []

    for pid, _, args in rows:
        words = args.split()

        if words and os.path.basename(words[0]) in names and pid not in mine:
            out.append(f"{pid} {args[:80]}")

    return out


def holders(paths: list[Path]) -> set[int]:
    """The pids that hold any of `paths` open, from lsof."""
    if not paths:
        return set()

    try:
        out = subprocess.run(["lsof", "-t", "--", *map(str, paths)], capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.TimeoutExpired):
        return set()

    return {int(x) for x in out.split() if x.isdigit()}


def locks_folder(cfg: dict) -> Path | None:
    """The folder of the machine's job gate, from the gate's "locks" key or else $COLDIVF_BENCH_LOCKS, or None."""
    value = str(cfg.get("locks") or os.environ.get("COLDIVF_BENCH_LOCKS") or "")
    return Path(os.path.expanduser(os.path.expandvars(value))) if value else None


def gate_jobs(rows: list[tuple[int, int, str]], mine: set[int],
              locks: Path | None = None) -> tuple[bool, float | None, list[str]]:
    """What the machine's job gate says, read from its folder `locks`. That folder holds a SOLO file with the process
    group and memory cap of the job that holds every slot, and one lock file per slot, which each running job holds
    open. It returns whether this process runs as the solo job, that job's cap, and the other jobs that hold a slot.
    A job waiting for a slot rewrites the slot files, so their text can't name the holder, but the open files can.
    Without the folder no job can be seen, so this process doesn't run as the solo job either."""
    if locks is None or not locks.is_dir():
        return False, None, []

    solo, cap = False, None

    try:
        pgid, value = Path(locks, "SOLO").read_text().split()[:2]
        solo, cap = int(pgid) == os.getpgrp() or int(pgid) in mine, float(value)
    except (OSError, ValueError):
        pass

    args = {pid: text for pid, _, text in rows}
    held = holders(sorted(Path(locks).glob("slot*")))
    return solo, cap, [f"{pid} {args.get(pid, '')[:100]}" for pid in sorted(held - mine)]


def check_gate(cfg: dict) -> dict:
    """The timing gate. Timing needs no other heavy job, no cargo or rustc running, at least the set share of idle
    CPU, AC power with low-power mode off, and caffeinate holding off idle sleep."""
    rows = processes()
    mine = ancestors(rows, os.getpid())
    locks = locks_folder(cfg)
    seen = locks is not None and locks.is_dir()
    solo, cap, others = gate_jobs(rows, mine, locks)
    power = schema.power()
    idle = cpu_idle(int(cfg["top_samples"]), float(cfg["top_interval_s"])) if cfg["mode"] != "off" else None
    problems = [f"another heavy job: {o}" for o in others]
    problems += [f"busy: {b}" for b in busy(rows, list(cfg["busy_names"]), mine)]

    if not seen and cfg["mode"] == "enforce":
        # Without the job gate's folder no other heavy job can show up, so the gate can't vouch for a quiet machine.
        where = f"its lock folder {locks} isn't there" if locks is not None else "no lock folder is set"
        problems.append(f"can't see the job gate, since {where}. Set the gate's locks key or COLDIVF_BENCH_LOCKS")

    if idle is not None and idle < float(cfg["cpu_idle_pct"]):
        problems.append(f"CPU only {idle:.1f}% idle")
    elif idle is None and cfg["mode"] == "enforce":
        problems.append("the idle CPU share couldn't be measured")

    if cfg["require_ac"] and power.get("source") not in (None, "AC Power"):
        problems.append(f"on {power.get('source')}")

    if cfg["require_ac"] and power.get("low_power"):
        problems.append("low-power mode is on")

    if cfg["require_caffeinate"] and not caffeinated(rows):
        problems.append("caffeinate isn't running")

    return {
        "mode": cfg["mode"], "solo": solo, "mem_gb": cap, "threads": _int(os.environ.get("OMP_NUM_THREADS")),
        "locks": str(locks) if locks is not None else None, "locks_seen": seen, "cpu_idle_pct": idle,
        "power": power.get("source"), "low_power": power.get("low_power"), "caffeinate": caffeinated(rows),
        "problems": problems, "passed": not problems,
    }


def _int(value: str | None) -> int | None:
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def free_gib(path: Path) -> float:
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / GIB


def _cputime(text: str) -> float:
    """ps's cumulative CPU time, eg. "345:27.44" or "1-02:03:04", in seconds."""
    days, _, rest = text.rpartition("-")
    parts = [float(p) for p in rest.split(":")]
    seconds = sum(p * 60 ** i for i, p in enumerate(reversed(parts)))
    return seconds + (float(days) * 86400 if days else 0.0)


def agents_snapshot(patterns: dict[str, list[str]]) -> dict[str, float]:
    """The cumulative CPU seconds of each group of endpoint security agents, eg. Falcon and FortiDLP."""
    try:
        out = subprocess.run(["ps", "-axo", "time=,args="], capture_output=True, text=True, errors="replace",
                             timeout=30).stdout
    except (OSError, subprocess.TimeoutExpired):
        return {}

    totals = dict.fromkeys(patterns, 0.0)

    for line in out.splitlines():
        cpu, _, args = line.strip().partition(" ")

        for name, pats in patterns.items():
            if any(p in args for p in pats):
                try:
                    totals[name] += _cputime(cpu)
                except ValueError:
                    pass

                break

    return totals


def agents_delta(before: dict[str, float], after: dict[str, float], seconds: float) -> dict:
    return {
        "window_s": round(seconds, 3),
        "cpu_ms": {k: round(1000 * max(0.0, after.get(k, 0) - v), 1) for k, v in before.items()},
    }


def identity(path: Path, checksum: bool = True) -> dict:
    """What a line records about an index. That is its size on disk, files, newest change, inode and a checksum of
    every file, which is computed once and kept in <path>.identity.json beside the index."""
    path = Path(path)
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    stats = [p.stat() for p in files]
    st = path.stat()
    record = {
        "path": str(path), "bytes": sum(s.st_size for s in stats), "alloc_bytes": sum(s.st_blocks * 512 for s in stats),
        "files": len(files), "mtime_ns": max([s.st_mtime_ns for s in stats] or [st.st_mtime_ns]), "inode": st.st_ino,
        "checksum": None,
    }

    if not checksum:
        return record

    side = path.with_name(path.name + ".identity.json")
    key = {k: record[k] for k in ("bytes", "files", "mtime_ns", "inode")}

    try:
        kept = json.loads(side.read_text())

        if {k: kept.get(k) for k in key} == key and kept.get("checksum"):
            return record | {"checksum": kept["checksum"]}
    except (OSError, ValueError):
        pass

    h = hashlib.sha256()

    for p in files:
        h.update(str(p.relative_to(path) if p != path else p.name).encode() + b"\0")
        h.update(schema.file_sha256(p).encode())

    record["checksum"] = h.hexdigest()

    try:
        side.write_text(json.dumps(record | {"computed": schema.utc_now()}))
    except OSError:
        pass

    return record


def disk_use(path: Path) -> tuple[int, set[str]]:
    """The allocated bytes and the file names under `path`."""
    total, names = 0, set()

    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.lstat(os.path.join(root, f)).st_blocks * 512
                names.add(os.path.join(root, f))
            except OSError:
                pass

    return total, names


class DiskSampler:
    """Measures a build's folder on a thread of its own, every `interval` seconds until `stop`, for the most disk the
    folder used and every file that was ever in it. So a temporary file that lives a fraction of a second still counts
    in a build's peak disk use and its temporary files."""

    def __init__(self, path: Path, interval: float):
        self.path, self.interval = path, interval
        self.peak, self.seen = 0, set()
        self._done = threading.Event()
        self._thread = threading.Thread(target=self._run, name="disk-sampler", daemon=True)
        self._thread.start()

    def sample(self) -> None:
        used, names = disk_use(self.path)
        self.peak, self.seen = max(self.peak, used), self.seen | names

    def _run(self) -> None:
        self.sample()

        while not self._done.wait(self.interval):
            self.sample()

    def stop(self) -> tuple[int, set[str]]:
        """Stops the thread and takes one last sample, and returns the peak bytes and every file seen."""
        self._done.set()
        self._thread.join()
        self.sample()
        return self.peak, self.seen


def remove_stale(root: Path) -> list[str]:
    """Removes temporary build files under `root` whose writer process is gone, and the index clones a stopped batch
    left behind, and returns their paths."""
    removed = []

    if not root.exists():
        return removed

    for folder in sorted((root / "clones.noindex").glob("*-*")):
        owner = folder.name.split("-")[0]

        if owner.isdigit() and not _alive(int(owner)):
            shutil.rmtree(folder, ignore_errors=True)
            removed.append(str(folder))

    for p in root.rglob("*"):
        m = STALE.search(p.name)

        if not m or not p.is_file():
            continue

        if _alive(int(m.group(1))):
            continue

        p.unlink()
        removed.append(str(p))

    return removed


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True

    return True


def touched_files(residency: evictor.Residency, files_read: dict | None, paths: list[str]) -> int:
    """How many of an index's files a trial touched. That is every file with a page cached after the trial, and every
    file of the index that the adapter says it read bytes of, by path, eg. through a handle that skips the page
    cache."""
    known = set(residency.totals) | {str(Path(p)) for p in paths}
    cached = {p for p, pages in residency.pages.items() if pages > 0}
    read = {str(Path(p)) for p, n in (files_read or {}).items() if n} & known
    return len(cached | read)


def moved_files(files: list[dict], src: Path, dst: Path) -> list[dict]:
    """An index's file list with every path under `src` moved under `dst`, eg. for a clone of the index."""
    return [f | {"path": str(f["path"]).replace(str(src), str(dst), 1)} for f in files]


class Child:
    """One adapter process. It reads events from the child's stdout and sends commands to its stdin. The child's
    stderr goes to a log file, so it never fills a pipe."""

    def __init__(self, cmd: list[str], env: dict, log: Path):
        log.parent.mkdir(parents=True, exist_ok=True)
        self.log_path = log
        self.log = open(log, "ab")
        self.log.write(f"\n== {schema.utc_now()} {' '.join(cmd)[:2000]}\n".encode())
        self.log.flush()
        self.cmd = cmd
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, env=env,
                                     cwd=PY)
        self._buf = b""

    def event(self, timeout: float) -> dict | None:
        """The next event, or None when the child ends or the time runs out."""
        deadline = time.monotonic() + timeout
        fd = self.proc.stdout.fileno()

        while True:
            while b"\n" in self._buf:
                raw, self._buf = self._buf.split(b"\n", 1)

                try:
                    msg = json.loads(raw)
                except ValueError:
                    continue

                if isinstance(msg, dict) and "event" in msg:
                    return msg

            left = deadline - time.monotonic()

            if left <= 0:
                return None

            ready, _, _ = select.select([fd], [], [], left)

            if not ready:
                return None

            chunk = os.read(fd, 1 << 16)

            if not chunk:
                return None

            self._buf += chunk

    def expect(self, timeout: float, *events: str) -> dict | None:
        """The next event among `events`, or an "error" or "unsupported" event, or None."""
        while (msg := self.event(timeout)) is not None:
            if msg["event"] in events or msg["event"] in ("error", "unsupported"):
                return msg

        return None

    def send(self, cmd: dict) -> None:
        try:
            self.proc.stdin.write((json.dumps(cmd) + "\n").encode())
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError):
            pass

    def close(self, timeout: float = 30) -> int | None:
        if self.proc.poll() is None:
            self.send({"cmd": "stop"})

        try:
            self.proc.stdin.close()
        except OSError:
            pass

        try:
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.kill()

        self.proc.stdout.close()
        self.log.close()
        return self.proc.returncode

    def kill(self) -> None:
        for sig, wait in ((signal.SIGTERM, 5), (signal.SIGKILL, 5)):
            if self.proc.poll() is not None:
                return

            try:
                self.proc.send_signal(sig)
                self.proc.wait(timeout=wait)
            except (OSError, subprocess.TimeoutExpired):
                pass

    def tail(self, lines: int = 20) -> str:
        try:
            text = self.log_path.read_bytes()[-8192:].decode(errors="replace")
        except OSError:
            return ""

        return "\n".join(text.splitlines()[-lines:])


class Driver:
    def __init__(self, suite: Suite, only: set[str] | None = None, systems: set[str] | None = None,
                 gate_mode: str | None = None, dry_run: bool = False, max_attempts: int | None = None):
        self.suite = suite
        self.only = only or {"build", "warm", "cold", "refs"}
        self.systems = systems
        self.dry_run = dry_run
        self.max_attempts = max_attempts or suite.max_attempts

        if gate_mode:
            suite.gate["mode"] = gate_mode

        self.batch_id = f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{suite.path.stem}-{os.urandom(2).hex()}"
        self.out_dir = suite.results_root / suite.phase
        self.log_dir = self.out_dir / "logs" / self.batch_id
        self.progress = schema.Progress(self.out_dir.glob("*.jsonl"), self.max_attempts)
        self.rng = random.Random(suite.seed)
        self.counts: dict[str, dict[str, int]] = {}
        self.resumed = 0
        self.gate: dict = {}
        self.git: dict = {}
        self.infos: dict[str, dict] = {}
        self.files_cache: dict[str, list[dict]] = {}
        self.identities: dict[str, dict] = {}
        self.open_us: float | None = None
        self.drift: float | None = None
        # The cell of the cold trial that ran last, which every cold line names as its predecessor.
        self.previous: str | None = None
        self.first_open_us: float | None = None
        self.dataset = data.Dataset.named(suite.dataset)
        self._recalls: dict[int, data.Recall | None] = {}
        # Each system's per-call cost on a small index, measured once per batch, by system, variant and interpreter.
        self.overheads: dict[str, dict] = {}

    def selected(self, system: System) -> bool:
        return self.systems is None or system.name in self.systems

    def log(self, msg: str) -> None:
        print(f"driver {time.strftime('%H:%M:%S')} {msg}", file=sys.stderr, flush=True)

    def recall_for(self, n: int) -> data.Recall | None:
        """Scores results of a build of the first `n` rows against the truth of that prefix. A build of the suite's n
        uses the suite's truth file. It is None when the prefix has no truth file, so its lines carry no recall."""
        if n not in self._recalls:
            path = self.suite.truth if n == self.suite.n else None
            truth_path = path or self.dataset.truth_path(min(n, self.dataset.count))

            if not Path(truth_path).exists():
                self.log(f"no ground truth at {truth_path}, so builds of {n} rows get no recall")
                self._recalls[n] = None
            else:
                self._recalls[n] = data.Recall(self.dataset, self.dataset.truth(n, path))

        return self._recalls[n]

    def truth_record(self, n: int) -> dict | None:
        recall = self.recall_for(n)
        return recall.truth.record() if recall is not None else None

    def env(self, system: System, threads: int) -> dict:
        env = dict(os.environ)
        env.update({k: str(threads) for k in ("OMP_NUM_THREADS", "RAYON_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                                              "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
        env["PYTHONPATH"] = str(PY) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env.update(system.env)
        return env

    def child(self, system: System, args: list[str], threads: int = 1) -> Child:
        cmd = [system.python, "-m", system.module, *args]
        return Child(cmd, self.env(system, threads), self.log_dir / f"{system.name}.log")

    def data_args(self, n: int, threads: int, system: System, k: int | None = None) -> list[str]:
        args = ["--data", str(self.dataset.base_path), "--rows", str(n), "--metric", self.dataset.metric,
                "--threads", str(threads), "--variant", system.variant]
        return args + (["--k", str(k)] if k is not None else [])

    def info(self, system: System) -> dict:
        """The adapter's info event for the system's variant, since a variant can change eg. its regimes."""
        key = f"{system.python}|{system.module}|{system.variant}"

        if key not in self.infos:
            child = self.child(system, ["info", "--variant", system.variant])
            msg = child.expect(300, "info")
            child.close()
            self.infos[key] = msg if msg and msg["event"] == "info" else {}

        return self.infos[key]

    def files(self, system: System, index: Path) -> list[dict]:
        """The index's files as the system's variant reads them, since a variant can read eg. a whole file at open."""
        key = f"{system.module}|{system.variant}|{index}"

        if key not in self.files_cache:
            child = self.child(system, ["files", "--index", str(index), "--variant", system.variant])
            msg = child.expect(300, "files")
            child.close()

            if not msg or msg["event"] != "files":
                raise RuntimeError(f"{system.name} couldn't list the files of {index}: {msg or child.tail()}")

            self.files_cache[key] = msg["files"]

        return self.files_cache[key]

    def index_identity(self, build: Build) -> dict | None:
        key = str(build.index)

        if key not in self.identities and build.index is not None and build.index.exists():
            self.identities[key] = identity(build.index)

        return self.identities.get(key)

    def base_line(self, kind: str, system: System | None, build: Build | None, key: str, attempt: int,
                  threads: int = 1) -> dict:
        line = schema.blank(kind)
        info = self.info(system) if system is not None else {}
        line.update(
            run_id=schema.new_run_id(), run_key=key, attempt=attempt, ts=schema.utc_now(), batch_id=self.batch_id,
            phase=self.suite.phase, system=system.name if system else "driver",
            variant=system.variant if system else "",
            system_version=info.get("system_version"), venv=info.get("venv"), binding=info.get("binding"),
            simd_level=info.get("simd_level"), dataset=self.suite.dataset, metric=self.dataset.metric,
            n=build.n if build else self.suite.n, dim=self.dataset.dim, base_dtype="float32",
            machine=schema.machine(), gate=self.gate, open_us_batch=self.open_us, ref_drift_pct=self.drift,
            load_before=list(os.getloadavg()),
        )

        if system is not None:
            env = self.env(system, threads)
            record = {k: env[k] for k in (*schema.ENV_KEYS, *system.env) if k in env}
            line["provenance"] = schema.provenance(system.python, info.get("wheel_sha256") or {}, self.git, record)
        else:
            line["provenance"] = schema.provenance(sys.executable, {}, self.git)

        if build is not None:
            line.update(config_id=build.config_id, build_params=build.config, index_identity=self.index_identity(build))

        return line

    def write(self, system_name: str, line: dict) -> None:
        line["load_after"] = list(os.getloadavg())
        line = schema.clean(line)
        errors = schema.validate(line)

        if errors:
            line = _as_failed(line, errors)

        schema.append(self.out_dir / f"{system_name}.jsonl", line)
        self.progress.add(line["run_key"], line["status"])
        by = self.counts.setdefault(system_name, {})
        by[line["status"]] = by.get(line["status"], 0) + 1

    def run(self) -> int:
        self.log(f"batch {self.batch_id}, suite {self.suite.path}, results in {self.out_dir}")

        if self.dry_run:
            self.plan()
            return 0

        self.out_dir.mkdir(parents=True, exist_ok=True)

        started = schema.utc_now()
        self.git = schema.git_state()
        self.gate = self.wait_for_gate()
        disk_before = free_gib(REPO)
        floor = float(self.suite.gate["disk_floor_gb"])

        if self.gate.get("abort"):
            self.summary(started, disk_before, {}, flagged=True)
            return 3

        if disk_before < floor:
            self.log(f"only {disk_before:.0f} GiB free, below the {floor:.0f} GiB floor")
            self.gate["problems"].append(f"disk {disk_before:.0f} GiB below the floor")
            self.summary(started, disk_before, {}, flagged=True)
            return 3

        for removed in remove_stale(self.suite.index_root):
            self.log(f"removed {removed}, a temporary file of a build that is gone")

        self.open_us = self.time_open()
        agents_before, agents_at = agents_snapshot(self.suite.gate["agents"]), time.monotonic()
        refs: dict = {}

        if "refs" in self.only and self.suite.refs["mode"] == "on":
            refs["start"] = self.run_ref("start")

        try:
            if "build" in self.only:
                for build in self.suite.builds:
                    if self.selected(build.system):
                        self.run_build(build)

            if "warm" in self.only:
                jobs = [w for w in self.suite.warm if self.selected(w.system)]
                self.rng.shuffle(jobs)

                for job in jobs:
                    self.run_warm(job)

            if "cold" in self.only:
                self.run_cold()
        finally:
            if "refs" in self.only and self.suite.refs["mode"] == "on":
                refs["end"] = self.run_ref("end")

            agents = agents_delta(agents_before, agents_snapshot(self.suite.gate["agents"]),
                                  time.monotonic() - agents_at)
            self.summary(started, disk_before, refs, agents=agents)

        return 0

    def wait_for_gate(self) -> dict:
        cfg = self.suite.gate
        waited = 0.0

        while True:
            gate = check_gate(cfg)

            if gate["passed"] or cfg["mode"] in ("warn", "off"):
                if not gate["passed"]:
                    self.log(f"timing gate not met, going on as mode is {cfg['mode']}: {gate['problems']}")

                return gate

            if waited >= float(cfg["max_wait_s"]):
                self.log(f"timing gate not met after {waited:.0f} s: {gate['problems']}")
                return gate | {"abort": True}

            self.log(f"timing gate not met, checking again in {cfg['recheck_s']} s: {gate['problems']}")
            time.sleep(float(cfg["recheck_s"]))
            waited += float(cfg["recheck_s"])

    def time_open(self, repeats: int = 21) -> float | None:
        """The cost of one open() of an existing file, the median of `repeats`, eg. with an endpoint security agent
        screening every open. The first open of the new file costs more, and is kept in `self.first_open_us`."""
        probe = self.out_dir / f".open-probe-{os.getpid()}"
        times = []

        try:
            probe.write_bytes(b"open")

            for _ in range(1 + repeats):
                start = time.perf_counter_ns()
                fd = os.open(probe, os.O_RDONLY)
                times.append((time.perf_counter_ns() - start) / 1e3)
                os.close(fd)

            probe.unlink()
        except OSError:
            return None

        self.first_open_us = times[0]
        return float(np.median(times[1:]))

    def plan(self) -> None:
        """Prints every job and whether it is done, without running anything."""
        for b in self.suite.builds:
            state = "prebuilt" if b.prebuilt else ("done" if self.progress.done(self.build_key(b)) else "to do")
            print(f"build {b.system.name} {b.config_id} n={b.n} seed={b.seed}: {state} -> {b.index}")

        for w in self.suite.warm:
            todo = sum(1 for s in w.settings if not self.progress.done(self.warm_key(w, s)))
            print(f"warm {w.system.name} {w.build.config_id} split={w.split} threads={w.threads}: "
                  f"{todo} of {len(w.settings)} settings to do")

        for c in self.suite.cells:
            todo = sum(1 for r in range(c.trials) if not self.progress.done(self.cold_key(c, r)))
            print(f"cold {c.label} group={c.group} params={c.params}: {todo} of {c.trials} trials to do")

    def build_key(self, b: Build) -> str:
        return schema.run_key({"kind": "build", "phase": self.suite.phase, "system": b.system.name,
                               "variant": b.system.variant, "config_id": b.config_id, "config": b.config,
                               "dataset": self.suite.dataset, "n": b.n, "seed": b.seed})

    def _index_key(self, build: Build) -> str:
        """The index's size, file count, newest change and inode. A rebuilt index gets new run keys."""
        index = build.index

        if index is None or not index.exists():
            return f"missing:{index}"

        ident = identity(index, checksum=False)
        return f"{ident['bytes']}:{ident['files']}:{ident['mtime_ns']}:{ident['inode']}"

    def warm_key(self, w: Warm, params: dict) -> str:
        return schema.run_key({"kind": "warm", "phase": self.suite.phase, "system": w.system.name,
                               "variant": w.system.variant, "config_id": w.build.config_id,
                               "dataset": self.suite.dataset, "n": w.build.n, "split": w.split, "params": params,
                               "mode": w.mode, "threads": w.threads, "k": w.k, "index": self._index_key(w.build)})

    def cold_key(self, c: Cell, trial: int) -> str:
        """A cold trial's run key. It names the index by the key its cell took before its process started, since a C1
        process may hold the index open, and change it, for as long as the cell runs."""
        index = c.index_key if c.index_key is not None else self._index_key(c.build)
        return schema.run_key({"kind": "cold", "phase": self.suite.phase, "system": c.system.name,
                               "variant": c.system.variant, "config_id": c.build.config_id,
                               "dataset": self.suite.dataset, "n": c.build.n, "split": c.split, "params": c.params,
                               "regime": c.regime, "cpu_state": c.cpu_state, "delay_ms": c.delay_ms,
                               "threads": c.threads, "k": c.k, "next": c.next, "trial": trial,
                               "fresh_clone": c.fresh_clone, "index": index})

    def run_build(self, build: Build) -> None:
        system = build.system

        if build.prebuilt is not None:
            if not build.prebuilt.exists():
                self.log(f"the prebuilt index {build.prebuilt} of {system.name} isn't there")

            return

        key = self.build_key(build)

        if self.progress.done(key) and build.out.exists():
            self.resumed += 1
            return

        floor = float(self.suite.gate["disk_floor_gb"])
        free = free_gib(REPO)
        attempt = self.progress.attempts(key) + 1
        line = self.base_line("build", system, None, key, attempt, build.threads)
        # The build's own n, eg. a dummy's 5K, not the suite's. The index's identity comes once it is built.
        line.update(config_id=build.config_id, build_params=build.config, threads=build.threads, seed=build.seed,
                    n=build.n)

        if free - build.disk_gb < floor:
            line.update(status="failed", error=f"{free:.0f} GiB free, the build needs {build.disk_gb:.0f} GiB above "
                                               f"the {floor:.0f} GiB floor")
            self.write(system.name, line)
            return

        if build.out.exists():
            shutil.rmtree(build.out)

        build.out.mkdir(parents=True)
        args = ["build", *self.data_args(build.n, build.threads, system), "--config", json.dumps(build.config),
                "--out", str(build.out), "--seed", str(build.seed)]

        if build.clusters is not None:
            args += ["--clusters", str(build.clusters)]

        self.log(f"building {system.name} {build.config_id} n={build.n} seed={build.seed}")
        agents, agents_at = agents_snapshot(self.suite.gate["agents"]), time.monotonic()
        child = self.child(system, args, build.threads)
        sampler = DiskSampler(build.out, DISK_SAMPLE_S)
        msg = None
        start = time.monotonic()

        try:
            while child.proc.poll() is None or msg is None:
                got = child.event(1.0)

                if got is not None and got["event"] in ("build", "error", "unsupported"):
                    msg = got

                if free_gib(REPO) < min(floor, 20.0):
                    child.kill()
                    msg = {"event": "error", "message": f"stopped with only {free_gib(REPO):.0f} GiB of disk free"}
                    break

                if time.monotonic() - start > build.timeout_s:
                    child.kill()
                    msg = {"event": "error", "message": f"timed out after {build.timeout_s:.0f} s"}
                    break

                if child.proc.poll() is not None and got is None and msg is None:
                    break
        finally:
            peak, seen = sampler.stop()

        code = child.close()
        _, final = disk_use(build.out)
        now = agents_snapshot(self.suite.gate["agents"])
        line["agents_cpu"] = agents_delta(agents, now, time.monotonic() - agents_at)

        if msg is None or msg["event"] != "build":
            line.update(status="unsupported" if msg and msg["event"] == "unsupported" else "failed",
                        error=(msg or {}).get("message") or f"exit {code}: {child.tail()}")
            self.write(system.name, line)
            return

        fields = dict(msg["fields"])
        extra = fields.pop("extra", {}) or {}
        ident = identity(build.out)
        self.identities[str(build.out)] = ident
        line.update({k: v for k, v in fields.items() if k in schema.FIELDS["build"]})
        line.update(
            peak_disk_bytes=peak, temporary_files=len(seen - final), index_bytes=ident["bytes"],
            index_alloc_bytes=ident["alloc_bytes"], index_files=ident["files"], index_identity=ident,
            extra=extra | {k: v for k, v in fields.items() if k not in schema.FIELDS["build"]},
        )
        self.write(system.name, line)

    def run_ref(self, position: str) -> dict:
        """Runs the drift references, a raw read wave and ColdIVF's C0 at one fixed setting."""
        cfg = self.suite.refs
        bins = Path(os.environ.get("COLDIVF_BIN", REPO / "target" / "mine" / "release"))
        line = self.base_line("ref", None, None, schema.run_key({"kind": "ref", "batch": self.batch_id,
                                                                 "position": position}), 1)
        line.update(position=position, wave_trials=int(cfg["wave_trials"]), coldivf_trials=int(cfg["coldivf_trials"]))

        with tempfile.TemporaryDirectory(prefix="ref-") as tmp:
            out = Path(tmp) / "out.jsonl"
            wave = _repo_path(cfg["wave_file"])
            cmd = [str(bins / "wave"), "--file", str(wave), "--size-mb", str(cfg["wave_size_mb"]), "--trials",
                   str(cfg["wave_trials"]), "--ranges", str(cfg["wave_ranges"]), "--page", str(cfg["wave_page"]),
                   "--results", str(out)]
            ok = _run_logged(cmd, self.log_dir / "refs.log", timeout=1800)
            lines = schema.read(out) if ok else []
            line["wave_us_p50"] = lines[-1].get("p50_us") if lines else None

            if cfg["coldivf_index"]:
                out.unlink(missing_ok=True)
                params = cfg["coldivf_params"]
                cmd = [str(bins / "cold"), "--index", str(_repo_path(cfg["coldivf_index"])), "--dataset",
                       cfg["coldivf_dataset"], "--n", str(cfg["coldivf_n"]), "--trials", str(cfg["coldivf_trials"]),
                       "--mode", "wave", "--variant", "prewarm", "--results", str(out)]

                for key, value in params.items():
                    cmd += [f"--{key}", str(value)]

                ok = _run_logged(cmd, self.log_dir / "refs.log", timeout=3600)
                firsts = [x for x in (schema.read(out) if ok else []) if x.get("seq") == 0]
                clean = [x["total_us"] for x in firsts if x.get("pages_resident_at_start") == 0]
                line["coldivf_c0_p50_us"] = float(np.median(clean)) if clean else None
                line["coldivf_rejected"] = len(firsts) - len(clean)

        drift = self.ref_drift(line)
        line["ref_drift_pct"] = drift
        line["extra"] = {"refs": {"wave_file": str(_repo_path(cfg["wave_file"])), "coldivf": cfg["coldivf_index"],
                                  "params": cfg["coldivf_params"]}}

        if position == "start":
            self.drift = drift

        if line["wave_us_p50"] is None and line["coldivf_c0_p50_us"] is None:
            line.update(status="failed", error="neither reference ran")

        self.write("ref", line)
        return {k: line[k] for k in ("wave_us_p50", "coldivf_c0_p50_us", "ref_drift_pct")}

    def ref_drift(self, line: dict) -> float | None:
        """The larger drift of the two references from their medians over every earlier batch that enforced and
        passed the timing gate, in percent."""
        earlier = []

        for path in self.suite.results_root.glob("*/ref.jsonl"):
            for x in schema.read(path):
                gate = x.get("gate") or {}
                same = (x.get("extra") or {}).get("refs", {}).get("coldivf") == self.suite.refs["coldivf_index"]

                if (x.get("status") == "ok" and x.get("batch_id") != self.batch_id and same
                        and gate.get("mode") == "enforce" and gate.get("passed")):
                    earlier.append(x)

        drifts = []

        for key in ("wave_us_p50", "coldivf_c0_p50_us"):
            past = [x[key] for x in earlier if x.get(key)]

            if past and line.get(key):
                drifts.append(100.0 * (line[key] - float(np.median(past))) / float(np.median(past)))

        return max(drifts, key=abs) if drifts else None

    def overhead_key(self, system: System) -> str:
        return f"{system.name}|{system.variant}|{system.python}"

    def overhead_of(self, system: System) -> dict:
        """The system's per-call cost as `measure_overhead` found it in this batch, or an empty dict before that."""
        return self.overheads.get(self.overhead_key(system), {})

    def measure_overhead(self, system: System) -> dict:
        """Measures the system's per-call cost once per batch for each system, variant and interpreter, through its
        overhead subcommand, which times calls on an index of the base's first 1,000 rows. Every warm line of the
        system then carries it. A failure is kept with its reason and leaves overhead_us null."""
        key = self.overhead_key(system)

        if key in self.overheads:
            return self.overheads[key]

        self.log(f"overhead of {system.name} {system.variant}".rstrip())
        child = self.child(system, ["overhead", *self.data_args(OVERHEAD_ROWS, 1, system, 10)], 1)
        msg = child.expect(OVERHEAD_TIMEOUT_S, "overhead")
        code = child.close()

        if msg is not None and msg["event"] == "overhead":
            result = dict(msg["fields"])
        elif msg is not None and msg["event"] == "unsupported":
            result = {"overhead_us": None, "unsupported": msg.get("message")}
        else:
            result = {"overhead_us": None, "error": (msg or {}).get("message") or f"exit {code}: {child.tail()}"}

        self.overheads[key] = result
        return result

    def run_warm(self, job: Warm) -> None:
        system = job.system
        keys = [self.warm_key(job, s) for s in job.settings]
        todo = [i for i, key in enumerate(keys) if not self.progress.done(key)]
        self.resumed += len(keys) - len(todo)

        if not todo:
            return

        self.measure_overhead(system)

        if job.build.index is None or not job.build.index.exists():
            for i in todo:
                line = self.warm_line(job, job.settings[i], keys[i])
                line.update(status="failed", error=f"the index {job.build.index} isn't there")
                self.write(system.name, line)

            return

        ids_dir = self.out_dir / "ids"
        tmp_ids = Path(tempfile.mkdtemp(prefix="ids-", dir=self.out_dir))
        args = ["warm", *self.data_args(job.build.n, job.threads, system, job.k), "--index", str(job.build.index),
                "--grid", json.dumps([job.settings[i] for i in todo]), "--split", job.split, "--mode", job.mode,
                "--ids-dir", str(tmp_ids), "--passes", str(job.passes), "--warmup", str(job.warmup),
                "--qps-seconds", str(job.qps_seconds), "--qps-repeats", str(job.qps_repeats)]
        self.log(f"warm {system.name} {job.build.config_id}: {len(todo)} settings, split {job.split}, "
                 f"{job.threads} threads")
        child = self.child(system, args, job.threads)
        deadline = time.monotonic() + job.timeout_s
        reported: set[int] = set()
        error = None
        agents, agents_at = agents_snapshot(self.suite.gate["agents"]), time.monotonic()

        try:
            while (msg := child.expect(max(1.0, deadline - time.monotonic()), "setting", "done")) is not None:
                if msg["event"] != "setting":
                    if msg["event"] != "done":
                        error = msg

                    break

                i = todo[int(msg["index"])]
                reported.add(i)
                line = self.warm_line(job, job.settings[i], keys[i])
                fields = dict(msg["fields"])
                extra = fields.pop("extra", {}) or {}
                line.update({k: v for k, v in fields.items() if k in schema.FIELDS["warm"]})
                line["extra"] = extra | {k: v for k, v in fields.items() if k not in schema.FIELDS["warm"]}

                if msg.get("ids_file"):
                    ids_dir.mkdir(parents=True, exist_ok=True)
                    path = ids_dir / f"{keys[i]}.{line['run_id'][:8]}.npy"
                    shutil.move(msg["ids_file"], path)
                    recall = self.recall_for(job.build.n)

                    if recall is not None:
                        line.update(recall.summary(np.asarray(msg["qids"]), np.load(path)))

                    line["ids_file"] = str(path)

                moved = self.vm_moved({"swapins": line.get("swapins_delta"),
                                       "compressions": line.get("compressions_delta")})

                if moved:
                    line.update(status="invalid", error=f"{moved} during the timed passes")

                now = agents_snapshot(self.suite.gate["agents"])
                line["agents_cpu"] = agents_delta(agents, now, time.monotonic() - agents_at)
                agents, agents_at = now, time.monotonic()
                self.write(system.name, line)
        finally:
            code = child.close()
            shutil.rmtree(tmp_ids, ignore_errors=True)

        for i in todo:
            if i not in reported:
                line = self.warm_line(job, job.settings[i], keys[i])
                status = "unsupported" if error and error["event"] == "unsupported" else "failed"
                line.update(status=status, error=(error or {}).get("message") or f"exit {code}: {child.tail()}")
                self.write(system.name, line)

    def warm_line(self, job: Warm, params: dict, key: str) -> dict:
        line = self.base_line("warm", job.system, job.build, key, self.progress.attempts(key) + 1, job.threads)
        line.update(search_params=params, split=job.split, mode=job.mode, threads=job.threads, k=job.k,
                    gt=self.truth_record(job.build.n), overhead_us=self.overhead_of(job.system).get("overhead_us"))
        return line

    def trial_queries(self, cell: Cell, trial: int) -> list[int]:
        """The queries of trial r, which are the split's query at position r and the next `cell.next` positions,
        wrapping around."""
        qids = data.split_ids(self.dataset.queries.shape[0], cell.split)
        return [int(qids[(trial + j) % len(qids)]) for j in range(1 + cell.next)]

    def warmup_queries(self, cell: Cell, count: int = WARMUP_QUERIES) -> list[int]:
        """Queries for the searches a cell's process runs before READY, none of which any trial of the cell searches.
        So a cache that a warm-up fills can't answer a timed query. They come from the other split of a tune or test
        cell, and otherwise from the end of the cell's own split."""
        total = self.dataset.queries.shape[0]
        used = {q for trial in range(cell.trials) for q in self.trial_queries(cell, trial)}
        other = {"tune": "test", "test": "tune"}.get(cell.split)
        own = data.split_ids(total, cell.split).tolist()[::-1]
        pool = (data.split_ids(total, other).tolist() if other else []) + own
        out: list[int] = []

        for q in pool:
            if q not in used and q not in out:
                out.append(int(q))

            if len(out) == count:
                break

        return out

    def run_cold(self) -> None:
        cells = [c for c in self.suite.cells if self.selected(c.system)]

        for group in sorted({c.group for c in cells}):
            members = [c for c in cells if c.group == group]
            self.log(f"cold group {group}: {len(members)} cells")
            self.cold_group(members)

    def cold_group(self, cells: list[Cell]) -> None:
        maps: dict[str, evictor.Evictor] = {}
        specs: dict[int, list[dict]] = {}

        for cell in cells:
            index = cell.build.index

            if index is None or not index.exists():
                cell.skip = f"the index {index} isn't there"
                continue

            key = str(index)

            try:
                # Each variant lists the files as it reads them, eg. whether its open copies one whole, and the cells
                # of one index share one evictor.
                specs[id(cell)] = self.files(cell.system, index)

                if key not in maps:
                    watched = [Path(f["path"]) for f in specs[id(cell)]]
                    maps[key] = evictor.Evictor(watched + ([index] if index.is_dir() else []))
            except (RuntimeError, OSError, ValueError, KeyError) as e:
                cell.skip = f"couldn't list or map the index's files: {e}"

            # No process of the group has opened the index yet, so it is as its build left it.
            cell.index_key = self._index_key(cell.build)

        live: dict[int, Child] = {}

        try:
            for trial in range(max([c.trials for c in cells] or [0])):
                order = [c for c in cells if trial < c.trials]
                self.rng.shuffle(order)

                for cell in order:
                    key = self.cold_key(cell, trial)

                    if self.progress.done(key):
                        self.resumed += 1
                        continue

                    if cell.skip is not None:
                        # One line says why, and the cell's other trials wait for a later batch.
                        if not cell.reported:
                            self.cold_failed(cell, trial, key, cell.skip_status, cell.skip, self.previous)
                            cell.reported = True

                        continue

                    ev, files = maps[str(cell.build.index)], specs[id(cell)]

                    if cell.regime == "c1":
                        self.c1_trial(cell, trial, key, ev, files, live, self.previous)
                    else:
                        self.c0_trial(cell, trial, key, ev, files, self.previous)

                    self.previous = cell.label
        finally:
            for child in live.values():
                child.close()

            for ev in maps.values():
                ev.close()

    def cold_line(self, cell: Cell, trial: int, key: str, previous: str | None) -> dict:
        line = self.base_line("cold", cell.system, cell.build, key, self.progress.attempts(key) + 1, cell.threads)
        qids = self.trial_queries(cell, trial)
        line.update(search_params=cell.params, split=cell.split, regime=cell.regime, cpu_state=cell.cpu_state,
                    delay_ms=cell.delay_ms, threads=cell.threads, k=cell.k, trial=trial, query_id=qids[0],
                    predecessor=previous, gt=self.truth_record(cell.build.n))
        return line

    def cold_failed(self, cell: Cell, trial: int, key: str, status: str, error: str, previous: str | None) -> None:
        line = self.cold_line(cell, trial, key, previous)
        line.update(status=status, error=error[:4000])
        self.write(cell.system.name, line)

    def not_ready(self, cell: Cell, trial: int, key: str, msg: dict | None, child: Child,
                  previous: str | None) -> None:
        """Records an adapter that didn't reach READY. An unsupported regime stops the cell for this batch."""
        message = (msg or {}).get("message") or f"no READY: {child.tail()}"

        if msg is not None and msg["event"] == "unsupported":
            cell.skip, cell.skip_status, cell.reported = message, "unsupported", True
            self.cold_failed(cell, trial, key, "unsupported", message, previous)
        else:
            self.cold_failed(cell, trial, key, "failed", message, previous)

    def cold_args(self, cell: Cell, queries: list[int]) -> list[str]:
        args = ["cold", *self.data_args(cell.build.n, cell.threads, cell.system, cell.k), "--index",
                str(cell.build.index), "--params", json.dumps(cell.params), "--queries",
                ",".join(str(q) for q in queries), "--regime", cell.regime, "--cpu-state", cell.cpu_state,
                "--delay-ms", str(cell.delay_ms), "--split", cell.split, "--warmup-queries",
                ",".join(str(q) for q in self.warmup_queries(cell))]

        if cell.dummy_path is not None and cell.regime != "c1":
            args += ["--dummy", str(cell.dummy_path)]

        return args

    def c0_trial(self, cell: Cell, trial: int, key: str, ev: evictor.Evictor, files: list[dict],
                 previous: str | None) -> None:
        """One C0 or intent trial, in a fresh adapter process. A trial that starts with any page cached is rejected
        and tried again, up to the limit of attempts."""
        queries = self.trial_queries(cell, trial)

        while not self.progress.done(key):
            clone = self.clone(cell, trial, files) if cell.fresh_clone else None

            try:
                self.c0_attempt(cell, trial, key, queries, clone[1] if clone else ev, clone[2] if clone else files,
                                previous, clone)
            finally:
                if clone is not None:
                    clone[1].close()
                    shutil.rmtree(clone[0].parent, ignore_errors=True)

            if cell.skip is not None:
                return

    def clone(self, cell: Cell, trial: int, files: list[dict]) -> tuple[Path, evictor.Evictor, list[dict]]:
        """A copy-on-write clone of the cell's index under the index root, an evictor that holds its files, and its
        file list."""
        src = cell.build.index
        folder = self.suite.index_root / "clones.noindex" / f"{os.getpid()}-{trial}-{os.urandom(3).hex()}"
        folder.mkdir(parents=True)
        dst = folder / src.name
        subprocess.run(["cp", "-c", "-R", str(src), str(dst)], check=True, capture_output=True)
        moved = moved_files(files, src, dst)
        watched = [Path(f["path"]) for f in moved] + ([dst] if dst.is_dir() else [])
        return dst, evictor.Evictor(watched), moved

    def c0_attempt(self, cell: Cell, trial: int, key: str, queries: list[int], ev: evictor.Evictor,
                   files: list[dict], previous: str | None, clone: tuple | None) -> None:
        """One attempt at a C0 or intent trial, which writes one line."""
        args = self.cold_args(cell, queries)

        if clone is not None:
            args[args.index("--index") + 1] = str(clone[0])

        child = self.child(cell.system, args, cell.threads)
        ready = child.expect(cell.ready_timeout_s, "ready")

        if ready is None or ready["event"] != "ready":
            child.close()
            self.not_ready(cell, trial, key, ready, child, previous)
            return

        agents, agents_at = agents_snapshot(self.suite.gate["agents"]), time.monotonic()
        line = self.cold_line(cell, trial, key, previous)
        line["extra"] = {"clone": str(clone[0])} if clone is not None else {}
        # A clone's files are new, and another process may read their first pages later than usual.
        cold = ev.make_cold(self.suite.wait_s * (2 if clone is not None else 1))
        line.update(resident_pages_at_go=cold.resident_pages, files_in_index=cold.files)

        if cold.resident_pages > 0:
            child.close()
            line.update(status="rejected", error=f"{cold.resident_pages} pages cached at GO")
            line["extra"]["residency_at_go"] = {k: v for k, v in cold.pages.items() if v}
            self.write(cell.system.name, line)
            return

        vm_before = rusage.vm_counts()
        child.send({"cmd": "go", "trial": trial, "queries": queries})
        msg = child.expect(cell.timeout_s, "trial")
        self.finish_trial(cell, line, msg, child, ev, vm_before, agents, agents_at, files)
        child.close()

    def c1_trial(self, cell: Cell, trial: int, key: str, ev: evictor.Evictor, files: list[dict],
                 live: dict[int, Child], previous: str | None) -> None:
        """One C1 trial in the cell's long-lived adapter process. The driver drops the bulk ranges the adapter
        declared, and at most 0.5% of them may stay cached."""
        bulk = [evictor.Range(Path(f["path"]), r["offset"], r["length"]) for f in files for r in f["ranges"]
                if r["kind"] == "bulk" and r["length"] > 0]
        queries = self.trial_queries(cell, trial)

        while not self.progress.done(key):
            child = live.get(id(cell))

            if child is None or child.proc.poll() is not None:
                qids = data.split_ids(self.dataset.queries.shape[0], cell.split).tolist()
                child = self.child(cell.system, self.cold_args(cell, qids), cell.threads)
                ready = child.expect(cell.ready_timeout_s, "ready")

                if ready is None or ready["event"] != "ready":
                    child.close()
                    self.not_ready(cell, trial, key, ready, child, previous)

                    if cell.skip is not None:
                        return

                    continue

                live[id(cell)] = child

            agents, agents_at = agents_snapshot(self.suite.gate["agents"]), time.monotonic()
            line = self.cold_line(cell, trial, key, previous)
            ev.refresh()
            held = [r for r in bulk if str(r.path) in ev.maps]
            ev.evict_ranges(held)
            cached, total = ev.resident_in(held)
            line.update(resident_pages_at_go=cached, files_in_index=len(ev.maps))

            if total and cached > 0.005 * total:
                line.update(status="rejected", error=f"{cached} of {total} bulk pages cached at GO")
                self.write(cell.system.name, line)
                continue

            vm_before = rusage.vm_counts()
            child.send({"cmd": "go", "trial": trial, "queries": queries, "self_evict": cell.self_evict})
            msg = child.expect(cell.timeout_s, "trial")
            self.finish_trial(cell, line, msg, child, ev, vm_before, agents, agents_at, files)

            if msg is None or msg["event"] != "trial":
                child.kill()
                child.close()
                live.pop(id(cell), None)

    def finish_trial(self, cell: Cell, line: dict, msg: dict | None, child: Child, ev: evictor.Evictor,
                     vm_before: dict, agents: dict, agents_at: float, files: list[dict]) -> None:
        """Completes a trial's line from the adapter's result and the files' residency after the trial. `files` is the
        index's file list, as the adapter's files() gives it for the files `ev` holds. The cached bytes of its "loaded"
        ranges, which the system copied into its own memory, are left out of the resident bytes and go apart."""
        after = ev.residency()
        loaded = ev.resident_bytes_in(evictor.loaded_ranges(files))
        vm = rusage.vm_delta(vm_before, rusage.vm_counts())
        line.update(files_touched=after.touched, resident_file_bytes_after=after.resident_bytes - loaded,
                    swapins_delta=vm["swapins"], compressions_delta=vm["compressions"],
                    agents_cpu=agents_delta(agents, agents_snapshot(self.suite.gate["agents"]),
                                            time.monotonic() - agents_at))
        line["extra"] = (line.get("extra") or {}) | {"resident_loaded_bytes_after": loaded}

        if msg is None or msg["event"] != "trial":
            line.update(status="failed", error=(msg or {}).get("message") or f"no trial result: {child.tail()}")
            self.write(cell.system.name, line)
            return

        fields = dict(msg["fields"])
        extra = fields.pop("extra", {}) or {}
        status = fields.pop("status", "ok")
        error = fields.pop("error", None)
        line.update({k: v for k, v in fields.items() if k in schema.FIELDS["cold"] and v is not None})
        line["extra"] = line["extra"] | extra | {k: v for k, v in fields.items() if k not in schema.FIELDS["cold"]}
        touched = touched_files(after, extra.get("files_read"), [f["path"] for f in files])

        if touched != after.touched:
            # Reads that skip the page cache leave nothing for mincore to see, so the adapter's own counts add them.
            line["extra"]["files_touched_cached"] = after.touched
            line["files_touched"] = touched

        recall = self.recall_for(cell.build.n)

        if line.get("ids") and recall is not None:
            found = np.asarray([line["ids"]])
            ties, plain = recall.per_query(np.asarray([line["query_id"]]), found, min(10, cell.k))
            line.update(recall10=float(ties[0]), recall10_ids_only=float(plain[0]))

        moved = self.vm_moved(vm)

        if status == "ok" and moved:
            status, error = "invalid", f"{moved} during the trial"

        line.update(status=status, error=error)
        self.write(cell.system.name, line)

    def vm_moved(self, deltas: dict) -> str:
        """Which of the suite's invalidating counters moved, eg. "swapins 12", or "" when none did."""
        moved = [f"{name} {deltas[name]}" for name in self.suite.gate["invalid_vm"] if (deltas.get(name) or 0) > 0]
        return ", ".join(moved)

    def summary(self, started: str, disk_before: float, refs: dict, agents: dict | None = None,
                flagged: bool = False) -> None:
        line = self.base_line("batch", None, None, schema.run_key({"kind": "batch", "batch": self.batch_id}), 1)
        limit = float(self.suite.refs["drift_pct"])
        drifts = [refs.get(p, {}).get("ref_drift_pct") for p in ("start", "end")]
        total: dict[str, int] = {}

        for by in self.counts.values():
            for status, count in by.items():
                total[status] = total.get(status, 0) + count

        git_after = schema.git_state()
        line.update(
            started=started, ended=schema.utc_now(), suite=str(self.suite.path), suite_sha256=self.suite.sha256,
            counts=total, by_system=self.counts, resumed=self.resumed, refs=refs, drift_start_pct=drifts[0],
            drift_end_pct=drifts[1], agents_cpu_after=agents,
            flagged=flagged or any(d is not None and abs(d) > limit for d in drifts) or not self.gate.get("passed"),
            provenance_changed=bool(self.git) and any(git_after.get(k) != self.git.get(k)
                                                      for k in ("head", "tree_hash", "diff_hash")),
            disk_free_gb={"before": round(disk_before, 1), "after": round(free_gib(REPO), 1)},
            extra={"open_us_first": self.first_open_us, "open_us_repeat_p50": self.open_us,
                   "gate_problems": self.gate.get("problems", []), "overhead": self.overheads,
                   "systems": {key: {k: info.get(k) for k in INFO_KEYS} for key, info in self.infos.items() if info}},
        )
        self.write("batch", line)
        self.log(f"batch {self.batch_id} done: {total}, {self.resumed} skipped as done")


def _as_failed(line: dict, errors: list[str]) -> dict:
    """A line that doesn't fit the schema, kept as a failed line so its attempt still counts. Fields with a wrong type
    become null, or an empty value where null isn't allowed, and the original goes under "extra"."""
    kind = line.get("kind") if line.get("kind") in schema.FIELDS else "batch"
    out = schema.blank(kind)
    spec = schema.COMMON | schema.FIELDS[kind]

    for name, want in spec.items():
        value = line.get(name)

        if value is not None and schema.TYPES[want.rstrip("?")](value):
            out[name] = value

    out.update(schema=schema.SCHEMA, kind=kind, status="failed", error=f"invalid line: {'; '.join(errors)}"[:4000],
               extra={"invalid": {k: v for k, v in line.items() if k not in ("machine", "provenance")}})

    for name, want in spec.items():
        if out[name] is None and not want.endswith("?"):
            out[name] = {"str": "", "int": 0, "num": 0.0, "bool": False, "dict": {}}.get(want, [])

    return out


def _run_logged(cmd: list[str], log: Path, timeout: float) -> bool:
    log.parent.mkdir(parents=True, exist_ok=True)

    with open(log, "ab") as f:
        f.write(f"\n== {schema.utc_now()} {' '.join(cmd)}\n".encode())
        f.flush()

        try:
            return subprocess.run(cmd, stdout=f, stderr=f, timeout=timeout, cwd=REPO).returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Runs a benchmark suite.")
    p.add_argument("suite")
    p.add_argument("--only", default="", help="a comma-separated subset of build, warm, cold and refs")
    p.add_argument("--systems", default="", help="a comma-separated subset of the suite's systems")
    p.add_argument("--gate", default=None, choices=("enforce", "warn", "off"))
    p.add_argument("--max-attempts", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    suite = load_suite(args.suite)
    only = {x.strip() for x in args.only.split(",") if x.strip()} or None
    systems = {x.strip() for x in args.systems.split(",") if x.strip()} or None
    driver = Driver(suite, only=only, systems=systems, gate_mode=args.gate, dry_run=args.dry_run,
                    max_attempts=args.max_attempts)
    return driver.run()


if __name__ == "__main__":
    sys.exit(main())
