"""Runs a benchmark suite through each system's adapter and appends the results.

    python driver.py SUITE.toml [--only build,stream,latency,maintain,warm,cold,refs] [--systems a,b] [--groups g,h]
                                [--gate enforce|warn|off] [--dry-run]

A batch checks the timing gate and the disk floor, runs the drift references, builds what isn't built, measures each
system's per-call cost and runs the warm grids, then the cold trials in rounds of randomly ordered cells, runs the
references again and writes a batch summary.
A suite with a [writes] table also runs the write benchmark's phases after the builds. The stream phase replays the
stream on clones of each build and keeps a clone of the work folder as each state, the latency phase times single
writes on throwaway clones of the latency states, and the maintain phase runs each maintenance action on a clone of
S10. A system that writes its files in place takes plain copies instead of clones, and each round's work folder becomes
its state. Warm and cold entries with states then search every state as an index of its own.
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
import selectors
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
import writes  # noqa: E402

REPO = PY.parents[1]
GIB = 1 << 30

SUITE_KEYS = {"phase", "dataset", "n", "index_root", "results_root", "seed", "k", "truth", "gate", "refs", "system",
              "max_attempts", "wait_s", "writes"}
GATE_KEYS = {"mode", "cpu_idle_pct", "top_samples", "top_interval_s", "disk_floor_gb", "max_wait_s", "recheck_s",
             "require_ac", "require_caffeinate", "busy_names", "agents", "invalid_vm", "locks"}
REFS_KEYS = {"mode", "wave_trials", "wave_ranges", "wave_page", "wave_file", "wave_size_mb", "coldivf_index",
             "coldivf_dataset", "coldivf_n", "coldivf_params", "coldivf_trials", "drift_pct"}
SYSTEM_KEYS = {"name", "adapter", "python", "variant", "env", "build", "warm", "cold", "write", "maintain"}
BUILD_KEYS = {"config_id", "config", "seeds", "threads", "disk_gb", "timeout_s", "index", "clusters", "n", "data"}
WARM_KEYS = {"config_id", "seed", "split", "threads", "mode", "grid", "settings", "passes", "warmup", "k",
             "qps_seconds", "qps_repeats", "timeout_s", "states", "findability"}
COLD_KEYS = {"config_id", "seed", "regime", "params", "trials", "split", "cpu_state", "next", "delay_ms", "threads",
             "group", "dummy", "dummy_n", "k", "self_evict", "fresh_clone", "timeout_s", "ready_timeout_s", "states"}
WRITE_KEYS = {"config_id", "seed", "role", "threads", "states", "params", "timeout_s"}
MAINTAIN_KEYS = {"config_id", "seed", "state", "actions", "threads", "action_threads", "params", "timeout_s", "disk_gb",
                 "unsearched", "same_file"}

READ_PHASES = {"build", "warm", "cold", "refs"}
WRITE_PHASES = {"stream", "latency", "maintain"}

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
INFO_KEYS = ("system", "system_version", "binding", "regimes", "batch_api", "qps_mode", "workers", "worker_qos",
             "write_role", "sync_kind", "writes_in_place")
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
    # Another base file to build from, eg. the copy of a state's live rows, with its queries and truth beside it.
    data: Path | None = None
    # A state of the write benchmark, eg. "s5", searched as an index of its own, and the build it came from. Its n is
    # the vectors it holds.
    state: str | None = None
    origin: Build | None = None

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
    # Whether the first setting also searches the state's probe rows as queries, for the findability check.
    findability: bool = False


@dataclass
class WriteJob:
    """A system's write entry: the build it writes to, its role, and the states its passes keep."""

    system: System
    build: Build
    role: str
    states: list[str]
    threads: int = 1
    params: dict = field(default_factory=dict)
    timeout_s: float = 7200.0


@dataclass
class MaintainJob:
    """A system's maintenance entry: each action runs on a clone of `state` and becomes the state <state>-<action>.
    `same_file` maps an action to another whose data file it must match, and `unsearched` lists the actions whose
    states a warm or cold entry's "all" leaves out."""

    system: System
    build: Build
    state: str
    actions: list[str]
    threads: int = 10
    action_threads: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    timeout_s: float = 7200.0
    disk_gb: float = 0.0
    unsearched: list[str] = field(default_factory=list)
    same_file: dict = field(default_factory=dict)

    def threads_of(self, action: str) -> int:
        return int(self.action_threads.get(action, self.threads))


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
    writes: list[WriteJob] = field(default_factory=list)
    maintains: list[MaintainJob] = field(default_factory=list)
    # The states its warm and cold entries search, by build and state.
    states: dict = field(default_factory=dict)

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
    writes: writes.WritesConfig | None = None

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

    if "writes" in doc:
        try:
            suite.writes = writes.parse_writes(doc["writes"], n, suite.dataset)
        except writes.StreamError as e:
            raise SuiteError(str(e)) from e

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
                              prebuilt=_repo_path(b.get("index")), clusters=_repo_path(b.get("clusters")),
                              data=_repo_path(b.get("data")))
                folder = _slug(name + (f"-{variant}" if variant else ""))
                build.out = index_root / folder / _slug(f"{build.config_id}-n{bn}-s{build.seed}")
                system.builds.append(build)

        _load_writes(suite, system, entry)

        for w in entry.get("warm", []):
            _check_keys(w, WARM_KEYS, f"a warm entry of {name}")
            build = _find_build(system, str(w["config_id"]), w.get("seed"), f"a warm entry of {name}")
            settings = expand_grid(w.get("grid"), w.get("settings"))

            if not settings:
                raise SuiteError(f"a warm entry of {name} has no settings")

            states = _entry_states(suite, system, build, w.get("states"), f"a warm entry of {name}")
            probed = _findability(suite, system, build, w.get("findability", False), states, name)

            for target in [state_build(suite, system, build, x) for x in states] if states is not None else [build]:
                system.warm.append(Warm(system=system, build=target, split=str(w.get("split", "tune")),
                                        threads=int(w.get("threads", 1)), mode=str(w.get("mode", "whole")),
                                        settings=settings, passes=int(w.get("passes", 3)),
                                        warmup=int(w.get("warmup", 1000)), k=int(w.get("k", k)),
                                        qps_seconds=float(w.get("qps_seconds", 10)),
                                        qps_repeats=int(w.get("qps_repeats", 3)),
                                        timeout_s=float(w.get("timeout_s", 7200)),
                                        findability=target.state in probed))

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

            states = _entry_states(suite, system, build, c.get("states"), f"a cold entry of {name}")

            for target in [state_build(suite, system, build, x) for x in states] if states is not None else [build]:
                system.cold.append(Cell(system=system, build=target, regime=regime, params=dict(c.get("params", {})),
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


def _load_writes(suite: Suite, system: System, entry: dict) -> None:
    """Reads a system's write and maintenance entries, which need the suite's [writes] table."""
    cfg = suite.writes

    for w in entry.get("write", []):
        _check_keys(w, WRITE_KEYS, f"a write entry of {system.name}")

        if cfg is None:
            raise SuiteError(f"a write entry of {system.name} needs the suite's [writes] table")

        build = _find_build(system, str(w["config_id"]), w.get("seed"), f"a write entry of {system.name}")
        role = str(w.get("role", "full"))

        if role not in writes.ROLES or role == "rebuild":
            raise SuiteError(f"a write entry of {system.name} has role {role}, not full, inserts or exact")

        states = [str(x) for x in w.get("states") or default_states(cfg, role)]

        for x in states:
            spec = _spec(cfg, x, f"a write entry of {system.name}")

            if not spec.pass_name or (role == "inserts" and spec.pass_name != "inserts") or \
                    (role == "exact" and spec.pass_name == "policy"):
                raise SuiteError(f"a write entry of {system.name} with role {role} can't make state {x}")

        system.writes.append(WriteJob(system=system, build=build, role=role, states=states,
                                      threads=int(w.get("threads", 1)), params=dict(w.get("params", {})),
                                      timeout_s=float(w.get("timeout_s", 7200))))

    for m in entry.get("maintain", []):
        _check_keys(m, MAINTAIN_KEYS, f"a maintenance entry of {system.name}")

        if cfg is None:
            raise SuiteError(f"a maintenance entry of {system.name} needs the suite's [writes] table")

        build = _find_build(system, str(m["config_id"]), m.get("seed"), f"a maintenance entry of {system.name}")
        final = cfg.final("joint")
        state = str(m.get("state") or (final.name if final else ""))
        _spec(cfg, state, f"a maintenance entry of {system.name}")
        actions = [str(a) for a in m.get("actions", [])]

        if not actions or not all(writes.STATE_NAME.match(a) for a in actions):
            raise SuiteError(f"a maintenance entry of {system.name} needs actions of letters and digits")

        job = MaintainJob(system=system, build=build, state=state, actions=actions, threads=int(m.get("threads", 10)),
                          action_threads={str(a): int(t) for a, t in m.get("action_threads", {}).items()},
                          params=dict(m.get("params", {})), timeout_s=float(m.get("timeout_s", 7200)),
                          disk_gb=float(m.get("disk_gb", 0)), unsearched=[str(a) for a in m.get("unsearched", [])],
                          same_file={str(a): str(b) for a, b in m.get("same_file", {}).items()})

        if not set(job.action_threads) | set(job.unsearched) | set(job.same_file) | set(job.same_file.values()) <= \
                set(actions):
            raise SuiteError(f"a maintenance entry of {system.name} names an action it doesn't run")

        system.maintains.append(job)


def _spec(cfg: writes.WritesConfig, name: str, where: str) -> writes.StateSpec:
    try:
        return cfg.spec(name)
    except writes.StreamError as e:
        raise SuiteError(f"{where}: {e}") from e


def default_states(cfg: writes.WritesConfig, role: str) -> list[str]:
    """The states a write entry keeps when it names none: the joint ladder and S10i for full writes, the inserts-only
    ladder for inserts only, and both ladders for the exact reference."""
    joint = [s.name for s in cfg.ladder("joint")]
    inserts = [s.name for s in cfg.ladder("inserts")]
    final = inserts[-1:] if inserts else []
    return {"full": joint + final, "inserts": inserts, "exact": joint + inserts}[role]


def role_of(system: System, build: Build) -> str:
    """The write role of a build, from the system's write entry for it, or "rebuild" without one."""
    for job in system.writes:
        if job.build is build:
            return job.role

    return "rebuild"


def known_states(system: System, build: Build, searched: bool = False) -> list[str]:
    """Every state of a build: S0, each state its write entries keep, and each state its maintenance actions make. With
    `searched` it leaves out the states of the actions an entry lists as unsearched."""
    out = ["s0"]

    for job in system.writes:
        if job.build is build:
            out += [x for x in job.states if x not in out]

    for job in system.maintains:
        if job.build is build:
            out += [f"{job.state}-{a}" for a in job.actions if not (searched and a in job.unsearched)]

    return out


def _entry_states(suite: Suite, system: System, build: Build, value: object, where: str) -> list[str] | None:
    """The states a warm or cold entry searches: None without `states`, every searched state for "all", or the ones
    it lists, each of which the build must make."""
    if value is None:
        return None

    if suite.writes is None:
        raise SuiteError(f"{where} names states, which need the suite's [writes] table")

    if value == "all":
        return known_states(system, build, searched=True)

    states = [str(x) for x in value] if isinstance(value, list) else []
    unknown = [x for x in states if x not in known_states(system, build)]

    if not states or unknown:
        raise SuiteError(f"{where} names states {unknown or value}, which its build doesn't make")

    return states


def _findability(suite: Suite, system: System, build: Build, value: object, states: list[str] | None,
                 name: str) -> set:
    """The states a warm entry's findability check runs at. `true` takes the last state of each ladder and every
    maintained state among the entry's states, and a list takes the states it names."""
    if not value:
        return set()

    if states is None:
        raise SuiteError(f"a warm entry of {name} with findability needs states")

    if isinstance(value, list):
        return {str(x) for x in value} & set(states)

    cfg = suite.writes
    ends = {s.name for s in (cfg.final("joint"), cfg.final("inserts")) if s is not None}
    return {x for x in states if x in ends or cfg.spec(x).action is not None}


def state_build(suite: Suite, system: System, build: Build, name: str) -> Build:
    """The state `name` of a build as an index of its own, a folder <config_id>-<state> beside the build's own. A
    rebuild-only system's S0 is its build itself."""
    key = (build.config_id, build.seed, name)

    if key not in system.states:
        cfg = suite.writes
        path = build.index if name == "s0" and role_of(system, build) == "rebuild" else \
            build.out.parent / _slug(f"{build.config_id}-{name}")
        target = Build(system=system, config_id=build.config_id, config=build.config, seed=build.seed,
                       n=cfg.live_count(name), threads=build.threads, prebuilt=path, state=name, origin=build)
        target.out = path
        system.states[key] = target

    return system.states[key]


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

    # A state the driver was placing when it stopped, under its temporary name.
    for folder in sorted(root.glob("*/*.tmp-*")):
        owner = folder.name.rsplit(".tmp-", 1)[1]

        if folder.is_dir() and owner.isdigit() and not _alive(int(owner)):
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


def folder_key(path: Path | None) -> str:
    """An index's size, file count, newest change and inode, the part of a run key that names it."""
    if path is None or not path.exists():
        return f"missing:{path}"

    ident = identity(path, checksum=False)
    return f"{ident['bytes']}:{ident['files']}:{ident['mtime_ns']}:{ident['inode']}"


def work_marker(work: Path) -> Path:
    """The record of what a pass's work folder holds, beside it."""
    return work.with_name(f"{work.name}.work.json")


def drop_cache(path: Path, wait: float) -> evictor.Residency:
    """Drops the cached pages of a new copy, which its writes left there, so a session finds the files on disk, as it
    finds a new clone's. It returns the pages still cached after a second drop `wait` seconds later."""
    with evictor.Evictor([path]) as ev:
        return ev.make_cold(wait)


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

            # select() takes no descriptor past FD_SETSIZE, 1,024 on macOS, and a cold trial passes that once it holds
            # every file of an index with thousands of them open, eg. LanceDB's after its writes. A selector has no limit.
            with selectors.DefaultSelector() as sel:
                sel.register(fd, selectors.EVENT_READ)
                ready = sel.select(left)

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
                 gate_mode: str | None = None, dry_run: bool = False, max_attempts: int | None = None,
                 groups: set[str] | None = None):
        self.suite = suite
        self.only = only or (READ_PHASES | (WRITE_PHASES if suite.writes is not None else set()))
        self.systems = systems
        # The cold groups this batch runs, eg. one rival's group with the ColdIVF cells that run beside it. None is all.
        self.groups = groups
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
        # The write benchmark's stream, its hash, and each state's scorer, read once per batch.
        self._stream: writes.Stream | None = None
        self._stream_sha: str | None = None
        self._state_recalls: dict[tuple, data.Recall | None] = {}
        # For a system that writes in place, each pass's work folder by path and the index key of the state its last
        # session in this batch made, whose copy for the next round may keep its cached pages.
        self.copied: dict[str, str] = {}

    def selected(self, system: System) -> bool:
        """Whether --systems names the system, by its name for every variant of it, or as name:variant for one."""
        return self.systems is None or bool({system.name, f"{system.name}:{system.variant}"} & self.systems)

    def selected_cell(self, cell: Cell) -> bool:
        return self.selected(cell.system) and (self.groups is None or cell.group in self.groups)

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

    def scorer(self, build: Build) -> data.Recall | None:
        """Scores the results of a search of `build`. A state, a copy of a state's live rows and a build of an
        inserts-only state's prefix take that state's truth, through StateRecall. Any other build takes the truth of
        its n, as in the head-to-head."""
        cfg = self.suite.writes

        if cfg is not None:
            if build.state is not None:
                return self.state_recall(build.state)

            if build.data is not None:
                record = writes.copy_record(build.data)

                if record is not None:
                    return self.state_recall(record["state"], build.data)

            prefix = self.prefix_state(build.n)

            if prefix is not None:
                return self.state_recall(prefix)

        return self.recall_for(build.n)

    def scorer_record(self, build: Build) -> dict | None:
        recall = self.scorer(build)
        return recall.truth.record() if recall is not None else None

    def prefix_state(self, n: int) -> str | None:
        """The inserts-only state whose vectors are the base's first n rows, other than S0, or None."""
        cfg = self.suite.writes

        for spec in cfg.ladder("inserts"):
            if cfg.live_count(spec.name) == n:
                return spec.name

        return None

    def stream(self) -> writes.Stream:
        """The suite's stream, made when its file is missing."""
        if self._stream is None:
            self._stream = writes.ensure_stream(self.suite.writes, self.dataset)
            self._stream_sha = schema.file_sha256(self.suite.writes.stream)

        return self._stream

    def stream_sha(self) -> str | None:
        cfg = self.suite.writes

        if cfg is not None and self._stream_sha is None and cfg.stream.exists():
            self._stream_sha = schema.file_sha256(cfg.stream)

        return self._stream_sha

    def state_truth(self, name: str) -> Path:
        """The truth file of a state. S0 takes the suite's truth file when it has one, as the head-to-head did."""
        cfg = self.suite.writes
        name = cfg.truth_state(name)

        if name == "s0" and self.suite.truth is not None and self.suite.truth.exists():
            return self.suite.truth

        return cfg.truth_path(name)

    def state_recall(self, name: str, copy: Path | None = None) -> data.Recall | None:
        """The scorer of state `name`, or of a copy of its live rows whose results are positions. It is None when the
        truth file is missing, so the lines carry no recall, as for a build without truth."""
        cfg = self.suite.writes
        key = (cfg.truth_state(name), str(copy))

        if key not in self._state_recalls:
            path = self.state_truth(name)

            if not path.exists() or not cfg.stream.exists():
                self.log(f"no ground truth at {path} or no stream, so searches of {name} get no recall. Make them "
                         f"with writes.py prepare")
                self._state_recalls[key] = None
            else:
                stream = self.stream()
                live = writes.state_mask(cfg, stream, name)
                # A copy's results are positions, and a drift stream's ids aren't its rows, so both map to rows.
                rows = writes.copy_rows(copy) if copy is not None else (None if stream.identity else stream.row_of)
                truth = self.dataset.truth(cfg.id_bound, path)
                self._state_recalls[key] = writes.StateRecall(self.dataset, truth, live, cfg.n, rows)

        return self._state_recalls[key]

    def stream_field(self, system: System, build: Build | None) -> dict | None:
        """The stream a line belongs to, for every line of a write suite."""
        cfg = self.suite.writes

        if cfg is None:
            return None

        origin = build.origin if build is not None and build.origin is not None else build
        own = [job.role for job in system.writes if job.build is origin]
        role = own[0] if own else next((job.role for job in system.writes), "rebuild")
        return {"path": str(cfg.stream), "sha256": self.stream_sha(), "batch": cfg.batch, "role": role}

    def state_label(self, build: Build | None) -> str | None:
        """A line's state: the state it searched, or "fresh" for a build of a copy of the joint ladder's last state, and
        "fresh-i" for a build of the inserts-only ladder's last prefix."""
        cfg = self.suite.writes

        if cfg is None or build is None:
            return None

        if build.state is not None:
            return build.state

        if build.data is not None:
            record = writes.copy_record(build.data)
            final = cfg.final("joint")

            if record is not None:
                return "fresh" if final is not None and record["state"] == final.name else f"fresh-{record['state']}"

        prefix = self.prefix_state(build.n)
        final = cfg.final("inserts")

        if prefix is not None:
            return "fresh-i" if final is not None and prefix == final.name else f"fresh-{prefix}"

        return None

    def rows_arg(self, build: Build) -> int:
        """The --rows of a search of `build`. A state's ids go up to its last insert, past its live count."""
        if build.state is not None:
            return self.suite.writes.n + self.suite.writes.spec(build.state).inserts

        return build.n

    def base_of(self, build: Build) -> Path:
        """The base file a build reads, or a search of it takes its queries beside."""
        return build.data if build.data is not None else self.dataset.base_path

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

    def data_args(self, n: int, threads: int, system: System, k: int | None = None,
                  base: Path | None = None) -> list[str]:
        args = ["--data", str(base or self.dataset.base_path), "--rows", str(n), "--metric", self.dataset.metric,
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
        """What lines record about an index. A state's comes from its record, taken when the driver made the state,
        since searches change some states' files."""
        if build.state is not None and not self.rebuild_s0(build):
            record = self.state_record(build)
            return record.get("identity") if record is not None else None

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

        if system is not None and self.suite.writes is not None:
            line.update(stream=self.stream_field(system, build), state=self.state_label(build))

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

            if self.suite.writes is not None and self.only & (WRITE_PHASES | {"warm", "cold"}):
                self.stream()

            if "stream" in self.only:
                self.run_streams()

            if "latency" in self.only:
                self.run_latency()

            if "maintain" in self.only:
                self.run_maintain()

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
            if not self.selected(b.system):
                continue

            state = "prebuilt" if b.prebuilt else ("done" if self.progress.done(self.build_key(b)) else "to do")
            print(f"build {b.system.name} {b.config_id} n={b.n} seed={b.seed}: {state} -> {b.index}")

        for system in (s for s in self.suite.systems if self.selected(s)):
            label = ":".join(x for x in (system.name, system.variant) if x)

            for job in system.writes:
                for pass_name, ladder in self.passes(job):
                    done = [f"{x.name} {'done' if self.state_done(self.state_of(job, x.name)) else 'to do'}"
                            for x in ladder]
                    print(f"stream {label} {job.build.config_id} {pass_name}: {', '.join(done)}")

                for name in self.latency_states(job):
                    state = "done" if self.latency_done(self.state_of(job, name)) else "to do"
                    print(f"latency {label} {job.build.config_id} {name}: {state}")

            for job in system.maintains:
                for action in job.actions:
                    target = state_build(self.suite, system, job.build, f"{job.state}-{action}")
                    state = "done" if self.state_done(target) else "to do"
                    print(f"maintain {label} {job.build.config_id} {action} on {job.state}: {state}")

        for w in self.suite.warm:
            if not self.selected(w.system):
                continue

            todo = sum(1 for s in w.settings if not self.progress.done(self.warm_key(w, s)))
            at = f" state={w.build.state}" if w.build.state else ""
            print(f"warm {w.system.name} {w.build.config_id}{at} split={w.split} threads={w.threads}: "
                  f"{todo} of {len(w.settings)} settings to do")

        for c in self.suite.cells:
            if not self.selected_cell(c):
                continue

            todo = sum(1 for r in range(c.trials) if not self.progress.done(self.cold_key(c, r)))
            at = f" state={c.build.state}" if c.build.state else ""
            print(f"cold {c.label}{at} group={c.group} params={c.params}: {todo} of {c.trials} trials to do")

    def build_key(self, b: Build) -> str:
        ident = {"kind": "build", "phase": self.suite.phase, "system": b.system.name, "variant": b.system.variant,
                 "config_id": b.config_id, "config": b.config, "dataset": self.suite.dataset, "n": b.n, "seed": b.seed}

        if b.data is not None:
            ident["data"] = str(b.data.relative_to(REPO)) if b.data.is_relative_to(REPO) else str(b.data)

        return schema.run_key(ident)

    def _index_key(self, build: Build) -> str:
        """The index's size, file count, newest change and inode. A rebuilt index gets new run keys. A state's key is
        the one the driver took when it made the state, kept in its record."""
        if build.state is not None and not self.rebuild_s0(build):
            record = self.state_record(build)
            return str(record["index_key"]) if record and record.get("index_key") else f"missing:{build.index}"

        return folder_key(build.index)

    def warm_key(self, w: Warm, params: dict) -> str:
        ident = {"kind": "warm", "phase": self.suite.phase, "system": w.system.name, "variant": w.system.variant,
                 "config_id": w.build.config_id, "dataset": self.suite.dataset, "n": w.build.n, "split": w.split,
                 "params": params, "mode": w.mode, "threads": w.threads, "k": w.k, "index": self._index_key(w.build)}

        # A state's key names it, and a head-to-head line's key stays as it was.
        if w.build.state is not None:
            ident["state"] = w.build.state

        return schema.run_key(ident)

    def cold_key(self, c: Cell, trial: int) -> str:
        """A cold trial's run key. It names the index by the key its cell took before its process started, since a C1
        process may hold the index open, and change it, for as long as the cell runs. A state's trial also names its
        group, since every group holds a ColdIVF cell at the same state and setting."""
        index = c.index_key if c.index_key is not None else self._index_key(c.build)
        ident = {"kind": "cold", "phase": self.suite.phase, "system": c.system.name, "variant": c.system.variant,
                 "config_id": c.build.config_id, "dataset": self.suite.dataset, "n": c.build.n, "split": c.split,
                 "params": c.params, "regime": c.regime, "cpu_state": c.cpu_state, "delay_ms": c.delay_ms,
                 "threads": c.threads, "k": c.k, "next": c.next, "trial": trial, "fresh_clone": c.fresh_clone,
                 "index": index}

        if c.build.state is not None:
            ident.update(state=c.build.state, group=c.group)

        return schema.run_key(ident)

    def write_key(self, system: System, build: Build, mode: str, op: str, batch: int | None, start: Build,
                  target: Build) -> str:
        """A write line's run key. It names the stream, the op and its batch, both states and the state it started
        from, as the driver made it."""
        return schema.run_key({"kind": "write", "phase": self.suite.phase, "system": system.name,
                               "variant": system.variant, "config_id": build.config_id, "stream": self.stream_sha(),
                               "mode": mode, "op": op, "batch": batch, "state_from": start.state,
                               "state_to": target.state, "index": self._index_key(start)})

    def session_key(self, job: WriteJob, mode: str, start: Build, target: Build) -> str:
        """The key of a session's failure line, which counts the session's attempts."""
        return self.write_key(job.system, job.build, mode, "session", None, start, target)

    def maintain_key(self, system: System, build: Build, action: str, threads: int | None, params: dict | None,
                     start: Build, target: Build, run: int | None = None) -> str:
        ident = {"kind": "maintain", "phase": self.suite.phase, "system": system.name, "variant": system.variant,
                 "config_id": build.config_id, "stream": self.stream_sha(), "action": action, "threads": threads,
                 "params": params or {}, "state_from": start.state, "state_to": target.state,
                 "index": self._index_key(start)}

        # Each run of the policy pass is a line of its own.
        if run is not None:
            ident["run"] = run

        return schema.run_key(ident)

    def rebuild_s0(self, build: Build) -> bool:
        """Whether `build` is a rebuild-only system's S0, which is its build itself."""
        return build.state == "s0" and build.origin is not None and role_of(build.system, build.origin) == "rebuild"

    def state_record(self, build: Build) -> dict | None:
        return writes.read_json(writes.record_path(build.index))

    def state_of(self, job: WriteJob | MaintainJob, name: str) -> Build:
        return state_build(self.suite, job.system, job.build, name)

    def state_done(self, build: Build) -> bool:
        """Whether a state is made. Its folder and record are there, the record names this stream, the state it came
        from is made and is the one it came from, and every line of the session that made it is done."""
        if self.rebuild_s0(build):
            return build.index is not None and build.index.exists()

        record = self.state_record(build)

        if record is None or not build.index.is_dir():
            return False

        if build.state == "s0":
            return record.get("from_key") == folder_key(build.origin.index)

        if record.get("stream_sha256") != self.stream_sha() or not record.get("from_state"):
            return False

        start = state_build(self.suite, build.system, build.origin, str(record["from_state"]))

        if not self.state_done(start) or record.get("from_key") != self._index_key(start):
            return False

        return all(self.progress.done(k) for k in record.get("keys", []))

    def state_ready(self, build: Build) -> bool:
        """Whether a state is there to search or to start from. S0 is a clone of the build, made here when missing."""
        if self.state_done(build):
            return True

        return build.state == "s0" and not self.dry_run and self.make_s0(build)

    def make_s0(self, build: Build) -> bool:
        origin = build.origin

        if origin.index is None or not origin.index.exists():
            self.log(f"{build.system.name} has no build {origin.config_id} at {origin.index} to clone as S0")
            return False

        self.log(f"cloning {origin.index} as S0 of {build.system.name} {origin.config_id}")
        self.place_state(build, origin.index, {"pass": "", "round": 0, "from_state": None,
                                               "from_key": folder_key(origin.index), "keys": []})
        return True

    def place_state(self, build: Build, source: Path, record: dict, move: bool = False) -> dict:
        """Makes the folder of state `build` from `source`, as an APFS clone, or with `move` by a rename, and writes its
        record beside it. The folder appears whole under its name or not at all, and the record comes after it, so a
        folder without a record is a state that a crash cut short, which is made again. The record also counts the
        runs of contiguous disk blocks its files take, which show how scattered its writes left them."""
        cfg, target = self.suite.writes, build.index
        tmp = target.with_name(f"{target.name}.tmp-{os.getpid()}")
        writes.remove_tree(tmp)

        if move:
            os.rename(source, tmp)
        else:
            writes.clone_tree(source, tmp)

        writes.remove_tree(writes.record_path(target))
        writes.remove_tree(target)
        os.rename(tmp, target)
        ident = identity(target)
        spec = cfg.spec(build.state)
        full = {"system": build.system.name, "variant": build.system.variant, "config_id": build.config_id,
                "state": build.state, "stream": str(cfg.stream), "stream_sha256": self.stream_sha(),
                "inserts": spec.inserts, "deletes": spec.deletes, "live": cfg.live_count(build.state),
                "index_key": f"{ident['bytes']}:{ident['files']}:{ident['mtime_ns']}:{ident['inode']}",
                "identity": ident, "extents": writes.extents(target), "unshared": False, "batch_id": self.batch_id,
                "made": schema.utc_now()} | record
        writes.write_json(writes.record_path(target), full)
        self.identities.pop(str(target), None)
        return full

    def passes(self, job: WriteJob) -> list[tuple[str, list[writes.StateSpec]]]:
        """Each pass of a write entry with the states it keeps, in the order the pass makes them."""
        out = []

        for pass_name in writes.PASSES:
            ladder = [self.suite.writes.spec(x) for x in job.states]
            ladder = sorted((x for x in ladder if x.pass_name == pass_name), key=lambda x: x.round)

            if ladder:
                out.append((pass_name, ladder))

        return out

    def latency_states(self, job: WriteJob) -> list[str]:
        """The states a write entry's latency runs take: the suite's latency states, with S10i for S10 in the
        inserts-only role, and none for the exact reference."""
        cfg, out = self.suite.writes, []

        if job.role not in ("full", "inserts"):
            return out

        for name in cfg.latency_states:
            spec = cfg.spec(name)

            if job.role == "inserts" and spec.with_deletes:
                twin = [x for x in cfg.ladder("inserts") if x.round == spec.round]
                name = twin[0].name if twin else ""

            if name and (name == "s0" or name in job.states) and name not in out:
                out.append(name)

        return out

    def latency_done(self, build: Build) -> bool:
        record = writes.read_json(writes.latency_path(build.index))

        if record is None or record.get("index_key") != self._index_key(build) or not self.state_done(build):
            return False

        return all(self.progress.done(k) for k in record.get("keys", []))

    def maintain_job(self, system: System, build: Build) -> MaintainJob | None:
        return next((m for m in system.maintains if m.build is build), None)

    def unshared(self, system: System, build: Build) -> bool:
        """Whether the write sessions of a build run on plain copies rather than clones. A system that writes its files
        in place does, since its writes to a clone would each move a block elsewhere on the disk, and a state made that
        way reads slower cold than the same bytes written in place. The exact reference keeps clones, since only its
        recall counts and a copy of its rows for every state would take ~20 GB at 1M."""
        return bool(self.info(system).get("writes_in_place")) and role_of(system, build) != "exact"

    def scratch(self, source: Path, dest: Path, unshared: bool, cached: bool = False) -> None:
        """Makes `dest` from `source` for a session to write to: a clone, or with `unshared` a plain copy whose cached
        pages are dropped unless `cached`, so the session finds its files on disk as it finds a new clone's."""
        if not unshared:
            writes.clone_tree(source, dest)
            return

        writes.copy_tree(source, dest)

        if not cached:
            left = drop_cache(dest, self.suite.wait_s)

            if left.resident_pages:
                self.log(f"{left.resident_pages} pages of the copy {dest} stayed cached")

    def session(self, system: System, args: list[str], threads: int, timeout_s: float,
                terminal: str = "done") -> tuple[list[dict], dict | None]:
        """Runs one write or maintenance session in a fresh adapter process, and returns its events and the one that
        ended it, which is `terminal` when it worked, or an error, unsupported or None."""
        child = self.child(system, args, threads)
        events, end = [], None
        deadline = time.monotonic() + timeout_s
        floor = float(self.suite.gate["disk_floor_gb"])

        try:
            while end is None:
                msg = child.event(1.0)

                if msg is not None:
                    if msg["event"] in (terminal, "error", "unsupported"):
                        end = msg
                    else:
                        events.append(msg)
                elif child.proc.poll() is not None:
                    end = {"event": "error", "message": f"exit {child.proc.returncode}: {child.tail()}"}
                elif time.monotonic() > deadline:
                    child.kill()
                    end = {"event": "error", "message": f"timed out after {timeout_s:.0f} s"}
                elif free_gib(REPO) < min(floor, 20.0):
                    child.kill()
                    end = {"event": "error", "message": f"stopped with only {free_gib(REPO):.0f} GiB of disk free"}
        finally:
            child.close()

        return events, end

    def run_streams(self) -> None:
        for system in self.suite.systems:
            if not self.selected(system):
                continue

            for job in system.writes:
                for pass_name, ladder in self.passes(job):
                    self.run_pass(job, pass_name, ladder)

    def run_pass(self, job: WriteJob, pass_name: str, ladder: list[writes.StateSpec]) -> None:
        """Makes the states of one pass in order. Each comes from the one before it, or from S0 for the first, by one
        session on the pass's work folder, which a crash or failure leaves to be cloned again from the last state."""
        start = self.state_of(job, "s0")
        work = job.build.out.parent / _slug(f"{job.build.config_id}-work-{pass_name}")

        for spec in ladder:
            target = self.state_of(job, spec.name)

            if self.state_done(target):
                self.resumed += 1
                start = target
                continue

            if not self.state_ready(start):
                self.log(f"stream {job.system.name} {pass_name}: {start.state} isn't made, so {spec.name} waits")
                return

            if not self.stream_step(job, pass_name, start, target, work):
                return

            start = target

        writes.remove_tree(work)
        writes.remove_tree(work_marker(work))

    def stream_step(self, job: WriteJob, pass_name: str, start: Build, target: Build, work: Path) -> bool:
        """Makes `target` from `start`, again after a failure or a line that isn't ok, until the attempts run out."""
        key = self.session_key(job, "stream", start, target)

        for _ in range(self.max_attempts + 1):
            if self.state_done(target):
                return True

            if self.progress.done(key):
                self.log(f"stream {job.system.name} {start.state} to {target.state}: no attempts left")
                return False

            self.stream_session(job, pass_name, start, target, work, key)

        return self.state_done(target)

    def prepare_work(self, start: Build, work: Path, unshared: bool = False) -> None:
        """Leaves the work folder holding `start`. It goes on from where the last session left it when that was
        `start` and nothing changed it since, and is cloned from `start` again otherwise, eg. after a killed round.

        With `unshared` every round gets a plain copy of `start`, since the last session's folder became `start`. The
        copy keeps its cached pages where the session before it in this batch made `start`, as a work folder that goes
        on keeps its own, and drops them otherwise, as a new clone has none."""
        marker = writes.read_json(work_marker(work))

        if (not unshared and marker is not None and work.is_dir() and marker.get("holds") == start.state
                and marker.get("key") == self._index_key(start) and marker.get("work_key") == folder_key(work)):
            return

        writes.remove_tree(work_marker(work))
        writes.remove_tree(work)
        cached = unshared and self.copied.pop(str(work), None) == self._index_key(start)
        self.scratch(start.index, work, unshared, cached)

    def stream_session(self, job: WriteJob, pass_name: str, start: Build, target: Build, work: Path,
                       key: str) -> None:
        """One round of a pass on its work folder, which becomes the state. A clone of the folder becomes the state
        and the folder goes on to the next round, or for a system that writes in place the folder itself becomes the
        state by a rename. Its files then hold exactly what its writes left on blocks of their own, and the next round
        copies them again. A plain copy into the state would rewrite every file in one piece, which the writes in place
        of a real index never do."""
        cfg, system = self.suite.writes, job.system
        spec, origin = cfg.spec(target.state), cfg.spec(start.state)
        unshared = self.unshared(system, job.build)
        self.prepare_work(start, work, unshared)
        args = ["write", *self.data_args(cfg.n, job.threads, system), "--index", str(work), "--stream",
                str(cfg.stream), "--mode", "stream", "--pass", pass_name, "--round", str(spec.round), "--from-round",
                str(origin.round), "--batch", str(cfg.batch), "--params", json.dumps(job.params)]

        if spec.policy:
            m = self.maintain_job(system, job.build)
            args += ["--policy", spec.policy, "--maintain-params", json.dumps(m.params if m else {}),
                     "--maintain-threads", str(m.threads if m else 10)]

        self.log(f"stream {system.name} {job.build.config_id} {pass_name}: {start.state} to {target.state}")
        events, end = self.session(system, args, job.threads, job.timeout_s)

        if end is None or end["event"] != "done":
            self.session_failed(job, "stream", start, target, key, end)
            writes.remove_tree(work_marker(work))
            writes.remove_tree(work)
            return

        session = schema.new_run_id()
        lines = self.session_lines(job, "stream", start, target, events, session)
        made = self.place_state(target, work, {"pass": pass_name, "round": spec.round, "from_state": start.state,
                                               "from_key": self._index_key(start), "session": session,
                                               "keys": [x["run_key"] for x in lines], "sealed": end.get("sealed"),
                                               "unshared": unshared}, move=unshared)

        if unshared:
            self.copied[str(work)] = made["index_key"]
        else:
            writes.write_json(work_marker(work), {"holds": target.state, "key": made["index_key"],
                                                  "work_key": folder_key(work)})

        for line in lines:
            self.write(system.name, line)

    def run_latency(self) -> None:
        for system in self.suite.systems:
            if not self.selected(system):
                continue

            for job in system.writes:
                for name in self.latency_states(job):
                    target = self.state_of(job, name)

                    if not self.state_ready(target):
                        self.log(f"latency {system.name} {name}: the state isn't made yet")
                    elif self.latency_done(target):
                        self.resumed += 1
                    else:
                        self.latency_step(job, target)

    def latency_step(self, job: WriteJob, target: Build) -> None:
        """Times single writes on a throwaway clone of a state, or a plain copy for a system that writes in place, again
        after a failure until the attempts run out."""
        cfg, system = self.suite.writes, job.system
        spec = cfg.spec(target.state)
        key = self.session_key(job, "latency", target, target)

        for _ in range(self.max_attempts + 1):
            if self.latency_done(target) or self.progress.done(key):
                return

            folder = self.suite.index_root / "clones.noindex" / f"{os.getpid()}-latency-{os.urandom(3).hex()}"
            clone = folder / target.index.name
            self.scratch(target.index, clone, self.unshared(system, job.build))
            args = ["write", *self.data_args(cfg.n, job.threads, system), "--index", str(clone), "--stream",
                    str(cfg.stream), "--mode", "latency", "--pass", spec.pass_name or "joint", "--round",
                    str(spec.round), "--calls", str(cfg.latency_calls), "--warmup", str(cfg.warmup_calls),
                    "--durable-calls", str(cfg.durable_calls), "--saves", str(cfg.saves), "--reserve",
                    str(cfg.reserve_from), "--seed", str(cfg.latency_seed), "--params", json.dumps(job.params)]
            self.log(f"latency {system.name} {job.build.config_id} at {target.state}")

            try:
                events, end = self.session(system, args, job.threads, job.timeout_s)
            finally:
                writes.remove_tree(folder)

            if end is None or end["event"] != "done":
                self.session_failed(job, "latency", target, target, key, end)
                continue

            session = schema.new_run_id()
            lines = self.session_lines(job, "latency", target, target, events, session)
            writes.write_json(writes.latency_path(target.index),
                              {"index_key": self._index_key(target), "session": session,
                               "keys": [x["run_key"] for x in lines], "made": schema.utc_now()})

            for line in lines:
                self.write(system.name, line)

    def run_maintain(self) -> None:
        for system in self.suite.systems:
            if not self.selected(system):
                continue

            for job in system.maintains:
                source = self.state_of(job, job.state)

                if not self.state_ready(source):
                    self.log(f"maintain {system.name}: {job.state} isn't made yet")
                    continue

                for action in job.actions:
                    target = self.state_of(job, f"{job.state}-{action}")

                    if self.state_done(target):
                        self.resumed += 1
                    else:
                        self.maintain_step(job, action, source, target)

    def maintain_step(self, job: MaintainJob, action: str, source: Build, target: Build) -> None:
        key = self.maintain_key(job.system, job.build, action, job.threads_of(action), job.params, source, target)

        for _ in range(self.max_attempts + 1):
            if self.state_done(target) or self.progress.done(key):
                return

            self.maintain_session(job, action, source, target, key)

    def maintain_session(self, job: MaintainJob, action: str, source: Build, target: Build, key: str) -> None:
        """One action on a clone of the source state, or a plain copy for a system that writes in place, in a fresh
        adapter process, whose folder becomes the state <source>-<action>. The driver measures the folder for its peak
        disk use, as for a build."""
        cfg, system, threads = self.suite.writes, job.system, job.threads_of(action)
        work = job.build.out.parent / _slug(f"{job.build.config_id}-work-{action}")
        unshared = self.unshared(system, job.build)
        writes.remove_tree(work)
        self.scratch(source.index, work, unshared)
        floor, free = float(self.suite.gate["disk_floor_gb"]), free_gib(REPO)
        fields: dict = {"action": action, "threads": threads, "params": job.params}

        if free - job.disk_gb < floor:
            end = {"event": "error", "message": f"{free:.0f} GiB free, the action needs {job.disk_gb:.0f} GiB above "
                                                f"the {floor:.0f} GiB floor"}
            events: list = []
        else:
            args = ["maintain", *self.data_args(cfg.n, threads, system), "--index", str(work), "--action", action,
                    "--params", json.dumps(job.params)]
            before = writes.folder_bytes(work)
            self.log(f"maintain {system.name} {job.build.config_id} {action} on a clone of {source.state}")
            sampler = DiskSampler(work, DISK_SAMPLE_S)

            try:
                events, end = self.session(system, args, threads, job.timeout_s, terminal="maintain")
            finally:
                peak, _ = sampler.stop()

            if end is not None and end["event"] == "maintain":
                fields = dict(end["fields"])
                fields.update(peak_disk_bytes=peak, index_bytes_before=before,
                              index_bytes_after=writes.folder_bytes(work))

        session = schema.new_run_id()
        line = self.maintain_line(system, job.build, source, target, fields, session)

        if end is None or end["event"] != "maintain":
            line.update(status="unsupported" if end and end["event"] == "unsupported" else "failed",
                        error=((end or {}).get("message") or "no result")[:4000])
            writes.remove_tree(work)
            self.write(system.name, line)
            return

        data_file = (line["extra"] or {}).get("data_file")
        rel = os.path.relpath(data_file, work) if data_file and Path(data_file).is_relative_to(work) else data_file
        self.place_state(target, work, {"pass": "", "round": cfg.spec(target.state).round, "action": action,
                                        "from_state": source.state, "from_key": self._index_key(source),
                                        "session": session, "keys": [line["run_key"]], "data_file": rel,
                                        "unshared": unshared}, move=True)

        if action in job.same_file:
            line["extra"]["same_file"] = self.same_file(job, action, job.same_file[action], rel)

        self.write(system.name, line)

    def same_file(self, job: MaintainJob, action: str, other: str, rel: str | None) -> dict:
        """Checks that an action's data file has the same bytes as another action's, eg. the job's and the auto
        run's, which are one run of kind auto with the same seed."""
        theirs = self.state_of(job, f"{job.state}-{other}")
        record = self.state_record(theirs) if self.state_done(theirs) else None
        mine = self.state_of(job, f"{job.state}-{action}").index

        if record is None or not rel or not record.get("data_file"):
            return {"action": other, "match": None}

        a = schema.file_sha256(Path(mine) / rel)
        b = schema.file_sha256(Path(theirs.index) / str(record["data_file"]))

        if a != b:
            self.log(f"maintain {job.system.name}: the data files of {action} and {other} differ")

        return {"action": other, "match": a == b, "sha256": a, "other_sha256": b}

    def session_lines(self, job: WriteJob, mode: str, start: Build, target: Build, events: list[dict],
                      session: str) -> list[dict]:
        """The lines of a session's op and maintain events, which every name the session."""
        lines = []

        for event in events:
            if event["event"] == "op":
                lines.append(self.write_line(job, mode, start, target, event["fields"], session))
            elif event["event"] == "maintain":
                run = (event["fields"].get("extra") or {}).get("run")
                lines.append(self.maintain_line(job.system, job.build, start, target, event["fields"], session, run,
                                                runs_in=target))

        return lines

    def write_line(self, job: WriteJob, mode: str, start: Build, target: Build, fields: dict, session: str) -> dict:
        fields = dict(fields)
        extra = fields.pop("extra", {}) or {}
        unsupported = fields.pop("unsupported", None)
        key = self.write_key(job.system, job.build, mode, fields["op"], fields.get("batch"), start, target)
        line = self.base_line("write", job.system, start, key, self.progress.attempts(key) + 1, job.threads)
        line.update({k: v for k, v in fields.items() if k in schema.FIELDS["write"]})
        line.update(mode=mode, state_from=start.state, state_to=target.state, state=target.state, n=target.n)
        line["extra"] = extra | {k: v for k, v in fields.items() if k not in schema.FIELDS["write"]} | \
            {"session": session}

        if unsupported is not None:
            line.update(status="unsupported", error=str(unsupported)[:4000])

        moved = self.vm_moved({"swapins": line.get("swapins_delta"), "compressions": line.get("compressions_delta")})

        if moved and line["status"] == "ok":
            line.update(status="invalid", error=f"{moved} during the timed calls")

        return line

    def maintain_line(self, system: System, build: Build, start: Build, target: Build, fields: dict, session: str,
                      run: int | None = None, runs_in: Build | None = None) -> dict:
        """A maintain line. A run of the policy pass has the pass's state in both of its states."""
        fields = dict(fields)
        extra = fields.pop("extra", {}) or {}
        start_name = runs_in.state if runs_in is not None else start.state
        key = self.maintain_key(system, build, str(fields.get("action")), fields.get("threads"), fields.get("params"),
                                start, target, run)
        line = self.base_line("maintain", system, start, key, self.progress.attempts(key) + 1,
                              int(fields.get("threads") or 1))
        line.update({k: v for k, v in fields.items() if k in schema.FIELDS["maintain"]})
        line.update(state_from=start_name, state_to=target.state, state=target.state,
                    n=fields.get("rows_live") or target.n)
        line["extra"] = extra | {k: v for k, v in fields.items() if k not in schema.FIELDS["maintain"]} | \
            {"session": session}
        return line

    def session_failed(self, job: WriteJob, mode: str, start: Build, target: Build, key: str,
                       end: dict | None) -> None:
        """Records a session that didn't finish, under the session's own key, so its attempts count."""
        line = self.base_line("write", job.system, start, key, self.progress.attempts(key) + 1, job.threads)
        status = "unsupported" if end is not None and end["event"] == "unsupported" else "failed"
        line.update(mode=mode, op="session", state_from=start.state, state_to=target.state, state=target.state,
                    n=target.n, threads=job.threads, status=status,
                    error=((end or {}).get("message") or "the session sent no result")[:4000])
        self.log(f"{mode} {job.system.name} {start.state} to {target.state}: {status}, {line['error'][:200]}")
        self.write(job.system.name, line)

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

        if build.data is not None:
            line.update(dataset=build.data.name[: -len(".base.fbin")], extra={"data": str(build.data)})

            if not build.data.exists() or data.header(build.data)[0] < build.n:
                line.update(status="failed", error=f"{build.data} isn't there or holds fewer than {build.n} rows. Make "
                                                   f"it with writes.py copy")
                self.write(system.name, line)
                return

        build.out.mkdir(parents=True)
        args = ["build", *self.data_args(build.n, build.threads, system, base=self.base_of(build)), "--config",
                json.dumps(build.config), "--out", str(build.out), "--seed", str(build.seed)]

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
            extra=(line.get("extra") or {}) | extra | {k: v for k, v in fields.items()
                                                       if k not in schema.FIELDS["build"]},
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

        if job.build.state is not None and not self.state_ready(job.build):
            self.log(f"warm {system.name} {job.build.config_id}: state {job.build.state} isn't made yet")
            return

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

        # The lines name the index as its build left it, read before the job's process opens it. A system that writes
        # to its index while it is open, eg. AsterVec's store, changes and removes files there until it closes.
        self.index_identity(job.build)
        ids_dir = self.out_dir / "ids"
        tmp_ids = Path(tempfile.mkdtemp(prefix="ids-", dir=self.out_dir))
        args = ["warm", *self.data_args(self.rows_arg(job.build), job.threads, system, job.k,
                                        base=self.base_of(job.build)),
                "--index", str(job.build.index), "--grid", json.dumps([job.settings[i] for i in todo]), "--split",
                job.split, "--mode", job.mode, "--ids-dir", str(tmp_ids), "--passes", str(job.passes), "--warmup",
                str(job.warmup), "--qps-seconds", str(job.qps_seconds), "--qps-repeats", str(job.qps_repeats)]
        probes = None

        if job.findability and job.build.state is not None and 0 in todo:
            probes = writes.probe_rows(self.suite.writes, self.stream(), job.build.state)
            np.save(tmp_ids / "probe-rows.npy", probes)
            args += ["--probes", str(tmp_ids / "probe-rows.npy")]
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
                    recall = self.scorer(job.build)

                    if recall is not None:
                        line.update(recall.summary(np.asarray(msg["qids"]), np.load(path)))

                    stale = self.stale(job.build, np.asarray(msg["qids"]), np.load(path))

                    if stale:
                        line["extra"]["stale"] = stale

                    line["ids_file"] = str(path)

                if msg.get("probes_file") and probes is not None:
                    found = np.load(msg["probes_file"])
                    stream = self.stream()
                    rows = None if stream.identity else stream.row_of
                    line["extra"]["findability"] = writes.findability(self.dataset, probes, found,
                                                                      self.suite.writes.n, rows)

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
                    gt=self.scorer_record(job.build), overhead_us=self.overhead_of(job.system).get("overhead_us"))
        return line

    def stale(self, build: Build, qids: np.ndarray, found: np.ndarray) -> dict:
        """A rebuild-only system's S0 results scored against each state's truth, i.e. what serving the old index costs
        until the rebuild. Empty for any other search."""
        cfg = self.suite.writes

        if cfg is None or build.state != "s0" or not self.rebuild_s0(build):
            return {}

        out = {}

        for spec in cfg.states.values():
            if spec.pass_name in ("joint", "inserts"):
                recall = self.state_recall(spec.name)

                if recall is not None:
                    summary = recall.summary(qids, found)
                    out[spec.name] = {k: summary.get(k) for k in ("recall10", "recall10_ids_only", "dead_returned",
                                                                   "recall10_inserted")}

        return out

    def query_blocks(self, cell: Cell) -> tuple[int, int, int, int]:
        """Where the trials of `cell` take their queries. The m cells of its group that search its index with its split
        run their trials in the same rounds, so they share the split's positions, in blocks as wide as the widest of
        their trials. The i-th cell's trial t takes block (t + i·d) mod b, where b is the number of blocks the cells go
        through before they come back to the first, and d, b / m rounded down, spaces the cells evenly. b is their most
        trials, or 2m for fewer, but at most the blocks that fit in the split. So in a round no two cells search the
        same query while the split holds a block for each, and none searches one that another searched in the round
        before while it holds two for each. Cells of as many trials, at least 2m, search the same queries over them.
        It returns the block width, b, d and i."""
        key = (cell.group, str(cell.build.index), cell.split)
        cells = [c for c in self.suite.cells if (c.group, str(c.build.index), c.split) == key]
        place = next((i for i, c in enumerate(cells) if c is cell), None)
        cells, place = (cells, place) if place is not None else ([cell], 0)
        width = max(1 + c.next for c in cells)
        fit = max(1, len(data.split_ids(self.dataset.queries.shape[0], cell.split)) // width)
        blocks = min(max(max(c.trials for c in cells), 2 * len(cells)), fit)
        return width, blocks, max(1, blocks // len(cells)), place

    def trial_queries(self, cell: Cell, trial: int) -> list[int]:
        """The queries of trial r, 1 + `cell.next` positions from the start of the block that `query_blocks` gives
        it."""
        return self.queries_of(cell, range(trial, trial + 1))

    def queries_of(self, cell: Cell, trials: range) -> list[int]:
        """The queries of the trials `trials` of `cell`, one trial after another."""
        qids = data.split_ids(self.dataset.queries.shape[0], cell.split)
        width, blocks, step, place = self.query_blocks(cell)
        starts = [((t + place * step) % blocks) * width for t in trials]
        return [int(qids[(s + j) % len(qids)]) for s in starts for j in range(1 + cell.next)]

    def warmup_queries(self, cell: Cell, count: int = WARMUP_QUERIES) -> list[int]:
        """Queries for the searches a cell's process runs before READY, none of which any trial on the cell's index in
        its group searches. So a cache that a warm-up fills can't answer a timed query. They come from the other split
        of a tune or test cell, and otherwise from the end of the cell's own split."""
        total = self.dataset.queries.shape[0]
        index = (cell.group, str(cell.build.index))
        near = [c for c in self.suite.cells if (c.group, str(c.build.index)) == index and c is not cell] + [cell]
        used = {q for c in near for q in self.queries_of(c, range(c.trials))}
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
        cells = [c for c in self.suite.cells if self.selected_cell(c)]

        for group in sorted({c.group for c in cells}):
            members = [c for c in cells if c.group == group]
            self.log(f"cold group {group}: {len(members)} cells")
            self.cold_group(members)

    def cold_group(self, cells: list[Cell]) -> None:
        maps: dict[str, evictor.Evictor] = {}
        specs: dict[int, list[dict]] = {}

        for cell in cells:
            index = cell.build.index

            if cell.build.state is not None and not self.state_ready(cell.build):
                # A state the stream phase hasn't made yet waits for a later batch, without a line.
                cell.skip, cell.reported = f"state {cell.build.state} isn't made yet", True
                self.log(f"cold {cell.label}: {cell.skip}")
                continue

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
            self.index_identity(cell.build)

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
                    predecessor=previous, gt=self.scorer_record(cell.build))
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
        args = ["cold", *self.data_args(self.rows_arg(cell.build), cell.threads, cell.system, cell.k,
                                        base=self.base_of(cell.build)),
                "--index", str(cell.build.index), "--params", json.dumps(cell.params), "--queries",
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

        recall = self.scorer(cell.build)

        if line.get("ids") and recall is not None:
            found = np.asarray([line["ids"]])

            if isinstance(recall, writes.StateRecall):
                line.update(recall.trial(int(line["query_id"]), found[0], min(10, cell.k)))
            else:
                ties, plain = recall.per_query(np.asarray([line["query_id"]]), found, min(10, cell.k))
                line.update(recall10=float(ties[0]), recall10_ids_only=float(plain[0]))

            stale = self.stale(cell.build, np.asarray([line["query_id"]]), found)

            if stale:
                line["extra"]["stale"] = stale

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
    p.add_argument("--only", default="",
                   help="a comma-separated subset of build, stream, latency, maintain, warm, cold and refs")
    p.add_argument("--systems", default="",
                   help="a comma-separated subset of the suite's systems, each a name or name:variant")
    p.add_argument("--groups", default="", help="a comma-separated subset of the cold groups")
    p.add_argument("--gate", default=None, choices=("enforce", "warn", "off"))
    p.add_argument("--max-attempts", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    suite = load_suite(args.suite)
    only = {x.strip() for x in args.only.split(",") if x.strip()} or None
    systems = {x.strip() for x in args.systems.split(",") if x.strip()} or None
    groups = {x.strip() for x in args.groups.split(",") if x.strip()} or None
    driver = Driver(suite, only=only, systems=systems, gate_mode=args.gate, dry_run=args.dry_run,
                    max_attempts=args.max_attempts, groups=groups)
    return driver.run()


if __name__ == "__main__":
    sys.exit(main())
