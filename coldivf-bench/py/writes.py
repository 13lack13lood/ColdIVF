"""The write benchmark's stream, its states, their ground truth, and the copy of a state's live rows.

    python writes.py stream SUITE.toml               makes the suite's stream file, or checks the one there
    python writes.py truth SUITE.toml [--engine auto|numpy|rust] [--threads 8]
    python writes.py copy SUITE.toml [--state s10]   writes the live rows of a state as a dataset of their own
    python writes.py prepare SUITE.toml              all three
    python writes.py show SUITE.toml                 each state's rounds, counts and files

A stream file is in the COLDSTRM format of coldivf-bench/src/stream.rs. It is little-endian, and starts with the magic
`COLDSTRM`, a u32 version and a u32 count of rounds. Each round holds three u32 counts, of its inserts, deletes and
upserts, then the inserted rows as u32, the deleted vectors as u64, and each upsert as a u64 and a u32 row. The first
round is the base, the build's rows 0 to n. A stream names a vector by the row it was inserted with, and deletes
vectors drawn at random from the ones live at the round's start. An index names a vector by its id: a base row's id is
its row, and each insert takes the next id after the last, as a library that numbers its inserts gives them. The
uniform pattern inserts the rows after the base in row order, so there every id is its row. The drift pattern inserts
the rows nearest a few topics that k-means finds among later rows, so there ids map to rows through `Stream.row_of`.

A state is what an index holds after some rounds. The suite's [writes.states] table names each by its inserts and
deletes since S0. A state with deletes is on the joint ladder, one without is on the inserts-only ladder, and one with
a policy makes the joint ladder's calls with a maintenance run whenever one is due. A maintained state, eg. s10-fold, is
S10 after one action and holds S10's vectors. Each state's truth is the exact top 100 over the rows it holds, scored
by `StateRecall`, which counts a returned id the state doesn't hold as a miss.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tomllib
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np

PY = Path(__file__).resolve().parent

if str(PY) not in sys.path:
    sys.path.insert(0, str(PY))

import data  # noqa: E402
import schema  # noqa: E402

REPO = PY.parents[1]

MAGIC = b"COLDSTRM"
VERSION = 1
MASK64 = (1 << 64) - 1

PASSES = ("joint", "inserts", "policy")
ROLES = ("full", "inserts", "rebuild", "exact")
LEVELS = ("memory", "process", "power")
MODES = ("latency", "stream")

WRITES_KEYS = {"stream", "inserts_from", "batch", "latency_states", "latency_calls", "warmup_calls", "durable_calls",
               "saves", "reserve_from", "seed", "latency_seed", "probes", "states", "pattern", "drift_from",
               "drift_rows", "topics", "hot_topics", "topic_sample", "kmeans_rounds"}
PATTERNS = ("uniform", "drift")
STATE_KEYS = {"inserts", "deletes", "policy"}

# A state's name, eg. s5 or s10i. A maintained state adds its action after a dash, eg. s10-fold.
STATE_NAME = re.compile(r"^[a-z][a-z0-9]*$")

# Above this many rows the truth runs in the gt binary, whose distances match the head-to-head's truth files.
NUMPY_TRUTH_ROWS = 200_000
TRUTH_K = 100
# Rows of the base that one step of a copy, of the numpy truth or of a drift stream's scan reads at a time.
CHUNK = 65536
TRUTH_CHUNK = 8192
DRIFT_CHUNK = 16384
# The most bytes of a file that one write of a plain copy takes.
COPY_CHUNK = 1 << 24


class StreamError(ValueError):
    pass


class Rng:
    """SplitMix64, the generator of coldivf-bench/src/stream.rs, so the same seed draws the same numbers in both."""

    def __init__(self, seed: int):
        self.state = int(seed) & MASK64

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return z ^ (z >> 31)

    def below(self, n: int) -> int:
        """A number below `n`, each as likely as the others, by Lemire's method as the Rust generator draws it."""
        if n <= 0:
            raise ValueError("below needs a bound above 0")

        floor = ((-n) & MASK64) % n

        while True:
            m = self.next_u64() * n

            if m & MASK64 >= floor:
                return m >> 64


def draw(rng: Rng, items: np.ndarray, count: int) -> np.ndarray:
    """`count` items of `items` drawn at random, all different, by a partial shuffle that leaves `items` as it is."""
    pool = np.array(items, dtype=np.int64)

    if count > len(pool):
        raise StreamError(f"can't draw {count} of {len(pool)} items")

    for i in range(count):
        j = i + rng.below(len(pool) - i)
        pool[i], pool[j] = pool[j], pool[i]

    return pool[:count].copy()


@dataclass
class Round:
    """One round of a stream: the rows it inserts as new vectors, the ids it deletes, and its upserts as (id, row)."""

    inserts: np.ndarray
    deletes: np.ndarray
    upserts: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.int64))


@dataclass
class Stream:
    base: np.ndarray
    rounds: list[Round]

    @property
    def base_rows(self) -> int:
        return len(self.base)

    @cached_property
    def row_of(self) -> np.ndarray:
        """The row of each id: the base's rows, then each insert's in the order the stream makes them."""
        return np.concatenate([np.asarray(self.base, dtype=np.int64)]
                              + [np.asarray(r.inserts, dtype=np.int64) for r in self.rounds])

    @property
    def id_bound(self) -> int:
        """One past the last id the stream gives, i.e. the base and every insert."""
        return len(self.row_of)

    @property
    def row_bound(self) -> int:
        """One past the last row the stream names."""
        return int(self.row_of.max()) + 1 if len(self.row_of) else 0

    @cached_property
    def identity(self) -> bool:
        """Whether every id is its row, as in a uniform stream."""
        return bool(np.array_equal(self.row_of, np.arange(len(self.row_of))))

    @cached_property
    def _sorted(self) -> tuple[np.ndarray, np.ndarray]:
        order = np.argsort(self.row_of, kind="stable")
        return self.row_of[order], order

    def ids(self, rows: np.ndarray) -> np.ndarray:
        """The id of each row of `rows`, which the stream must name."""
        rows = np.asarray(rows, dtype=np.int64)

        if self.identity:
            return rows.copy()

        sorted_rows, order = self._sorted
        at = np.searchsorted(sorted_rows, rows)

        if len(rows) and (at.max() >= len(sorted_rows) or not np.array_equal(sorted_rows[at], rows)):
            raise StreamError("the stream names no vector by one of these rows")

        return order[at].astype(np.int64)

    def check(self) -> None:
        """Raises StreamError unless the base is rows 0 to n, so a base row's id is its row, and every insert is a row
        of its own, outside the base and inserted once, so ids and rows map one to one."""
        if not np.array_equal(self.base, np.arange(self.base_rows)):
            raise StreamError("the stream's base must be rows 0 to n in order")

        inserts = self.row_of[self.base_rows :]

        if len(inserts) and (inserts.min() < self.base_rows or len(np.unique(inserts)) != len(inserts)):
            raise StreamError("the stream must insert each row once, and no row of its base")

        for r, rnd in enumerate(self.rounds, 1):
            if len(rnd.upserts):
                raise StreamError(f"round {r} has upserts, which the write benchmark doesn't take")


def write_stream(path: Path | str, stream: Stream) -> None:
    """Writes `stream` to `path` through a temporary file, synced and renamed, so a crash never leaves half a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    base = Round(stream.base, np.zeros(0, dtype=np.int64))

    with open(tmp, "wb") as f:
        f.write(MAGIC + struct.pack("<II", VERSION, len(stream.rounds) + 1))

        for rnd in [base, *stream.rounds]:
            ups = np.asarray(rnd.upserts, dtype=np.int64).reshape(-1, 2)
            f.write(struct.pack("<III", len(rnd.inserts), len(rnd.deletes), len(ups)))
            f.write(np.asarray(rnd.inserts, dtype="<u4").tobytes())
            f.write(np.asarray(rnd.deletes, dtype="<u8").tobytes())

            for uid, row in ups.tolist():
                f.write(struct.pack("<QI", uid, row))

        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp, path)


def read_stream(path: Path | str) -> Stream:
    """Reads a stream file. A file that isn't one, or whose base holds deletes or upserts, raises StreamError."""
    raw = Path(path).read_bytes()

    if len(raw) < 16 or raw[:8] != MAGIC or struct.unpack_from("<I", raw, 8)[0] != VERSION:
        raise StreamError(f"{path} isn't a stream file of version {VERSION}")

    count, at, rounds = struct.unpack_from("<I", raw, 12)[0], 16, []

    if count == 0:
        raise StreamError(f"{path} holds no base")

    for _ in range(count):
        if at + 12 > len(raw):
            raise StreamError(f"{path} ends inside a round")

        ins, dels, ups = struct.unpack_from("<III", raw, at)
        at += 12
        end = at + 4 * ins + 8 * dels + 12 * ups

        if end > len(raw):
            raise StreamError(f"{path} ends inside a round")

        inserts = np.frombuffer(raw, dtype="<u4", count=ins, offset=at).astype(np.int64)
        deletes = np.frombuffer(raw, dtype="<u8", count=dels, offset=at + 4 * ins).astype(np.int64)
        pairs = np.frombuffer(raw, dtype=np.dtype([("id", "<u8"), ("row", "<u4")]), count=ups,
                              offset=at + 4 * ins + 8 * dels)
        upserts = np.stack([pairs["id"].astype(np.int64), pairs["row"].astype(np.int64)], axis=1) if ups else \
            np.zeros((0, 2), dtype=np.int64)
        rounds.append(Round(inserts, deletes, upserts))
        at = end

    if at != len(raw):
        raise StreamError(f"{path} goes on past its last round")

    base = rounds.pop(0)

    if len(base.deletes) or len(base.upserts):
        raise StreamError(f"{path} has deletes or upserts in its base")

    return Stream(base.inserts, rounds)


@dataclass(frozen=True)
class StateSpec:
    """A state of the stream: its inserts and deletes since S0, the stream rounds it holds, and for the policy pass the
    run kind it starts. A maintained state also names the state it came from and its action."""

    name: str
    inserts: int
    deletes: int
    round: int
    policy: str | None = None
    origin: str | None = None
    action: str | None = None

    @property
    def pass_name(self) -> str:
        """The pass that makes it: "joint", "inserts" or "policy", or "" for S0 and a maintained state."""
        if self.round == 0 or self.action is not None:
            return ""

        return "policy" if self.policy else ("inserts" if self.deletes == 0 else "joint")

    @property
    def with_deletes(self) -> bool:
        return self.deletes > 0


@dataclass
class WritesConfig:
    """The suite's [writes] table, with every default filled in and checked against the suite's n."""

    stream: Path
    dataset: str
    n: int
    inserts_from: int
    batch: int
    latency_states: list[str]
    latency_calls: int
    warmup_calls: int
    durable_calls: int
    saves: int
    reserve_from: int
    seed: int
    latency_seed: int
    probes: int
    # Each round's inserts and deletes, from the joint ladder's states.
    rounds: list[tuple[int, int]]
    states: dict[str, StateSpec]
    # Where the inserts come from. A drift stream takes them among `drift_rows` rows from `drift_from` on, the rows
    # nearest `hot_topics` of the `topics` clusters that `kmeans_rounds` rounds of k-means find in `topic_sample` of
    # those rows.
    pattern: str = "uniform"
    drift_from: int = 0
    drift_rows: int | None = None
    topics: int = 200
    hot_topics: int = 10
    topic_sample: int = 100000
    kmeans_rounds: int = 20

    @property
    def total_inserts(self) -> int:
        return sum(i for i, _ in self.rounds)

    @property
    def id_bound(self) -> int:
        return self.n + self.total_inserts

    @property
    def reserved(self) -> int:
        """The reserved rows each latency run inserts, after its warm-up and timed calls and its durable inserts."""
        return self.warmup_calls + self.latency_calls + self.durable_calls

    @property
    def probe_seed(self) -> int:
        """The seed of the findability check's draws, after the stream's own seed and the latency runs'."""
        return self.seed + 2

    @property
    def stem(self) -> str:
        name = self.stream.name
        return name[: -len(".stream")] if name.endswith(".stream") else self.stream.stem

    @property
    def recipe(self) -> dict:
        """What makes the stream, which its record beside the file keeps, so a suite can't take another's stream."""
        out = {"pattern": self.pattern, "dataset": self.dataset, "n": self.n, "rounds": self.rounds, "seed": self.seed}

        if self.pattern == "drift":
            out.update(drift_from=self.drift_from, drift_rows=self.drift_rows, topics=self.topics,
                       hot_topics=self.hot_topics, topic_sample=self.topic_sample, kmeans_rounds=self.kmeans_rounds)

        return out

    def spec(self, name: str) -> StateSpec:
        """The state `name`, eg. "s5", "s0" or a maintained "s10-fold"."""
        if name in self.states:
            return self.states[name]

        origin, _, action = name.partition("-")

        if action and origin in self.states and STATE_NAME.match(action):
            o = self.states[origin]
            return StateSpec(name, o.inserts, o.deletes, o.round, None, origin, action)

        raise StreamError(f"no state {name}, expected one of {sorted(self.states)} or one with an action")

    def truth_state(self, name: str) -> str:
        """The state whose truth `name` takes. A policy state or a maintained one holds the vectors of the plain state
        with its counts."""
        spec = self.spec(name)

        if spec.action is not None:
            return self.truth_state(spec.origin)

        if spec.policy:
            for other in self.states.values():
                if not other.policy and (other.inserts, other.deletes) == (spec.inserts, spec.deletes):
                    return other.name

        return spec.name

    def live_count(self, name: str) -> int:
        spec = self.spec(name)
        return self.n + spec.inserts - spec.deletes

    def truth_path(self, name: str) -> Path:
        return self.stream.with_name(f"{self.stem}-{self.truth_state(name)}.gt.ibin")

    def copy_base(self, name: str) -> Path:
        """The base file of the copy of a state's live rows. Its name differs from the state's truth file's, since the
        copy's own truth, by position, sits beside it under the copy's name."""
        return self.stream.with_name(f"{self.stem}-{self.truth_state(name)}-live.base.fbin")

    def ladder(self, pass_name: str) -> list[StateSpec]:
        """The states of one pass, in the order the pass makes them."""
        return sorted((s for s in self.states.values() if s.pass_name == pass_name), key=lambda s: s.round)

    def final(self, pass_name: str) -> StateSpec | None:
        """The last state of a pass, eg. S10 of the joint ladder."""
        ladder = self.ladder(pass_name)
        return ladder[-1] if ladder else None


def parse_writes(table: dict, n: int, dataset: str) -> WritesConfig:
    """Checks a suite's [writes] table and fills in its defaults. Unknown keys are errors, as everywhere in a suite."""
    unknown = sorted(set(table) - WRITES_KEYS)

    if unknown:
        raise StreamError(f"[writes] has unknown keys {unknown}")

    if "stream" not in table:
        raise StreamError("[writes] needs stream, the stream file")

    seed = int(table.get("seed", 1))
    inserts_from = int(table.get("inserts_from", n))
    pattern = str(table.get("pattern", "uniform"))

    if pattern not in PATTERNS:
        raise StreamError(f"pattern must be one of {PATTERNS}, not {pattern}")

    if inserts_from != n and pattern == "uniform":
        raise StreamError(f"inserts_from must be the base count {n}, so every insert's id is its row")

    joint, inserts, policy = [], [], []

    for name, raw in (table.get("states") or {}).items():
        if not STATE_NAME.match(name) or name == "s0":
            raise StreamError(f"state {name!r} must be letters and digits, start with a letter and not be s0")

        bad = sorted(set(raw) - STATE_KEYS)

        if bad:
            raise StreamError(f"state {name} has unknown keys {bad}")

        spec = (name, int(raw.get("inserts", 0)), int(raw.get("deletes", 0)), raw.get("policy"))

        if spec[1] <= 0:
            raise StreamError(f"state {name} needs inserts")

        (policy if spec[3] else joint if spec[2] > 0 else inserts).append(spec)

    # The joint ladder's states set the rounds, or the inserts-only ladder's when there is no joint one.
    steps = sorted({(i, d) for _, i, d, _ in joint}) or sorted({(i, 0) for _, i, _, _ in inserts})
    rounds, last = [], (0, 0)

    for i, d in steps:
        if i <= last[0] or d < last[1]:
            raise StreamError("each state of the joint ladder must hold more inserts, and no fewer deletes, than the "
                              "one before")

        rounds.append((i - last[0], d - last[1]))
        last = (i, d)

    cum = [sum(r[0] for r in rounds[: k + 1]) for k in range(len(rounds))]
    states = {"s0": StateSpec("s0", 0, 0, 0)}

    for name, i, d, pol in joint + inserts + policy:
        if i not in cum:
            raise StreamError(f"state {name} holds {i} inserts, which no round of the stream ends at")

        r = cum.index(i) + 1

        if pol and (i, d) not in {(x, y) for _, x, y, _ in joint}:
            raise StreamError(f"policy state {name} must make the calls of a state of the joint ladder")

        if d and not pol and steps[r - 1][1] != d:
            raise StreamError(f"state {name} holds {d} deletes, not the {steps[r - 1][1]} of its round")

        states[name] = StateSpec(name, i, d, r, str(pol) if pol else None)

    total = cum[-1] if cum else 0
    reserve_from = int(table.get("reserve_from", n + total))

    if reserve_from < n + total:
        raise StreamError(f"reserve_from must be at least {n + total}, past the stream's last insert")

    cfg = WritesConfig(
        stream=_path(table["stream"]), dataset=dataset, n=n, inserts_from=inserts_from,
        batch=int(table.get("batch", 1000)), latency_states=[str(s) for s in table.get("latency_states", [])],
        latency_calls=int(table.get("latency_calls", 1000)), warmup_calls=int(table.get("warmup_calls", 100)),
        durable_calls=int(table.get("durable_calls", 200)), saves=int(table.get("saves", 5)),
        reserve_from=reserve_from, seed=seed, latency_seed=int(table.get("latency_seed", seed + 1)),
        probes=int(table.get("probes", 1000)), rounds=rounds, states=states, pattern=pattern,
        topics=int(table.get("topics", 200)), hot_topics=int(table.get("hot_topics", 10)),
        topic_sample=int(table.get("topic_sample", 100000)), kmeans_rounds=int(table.get("kmeans_rounds", 20)),
    )
    # A drift stream's candidates start past the reserved rows of the latency runs, so no row serves both.
    cfg.drift_from = int(table.get("drift_from", reserve_from + cfg.reserved))
    cfg.drift_rows = int(table["drift_rows"]) if table.get("drift_rows") is not None else None

    if cfg.batch <= 0:
        raise StreamError("batch must be at least 1")

    if pattern == "drift" and (cfg.drift_from < reserve_from + cfg.reserved or not 0 < cfg.hot_topics <= cfg.topics):
        raise StreamError("a drift stream needs drift_from past the reserved rows, and hot_topics from 1 to topics")

    for name in cfg.latency_states:
        cfg.spec(name)

    return cfg


def _path(value: str) -> Path:
    path = Path(os.path.expanduser(os.path.expandvars(str(value))))
    return path if path.is_absolute() else REPO / path


def load_config(suite: Path | str) -> WritesConfig:
    """The [writes] table of a suite file."""
    doc = tomllib.loads(Path(suite).read_text())

    if "writes" not in doc:
        raise StreamError(f"{suite} has no [writes] table")

    return parse_writes(doc["writes"], int(doc["n"]), str(doc["dataset"]))


def make_stream(cfg: WritesConfig, dataset: data.Dataset | None = None) -> Stream:
    """The stream the config describes. The base is rows 0 to n. Each round inserts the next rows of the pattern's
    inserts, the rows after the base in row order for the uniform pattern, and deletes vectors drawn uniformly at random
    with the config's seed from the ones live at the round's start, as the Rust maker's rounds draw them. So the calls
    of a round can come in any order and leave the same state. The drift pattern needs the dataset, and the uniform one
    takes it only to check its rows."""
    need = cfg.reserve_from + cfg.reserved

    if dataset is not None and dataset.count < need:
        raise StreamError(f"the dataset holds {dataset.count} rows, but the stream and its latency runs need {need}")

    if cfg.pattern == "drift":
        if dataset is None:
            raise StreamError("a drift stream needs the dataset")

        inserts = drift_inserts(cfg, dataset)
    else:
        inserts = np.arange(cfg.n, cfg.n + cfg.total_inserts, dtype=np.int64)

    rng = Rng(cfg.seed)
    live = list(range(cfg.n))
    at, rounds = 0, []

    for count, deletes in cfg.rounds:
        if deletes > len(live):
            raise StreamError(f"a round deletes {deletes} of {len(live)} live vectors")

        deleted = []

        for _ in range(deletes):
            i = rng.below(len(live))
            last = live.pop()

            if i < len(live):
                deleted.append(live[i])
                live[i] = last
            else:
                deleted.append(last)

        added = inserts[at : at + count]
        live.extend(added.tolist())
        rounds.append(Round(added, np.asarray(deleted, dtype=np.int64)))
        at += count

    return Stream(np.arange(cfg.n, dtype=np.int64), rounds)


def kmeans(rows: np.ndarray, k: int, rounds: int, rng: Rng, metric: str) -> np.ndarray:
    """`k` centers of `rows` from `rounds` rounds of Lloyd's k-means, started at `k` rows drawn with `rng`. For cosine
    the rows and the centers are unit vectors, and each point goes to the center of the largest dot product."""
    x = np.asarray(rows, dtype=np.float32)

    if metric == "cosine":
        x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)

    centers = x[draw(rng, np.arange(len(x)), k)].copy()
    sq = np.einsum("ij,ij->i", x, x)

    for _ in range(rounds):
        dots = x @ centers.T
        near = np.argmax(dots, axis=1) if metric != "l2" else \
            np.argmin(sq[:, None] - 2 * dots + np.einsum("ij,ij->i", centers, centers)[None, :], axis=1)
        # Each center becomes the mean of its rows, summed in one pass over the rows sorted by center. A center no row
        # picks keeps where it was.
        counts = np.bincount(near, minlength=k)
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        used = counts > 0
        sums = np.add.reduceat(x[np.argsort(near, kind="stable")], starts[used], axis=0)
        centers[used] = sums / counts[used][:, None]

        if metric == "cosine":
            centers /= np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12)

    return centers


def drift_inserts(cfg: WritesConfig, dataset: data.Dataset, log=None) -> np.ndarray:
    """The inserts of a drift stream, the drift pattern of the Rust stream maker. k-means finds `topics` clusters in
    `topic_sample` rows drawn among the candidates, the rows from `drift_from` on, `hot_topics` of them drawn at random
    are the topics, and the candidates nearest any topic's center, as many as the stream inserts, are the inserts, in a
    random order. A center ranks rows as k-means does, by their unit vectors for cosine."""
    end = dataset.count if cfg.drift_rows is None else min(dataset.count, cfg.drift_from + cfg.drift_rows)
    candidates = np.arange(cfg.drift_from, end, dtype=np.int64)

    if len(candidates) < max(cfg.total_inserts, cfg.topics):
        raise StreamError(f"a drift stream of {cfg.total_inserts} inserts needs more than the {len(candidates)} rows "
                          f"from {cfg.drift_from}")

    rng = Rng(cfg.seed)
    sample = np.sort(draw(rng, candidates, min(cfg.topic_sample, len(candidates))))
    centers = kmeans(dataset.base[sample], cfg.topics, cfg.kmeans_rounds, rng, dataset.metric)
    topics = centers[np.sort(draw(rng, np.arange(cfg.topics), cfg.hot_topics))].astype(np.float64)
    best_d = np.zeros(0)
    best_r = np.zeros(0, dtype=np.int64)

    for start in range(0, len(candidates), DRIFT_CHUNK):
        rows = candidates[start : start + DRIFT_CHUNK]
        x = np.asarray(dataset.base[rows[0] : rows[-1] + 1], dtype=np.float64)

        if dataset.metric == "cosine":
            x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)

        d = (np.einsum("ij,ij->i", x, x)[:, None] - 2 * (x @ topics.T)
             + np.einsum("ij,ij->i", topics, topics)[None, :]).min(axis=1)
        # The nearest so far hold lower rows than the chunk, so a stable sort keeps ties to the lower row.
        all_d, all_r = np.concatenate([best_d, d]), np.concatenate([best_r, rows])
        keep = np.argsort(all_d, kind="stable")[: cfg.total_inserts]
        best_d, best_r = all_d[keep], all_r[keep]

        if log is not None and start // DRIFT_CHUNK % 64 == 0:
            log(f"drift: {start + len(rows)} of {len(candidates)} candidate rows")

    order = draw(rng, np.arange(len(best_r)), len(best_r))
    return best_r[order]


def same_stream(a: Stream, b: Stream) -> bool:
    if not np.array_equal(a.base, b.base) or len(a.rounds) != len(b.rounds):
        return False

    return all(np.array_equal(x.inserts, y.inserts) and np.array_equal(x.deletes, y.deletes)
               and np.array_equal(x.upserts, y.upserts) for x, y in zip(a.rounds, b.rounds))


def stream_record(cfg: WritesConfig) -> Path:
    """The record beside a stream file, of what made it and its hash."""
    return cfg.stream.with_name(f"{cfg.stream.name}.json")


def ensure_stream(cfg: WritesConfig, dataset: data.Dataset | None = None) -> Stream:
    """The config's stream file, made when it is missing, with a record beside it of what made it. A file that another
    recipe made raises StreamError, since states, truths and results all name the stream by its hash. A uniform stream
    without a record is made again and compared, which is cheap, and a drift one is refused."""
    record = read_json(stream_record(cfg))

    if cfg.stream.exists():
        found = read_stream(cfg.stream)

        if record is not None:
            made = record.get("recipe") == schema.clean(cfg.recipe)

            if not made or record.get("sha256") != schema.file_sha256(cfg.stream):
                raise StreamError(f"{cfg.stream} holds another stream than the suite's [writes] makes")

            return found

        if cfg.pattern != "uniform" or not same_stream(found, make_stream(cfg, dataset)):
            raise StreamError(f"{cfg.stream} has no record of what made it, or holds another stream than the suite's "
                              f"[writes] makes")
    else:
        found = make_stream(cfg, dataset)
        write_stream(cfg.stream, found)

    write_json(stream_record(cfg), {"recipe": cfg.recipe, "sha256": schema.file_sha256(cfg.stream),
                                    "made": schema.utc_now()})
    return found


def live_mask(stream: Stream, rounds: int, deletes: bool = True, bound: int | None = None) -> np.ndarray:
    """Whether each row up to `bound`, the stream's last row by default, is live after its first `rounds` rounds, with
    their deletes when `deletes` is set. A delete of a vector that isn't live raises StreamError."""
    mask = np.zeros(stream.row_bound if bound is None else bound, dtype=bool)
    mask[stream.base] = True

    for r, rnd in enumerate(stream.rounds[:rounds], 1):
        if deletes and len(rnd.deletes):
            if not mask[rnd.deletes].all():
                raise StreamError(f"round {r} deletes an id that isn't live")

            mask[rnd.deletes] = False

        mask[rnd.inserts] = True

    return mask


def state_mask(cfg: WritesConfig, stream: Stream, name: str) -> np.ndarray:
    spec = cfg.spec(name)
    return live_mask(stream, spec.round, spec.with_deletes)


@dataclass
class Call:
    """One write call of a session: "insert" or "delete", of the vectors with the ids `ids` and the rows `rows`, in
    stream round `round`."""

    op: str
    ids: np.ndarray
    rows: np.ndarray
    round: int


def stream_calls(stream: Stream, pass_name: str, from_round: int, to_round: int, batch: int) -> list[list[Call]]:
    """The calls of a stream session, one list per round from `from_round` + 1 to `to_round`. Within a round the
    calls alternate a batch of inserts and a batch of deletes, and the inserts-only pass leaves the deletes out."""
    if pass_name not in PASSES:
        raise StreamError(f"unknown pass {pass_name}, expected one of {PASSES}")

    if not 0 <= from_round < to_round <= len(stream.rounds):
        raise StreamError(f"rounds {from_round} to {to_round} aren't in a stream of {len(stream.rounds)} rounds")

    out = []

    for r in range(from_round + 1, to_round + 1):
        rnd = stream.rounds[r - 1]
        dels = rnd.deletes if pass_name != "inserts" else rnd.deletes[:0]
        calls = []

        for at in range(0, max(len(rnd.inserts), len(dels)), batch):
            if at < len(rnd.inserts):
                rows = rnd.inserts[at : at + batch]
                calls.append(Call("insert", stream.ids(rows), rows, r))

            if at < len(dels):
                rows = dels[at : at + batch]
                calls.append(Call("delete", stream.ids(rows), rows, r))

        out.append(calls)

    return out


@dataclass
class LatencyPlan:
    """The rows a latency run inserts, its warm-up and timed ones first, then its durable ones, and the base ids it
    deletes, its warm-up ones first."""

    inserts: np.ndarray
    deletes: np.ndarray
    warmup: int
    calls: int
    durable: int


def latency_plan(stream: Stream, rounds: int, deletes: bool, reserve_from: int, warmup: int, calls: int,
                 durable: int, seed: int) -> LatencyPlan:
    """The plan of a latency run on a clone of the state after `rounds` rounds, with their deletes when `deletes` is
    set. It inserts reserved rows from `reserve_from`, and deletes base ids the state holds, drawn with `seed`."""
    live = np.flatnonzero(live_mask(stream, rounds, deletes)[: stream.base_rows])
    inserts = np.arange(reserve_from, reserve_from + warmup + calls + durable, dtype=np.int64)
    return LatencyPlan(inserts, draw(Rng(seed), live, warmup + calls), warmup, calls, durable)


def probe_rows(cfg: WritesConfig, stream: Stream, name: str, count: int | None = None) -> np.ndarray:
    """The rows a findability check searches as queries at state `name`: `count` inserted rows the state holds, then as
    many of its base rows, each drawn at random. A state with fewer of either takes them all."""
    count = cfg.probes if count is None else count
    mask = state_mask(cfg, stream, name)
    rng = Rng(cfg.probe_seed)
    inserted = np.flatnonzero(mask[cfg.n :]) + cfg.n
    base = np.flatnonzero(mask[: cfg.n])
    picked = [draw(rng, inserted, min(count, len(inserted))), draw(rng, base, min(count, len(base)))]
    return np.concatenate(picked).astype(np.int64)


def vectors(dataset: data.Dataset, rows: np.ndarray) -> np.ndarray:
    """The rows `rows` of the base as one contiguous float32 array."""
    rows = np.asarray(rows, dtype=np.int64)
    return np.ascontiguousarray(dataset.base[rows], dtype=np.float32).reshape(len(rows), dataset.dim)


def exact_truth(dataset: data.Dataset, mask: np.ndarray, k: int = TRUTH_K,
                queries: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Each query's `k` nearest of the rows `mask` marks live, nearest first and ties to the lower row, with their
    distances, from an exact float64 scan in numpy. It suits the small suites. The big ones use the gt binary."""
    q = np.asarray(dataset.queries if queries is None else queries, dtype=np.float64)
    nq, metric = len(q), dataset.metric
    best_d = np.full((nq, 0), np.inf)
    best_i = np.zeros((nq, 0), dtype=np.int64)
    qn = np.linalg.norm(q, axis=1)

    for start in range(0, len(mask), TRUTH_CHUNK):
        ids = np.flatnonzero(mask[start : start + TRUTH_CHUNK]) + start

        if not len(ids):
            continue

        x = np.asarray(dataset.base[ids], dtype=np.float64)
        dots = q @ x.T

        if metric == "l2":
            d = np.einsum("ij,ij->i", x, x)[None, :] - 2 * dots + (qn ** 2)[:, None]
        elif metric == "ip":
            d = -dots
        else:
            norms = np.linalg.norm(x, axis=1)[None, :] * qn[:, None]
            d = -dots / np.where(norms > 0, norms, 1.0)

        # The best so far hold lower rows than the chunk, and both are in row order where distances tie, so a stable
        # sort by distance keeps ties to the lower row.
        all_d = np.concatenate([best_d, d], axis=1)
        all_i = np.concatenate([best_i, np.broadcast_to(ids, d.shape)], axis=1)
        order = np.argsort(all_d, axis=1, kind="stable")[:, :k]
        best_d = np.take_along_axis(all_d, order, axis=1)
        best_i = np.take_along_axis(all_i, order, axis=1)

    if best_i.shape[1] < k:
        raise StreamError(f"a state holds {best_i.shape[1]} vectors, fewer than the truth's {k}")

    return best_i.astype(np.int32), best_d.astype(np.float32)


def gt_binary() -> Path:
    return Path(os.environ.get("COLDIVF_BIN") or REPO / "target" / "mine" / "release") / "gt"


def gt_states(cfg: WritesConfig, names: list[str]) -> str:
    """The --states argument of the gt binary, eg. "s1=1,s1i=1i": each state's rounds, with "i" when it skips the
    deletes."""
    specs = [cfg.spec(n) for n in names]
    return ",".join(f"{s.name}={s.round}{'' if s.with_deletes or s.round == 0 else 'i'}" for s in specs)


def truth_states(cfg: WritesConfig) -> list[str]:
    """The states with a truth of their own, S0 and every plain state of both ladders."""
    return ["s0"] + [s.name for s in cfg.states.values() if s.name != "s0" and cfg.truth_state(s.name) == s.name]


def compute_truths(cfg: WritesConfig, dataset: data.Dataset, engine: str = "auto", threads: int = 8,
                   names: list[str] | None = None, log=print) -> list[Path]:
    """Writes each state's truth beside the stream, as <stem>-<state>.gt.ibin with its .gt.dist.fbin. The numpy
    engine scans in float64, and the rust one runs gt --stream, which scans every state in one pass with the
    head-to-head's distances."""
    stream = read_stream(cfg.stream)
    stream.check()
    names = names or truth_states(cfg)
    engine = engine if engine != "auto" else ("numpy" if stream.id_bound <= NUMPY_TRUTH_ROWS else "rust")
    paths = [cfg.truth_path(n) for n in names]

    if engine == "rust":
        prefix = cfg.stream.with_name(cfg.stem)
        cmd = [str(gt_binary()), "--dataset", dataset.name, "--stream", str(cfg.stream), "--states",
               gt_states(cfg, names), "--out-prefix", str(prefix), "--k", str(TRUTH_K), "--threads", str(threads)]
        log(" ".join(cmd))
        subprocess.run(cmd, check=True, env=os.environ | {"COLDIVF_DATA": str(dataset.root)})
        return paths

    for name, path in zip(names, paths):
        ids, dists = exact_truth(dataset, state_mask(cfg, stream, name))
        data.write_ibin(path, ids)
        data.write_fbin(data.dist_path(path), dists)
        log(f"truth of {name}: {len(ids)} queries over {cfg.live_count(name)} vectors, to {path}")

    return paths


def write_copy(cfg: WritesConfig, dataset: data.Dataset, name: str = "s10") -> Path:
    """Writes the live rows of a state as a dataset of its own beside the stream, for the fresh builds: the rows in id
    order as <stem>-<state>-live.base.fbin, the queries, the row of each position as .rows.ibin, the state's truth by
    position as .gt.ibin and .gt.dist.fbin, and a .copy.json that names the state and the stream."""
    stream = read_stream(cfg.stream)
    rows = np.flatnonzero(state_mask(cfg, stream, name)).astype(np.int64)
    base = cfg.copy_base(name)
    stem = base.name[: -len(".base.fbin")]
    root = base.parent
    tmp = base.with_name(f"{base.name}.{os.getpid()}.tmp")

    with open(tmp, "wb") as f:
        f.write(np.array([len(rows), dataset.dim], dtype="<i4").tobytes())

        for start in range(0, len(rows), CHUNK):
            f.write(np.ascontiguousarray(dataset.base[rows[start : start + CHUNK]], dtype="<f4").tobytes())

        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp, base)
    queries = root / f"{stem}.query.fbin"
    queries.unlink(missing_ok=True)
    clone_tree(dataset.query_path, queries)
    data.write_ibin(root / f"{stem}.rows.ibin", rows.reshape(-1, 1).astype(np.int32))
    truth = dataset.truth(cfg.id_bound, cfg.truth_path(name))
    ids = np.asarray(truth.ids, dtype=np.int64)
    pos = np.searchsorted(rows, ids)

    if not np.array_equal(rows[np.minimum(pos, len(rows) - 1)], ids):
        raise StreamError(f"the truth of {name} names a vector the state doesn't hold")

    data.write_ibin(root / f"{stem}.gt.ibin", pos.astype(np.int32))

    if truth.dists is not None:
        data.write_fbin(root / f"{stem}.gt.dist.fbin", np.asarray(truth.dists))

    write_json(root / f"{stem}.copy.json", {"state": cfg.truth_state(name), "stream": str(cfg.stream),
                                            "stream_sha256": schema.file_sha256(cfg.stream), "rows": len(rows),
                                            "dataset": dataset.name, "made": schema.utc_now()})
    return base


def copy_record(base: Path | str) -> dict | None:
    """The .copy.json beside a copy's base file, or None for a base file that isn't a copy of a state."""
    base = Path(base)
    name = base.name[: -len(".base.fbin")] if base.name.endswith(".base.fbin") else base.stem
    return read_json(base.with_name(f"{name}.copy.json"))


def copy_rows(base: Path | str) -> np.ndarray:
    """The row of each position of a copy, from the .rows.ibin beside its base file."""
    base = Path(base)
    name = base.name[: -len(".base.fbin")]
    return np.asarray(data.ibin(base.with_name(f"{name}.rows.ibin"))[:, 0], dtype=np.int64)


class StateRecall(data.Recall):
    """Scores results against a state's truth, whose ids and live mask are rows. A returned id the state doesn't hold,
    deleted or not inserted yet, is a miss before the tie check, since a deleted row near the query would otherwise tie.
    Ids count up to the stream's last one at every state, so an insert is never dropped as past the live count. With
    `rows`, each returned id becomes its row first, eg. a position in a copy of the state's live rows, or an id of a
    drift stream."""

    def __init__(self, dataset: data.Dataset, truth: data.Truth, live: np.ndarray, base_rows: int,
                 rows: np.ndarray | None = None, rtol: float = data.TIE_RTOL):
        super().__init__(dataset, truth, rtol)
        self.live = np.asarray(live, dtype=bool)
        self.base_rows = base_rows
        self.rows = rows

    def clean(self, found: np.ndarray, k: int) -> list[int]:
        """The first `k` distinct ids of a row of results that name a row of the stream, as rows, in order."""
        out: list[int] = []
        bound = len(self.live)

        for i in np.asarray(found).ravel().tolist():
            if self.rows is not None:
                i = int(self.rows[i]) if 0 <= i < len(self.rows) else -1

            if 0 <= i < bound and i not in out:
                out.append(int(i))

            if len(out) == k:
                break

        return out

    def hits(self, q: int, found: np.ndarray, k: int) -> tuple[int, int]:
        true = set(int(i) for i in self.truth.ids[q, :k])
        ids = self.clean(found, k)
        plain = sum(1 for i in ids if i in true)
        rest = [i for i in ids if i not in true and self.live[i]]

        if not rest:
            return plain, plain

        kth = self.kth(q, k)
        d = data.distances(self.dataset.metric, self.dataset.queries[q], self.dataset.base[np.array(rest)])
        tied = int(np.count_nonzero(d <= kth + self._tol(kth)))
        return min(plain + tied, k), plain

    def dead(self, found: np.ndarray, k: int = 10) -> int:
        """How many of the first `k` results name an id the state doesn't hold."""
        return sum(1 for i in self.clean(found, k) if not self.live[i])

    def inserted(self, q: int, found: np.ndarray, k: int = 10) -> tuple[int, int]:
        """How many of query `q`'s true `k` nearest that are inserts the results found, and how many there are."""
        true = [int(i) for i in self.truth.ids[q, :k] if int(i) >= self.base_rows]
        ids = set(self.clean(found, k))
        return sum(1 for i in true if i in ids), len(true)

    def summary(self, qids: np.ndarray, found: np.ndarray) -> dict:
        out = super().summary(qids, found)
        found = np.asarray(found)
        dead = [self.dead(found[i]) for i in range(len(qids))]
        hit = total = 0

        for i, q in enumerate(np.asarray(qids).tolist()):
            a, b = self.inserted(int(q), found[i])
            hit, total = hit + a, total + b

        out["dead_returned"] = float(np.mean(dead)) if dead else None
        out["recall10_inserted"] = hit / total if total else None
        return out

    def trial(self, q: int, found: np.ndarray, k: int = 10) -> dict:
        """The recall fields of one cold trial's results for query `q`."""
        ties, plain = self.hits(q, found, k)
        a, b = self.inserted(q, found, k)
        return {"recall10": ties / k, "recall10_ids_only": plain / k, "dead_returned": float(self.dead(found, k)),
                "recall10_inserted": a / b if b else None}


def findability(dataset: data.Dataset, probes: np.ndarray, found: np.ndarray, base_rows: int,
                rows: np.ndarray | None = None, rtol: float = data.TIE_RTOL) -> dict:
    """The share of probe rows whose own id came back first when searched as a query, or behind only rows at the same
    distance, eg. an exact duplicate, for the inserted probes and the base ones apart. With `rows`, each returned id
    becomes its row first, as in StateRecall."""
    hits = {"inserted": [0, 0], "base": [0, 0]}
    missed: list[int] = []

    for row, ids in zip(np.asarray(probes).tolist(), np.asarray(found)):
        kind = "inserted" if row >= base_rows else "base"
        hits[kind][1] += 1
        ids = [int(i) for i in np.asarray(ids).ravel().tolist() if 0 <= i < (len(rows) if rows is not None else
                                                                             dataset.count)]
        ids = [int(rows[i]) for i in ids] if rows is not None else ids

        if row not in ids:
            missed.append(row)
            continue

        at = ids.index(row)

        if at:
            d = data.distances(dataset.metric, dataset.base[row], dataset.base[np.array(ids[: at + 1])])

            if np.any(np.abs(d[:at] - d[at]) > rtol * max(1.0, abs(float(d[at])))):
                missed.append(row)
                continue

        hits[kind][0] += 1

    return {"inserted": hits["inserted"][0] / hits["inserted"][1] if hits["inserted"][1] else None,
            "base": hits["base"][0] / hits["base"][1] if hits["base"][1] else None,
            "probes": {k: v[1] for k, v in hits.items()}, "missed": missed[:20]}


def write_json(path: Path | str, value: dict) -> None:
    """Writes `value` as JSON through a temporary file, synced and renamed, so a reader sees all of it or none."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")

    with open(tmp, "w") as f:
        f.write(json.dumps(schema.clean(value), indent=1))
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp, path)


def read_json(path: Path | str) -> dict | None:
    """The JSON object at `path`, or None when it is missing or cut short."""
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None

    return value if isinstance(value, dict) else None


def record_path(folder: Path | str) -> Path:
    """Where a state's record lives, beside its folder, so it is never one of the index's files."""
    folder = Path(folder)
    return folder.with_name(f"{folder.name}.state.json")


def latency_path(folder: Path | str) -> Path:
    folder = Path(folder)
    return folder.with_name(f"{folder.name}.latency.json")


def clone_tree(src: Path | str, dst: Path | str) -> None:
    """Copies the folder or file `src` to `dst`, which mustn't exist, as an APFS clone of every file. The copy shares
    its disk blocks, and no write to one reaches the other. Where clones don't work it makes a plain copy."""
    src, dst = Path(src), Path(dst)

    if dst.exists():
        raise FileExistsError(f"{dst} is there already")

    dst.parent.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.run(["cp", "-c", "-R", str(src), str(dst)], check=True, capture_output=True)
        return
    except (OSError, subprocess.CalledProcessError):
        remove_tree(dst)

    if src.is_dir():
        shutil.copytree(src, dst, copy_function=shutil.copy2)
    else:
        shutil.copy2(src, dst)


def copy_file(src: Path | str, dst: Path | str) -> None:
    """Copies one file into disk blocks of its own, each run of data in writes of up to COPY_CHUNK bytes at its own
    offset, so a hole stays a hole, then syncs it, so its blocks are placed before anything else is written. Files
    with holes are common here, eg. qdrant's log segments and vector chunks, which it makes at a set size and mostly
    empty. macOS's cp copies such a file a few KiB at a time, so another process's writes in between leave the copy in
    hundreds of pieces, and shutil's copy fills the holes in. The copy keeps the file's mode and times."""
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        size, at = os.fstat(fin.fileno()).st_size, 0

        while at < size:
            try:
                start = os.lseek(fin.fileno(), at, os.SEEK_DATA)
            except OSError:
                break

            end = min(size, os.lseek(fin.fileno(), start, os.SEEK_HOLE))

            while start < end:
                block = os.pread(fin.fileno(), min(COPY_CHUNK, end - start), start)

                if not block:
                    break

                os.pwrite(fout.fileno(), block, start)
                start += len(block)

            at = end

        os.ftruncate(fout.fileno(), size)
        os.fsync(fout.fileno())

    shutil.copystat(src, dst)


def copy_tree(src: Path | str, dst: Path | str) -> None:
    """Copies the folder or file `src` to `dst`, which mustn't exist, byte for byte into disk blocks of its own, where
    clone_tree shares them. A write to the copy then lands on its own block in place. Each file goes whole, holes and
    all, through copy_file, and keeps its change time, as a clone does, since a sealed store finds a changed file by its
    size and change time."""
    src, dst = Path(src), Path(dst)

    if dst.exists():
        raise FileExistsError(f"{dst} is there already")

    dst.parent.mkdir(parents=True, exist_ok=True)

    if not src.is_dir():
        copy_file(src, dst)
        return

    for root, dirs, files in os.walk(src):
        here = dst / Path(root).relative_to(src)
        here.mkdir(exist_ok=True)

        for name in dirs + files:
            if (Path(root) / name).is_symlink():
                os.symlink(os.readlink(Path(root) / name), here / name)

        for name in files:
            if not (Path(root) / name).is_symlink():
                copy_file(Path(root) / name, here / name)

    # Folders take their times last, since each file made in one changes them.
    for root, dirs, _ in sorted(os.walk(src), key=lambda x: -len(Path(x[0]).parts)):
        for name in dirs:
            if not (Path(root) / name).is_symlink():
                shutil.copystat(Path(root) / name, dst / Path(root, name).relative_to(src))

    shutil.copystat(src, dst)


# The fcntl call of macOS that maps a file's bytes to where they sit on the disk.
F_LOG2PHYS_EXT = 65


def file_extents(path: Path | str) -> int | None:
    """How many runs of contiguous disk blocks hold the file's bytes, from F_LOG2PHYS_EXT, or None where the call
    isn't there. A hole takes none."""
    if sys.platform != "darwin":
        return None

    fd = os.open(path, os.O_RDONLY)

    try:
        size, at, runs, end = os.fstat(fd).st_size, 0, 0, None

        while at < size:
            asked = struct.pack("=Iqq", 0, size - at, at)
            _, length, disk = struct.unpack("=Iqq", fcntl.fcntl(fd, F_LOG2PHYS_EXT, asked))

            if length <= 0:
                break

            if disk >= 0 and disk != end:
                runs += 1

            end = disk + length if disk >= 0 else None
            at += length

        return runs
    except OSError:
        return None
    finally:
        os.close(fd)


def extents(path: Path | str) -> dict | None:
    """The runs of contiguous disk blocks of every file under `path`, in all and for the file with the most. A file
    written in one go takes a few, and one whose blocks writes to a clone moved can take thousands, which a cold read
    pays for. None where the count isn't there."""
    path = Path(path)
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file() and not p.is_symlink())
    counts = {}

    for p in files:
        n = file_extents(p)

        if n is None:
            return None

        counts[str(p.relative_to(path)) if p != path else p.name] = n

    most = max(counts, key=counts.get) if counts else None
    return {"total": sum(counts.values()), "files": len(counts), "most": most, "most_extents": counts.get(most, 0)}


def remove_tree(path: Path | str) -> None:
    path = Path(path)

    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists() or path.is_symlink():
        path.unlink(missing_ok=True)


def folder_bytes(path: Path | str) -> int:
    """The bytes of every file under `path`, or of the file itself."""
    path = Path(path)

    if path.is_file():
        return path.stat().st_size

    total = 0

    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.lstat(os.path.join(root, f)).st_size
            except OSError:
                pass

    return total


def fullfsync(path: Path | str) -> None:
    """Makes a file survive a power loss. On macOS that takes F_FULLFSYNC, since a plain fsync stops at the drive."""
    fd = os.open(path, os.O_RDONLY)

    try:
        if hasattr(fcntl, "F_FULLFSYNC"):
            fcntl.fcntl(fd, fcntl.F_FULLFSYNC)
        else:
            os.fsync(fd)
    finally:
        os.close(fd)


def show(cfg: WritesConfig) -> None:
    print(f"stream {cfg.stream}, base {cfg.n}, rounds {cfg.rounds}, ids up to {cfg.id_bound}")

    for spec in sorted(cfg.states.values(), key=lambda s: (s.pass_name, s.round)):
        truth = cfg.truth_path(spec.name)
        print(f"  {spec.name}: pass {spec.pass_name or 'build'}, round {spec.round}, {spec.inserts} inserts, "
              f"{spec.deletes} deletes, {cfg.live_count(spec.name)} live, truth {cfg.truth_state(spec.name)} "
              f"{'there' if truth.exists() else 'missing'}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="The write benchmark's stream, truth and copies.")
    p.add_argument("command", choices=("stream", "truth", "copy", "prepare", "show"))
    p.add_argument("suite")
    p.add_argument("--engine", default="auto", choices=("auto", "numpy", "rust"))
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--state", default="", help="the state to copy, by default the joint ladder's last")
    args = p.parse_args(argv)
    cfg = load_config(args.suite)
    dataset = data.Dataset.named(cfg.dataset)

    if args.command == "show":
        show(cfg)
        return 0

    if args.command in ("stream", "prepare"):
        stream = ensure_stream(cfg, dataset)
        print(f"stream {cfg.stream}: {len(stream.rounds)} rounds, sha256 {schema.file_sha256(cfg.stream)}")

    if args.command in ("truth", "prepare"):
        compute_truths(cfg, dataset, args.engine, args.threads)

    if args.command in ("copy", "prepare"):
        final = cfg.final("joint")
        name = args.state or (final.name if final else "")

        if not name:
            raise StreamError("the suite has no joint ladder, so name the state to copy with --state")

        print(f"copy of {name}: {write_copy(cfg, dataset, name)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
