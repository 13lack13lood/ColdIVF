"""The "coldivf-bench/2" result lines, their validator and writer, run keys for resuming, and provenance.

Every line is one JSON object. It holds the common fields, then the fields of its kind, which is build, warm, cold,
write, maintain, ref or batch. Every field of both sets is present in every line, with null for what wasn't measured.
Anything else a system measures goes under "extra". A line's run key names the measurement it is, so a resumed run
skips what is done.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable

import fcntl

SCHEMA = "coldivf-bench/2"
KINDS = ("build", "warm", "cold", "write", "maintain", "ref", "batch")
STATUSES = ("ok", "rejected", "invalid", "failed", "unsupported")
# Statuses that end a measurement. A resumed run repeats the others, up to its limit of attempts.
FINAL = ("ok", "unsupported")

REPO = Path(__file__).resolve().parents[2]


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v: Any) -> bool:
    return (_is_int(v) or isinstance(v, float)) and math.isfinite(v)


TYPES: dict[str, Callable[[Any], bool]] = {
    "str": lambda v: isinstance(v, str),
    "int": _is_int,
    "num": _is_num,
    "bool": lambda v: isinstance(v, bool),
    "dict": lambda v: isinstance(v, dict),
    "nums": lambda v: isinstance(v, list) and all(_is_num(x) for x in v),
    "ints": lambda v: isinstance(v, list) and all(_is_int(x) for x in v),
}

# Field name to type. A trailing "?" means the value may be null.
COMMON = {
    "schema": "str", "run_id": "str", "run_key": "str", "attempt": "int", "status": "str", "error": "str?",
    "ts": "str", "batch_id": "str", "phase": "str", "kind": "str", "system": "str", "system_version": "str?",
    "venv": "str?", "binding": "str?", "simd_level": "str?", "variant": "str", "config_id": "str?",
    "build_params": "dict?", "search_params": "dict?", "dataset": "str?", "metric": "str?", "n": "int?",
    "dim": "int?", "base_dtype": "str?", "split": "str?", "gt": "dict?", "machine": "dict", "provenance": "dict",
    "index_identity": "dict?", "gate": "dict", "load_before": "nums?", "load_after": "nums?", "agents_cpu": "dict?",
    "open_us_batch": "num?", "ref_drift_pct": "num?", "stream": "dict?", "state": "str?", "extra": "dict",
}

FIELDS = {
    "build": {
        "seconds": "num?", "cpu_seconds": "num?", "threads": "int?", "seed": "int?", "cluster_seconds": "num?",
        "encode_seconds": "num?", "peak_footprint": "int?", "bytes_read": "int?", "bytes_written": "int?",
        "peak_disk_bytes": "int?", "temporary_files": "int?", "index_bytes": "int?", "index_alloc_bytes": "int?",
        "index_files": "int?", "stored_vector_bytes": "int?", "exact_f32_stored": "bool?",
    },
    "warm": {
        "mode": "str", "threads": "int", "batch_api": "bool?", "processes": "int?", "queries": "int?",
        "passes": "int?", "k": "int?", "p50_us": "num?", "p90_us": "num?", "p99_us": "num?", "mean_us": "num?",
        "qps": "num?", "cpu_ms_per_query": "num?", "recall10": "num?", "recall10_ids_only": "num?",
        "recall1": "num?", "recall100": "num?", "recall10_p05": "num?", "recall10_p10": "num?",
        "share_ge_0_9": "num?", "footprint": "int?", "resident_file_bytes": "int?", "memory_total": "int?",
        "dist_evals": "num?", "bytes_scanned": "num?", "swapins_delta": "int?", "compressions_delta": "int?",
        "overhead_us": "num?", "ids_file": "str?", "dead_returned": "num?", "recall10_inserted": "num?",
    },
    "cold": {
        "regime": "str", "cpu_state": "str?", "delay_ms": "num?", "threads": "int?", "k": "int?", "trial": "int",
        "query_id": "int?", "predecessor": "str?", "resident_pages_at_go": "int?", "open_us": "num?",
        "search_us": "num?", "total_us": "num?", "resident_rerun_us": "num?", "next_us": "nums?", "stages": "dict?",
        "disk_bytes": "int?", "bytes_requested": "int?", "pageins": "int?", "files_in_index": "int?",
        "files_touched": "int?", "footprint_before": "int?", "footprint_delta": "int?",
        "resident_file_bytes_after": "int?", "lifetime_max_footprint": "int?", "cpu_ms": "num?",
        "cpu_qos_ms": "dict?", "p_share": "num?", "ghz": "num?", "energy_nj": "int?", "server_start_ms": "num?",
        "ids": "ints?", "recall10": "num?", "recall10_ids_only": "num?", "swapins_delta": "int?",
        "compressions_delta": "int?", "qos": "str?", "dead_returned": "num?", "recall10_inserted": "num?",
    },
    "write": {
        "mode": "str", "op": "str", "state_from": "str?", "state_to": "str?", "batch": "int?", "calls": "int?",
        "rows": "int?", "threads": "int?", "durability": "str?", "open_us": "num?", "open_footprint": "int?",
        "first_us": "num?", "p50_us": "num?", "p90_us": "num?", "p99_us": "num?", "max_us": "num?", "mean_us": "num?",
        "rows_per_s": "num?", "cpu_ms": "num?", "cpu_qos_ms": "dict?", "logical_bytes_written": "int?",
        "disk_bytes_written": "int?", "index_bytes_before": "int?", "index_bytes_after": "int?",
        "footprint_before": "int?", "footprint_after": "int?", "lifetime_max_footprint": "int?",
        "swapins_delta": "int?", "compressions_delta": "int?",
    },
    "maintain": {
        "action": "str", "state_from": "str?", "state_to": "str?", "threads": "int?", "params": "dict?",
        "seconds": "num?", "cpu_seconds": "num?", "open_us": "num?", "footprint_before": "int?",
        "peak_footprint": "int?", "bytes_read": "int?", "bytes_written": "int?", "peak_disk_bytes": "int?",
        "index_bytes_before": "int?", "index_bytes_after": "int?", "rows_live": "int?", "durability": "str?",
    },
    "ref": {
        "position": "str", "wave_us_p50": "num?", "coldivf_c0_p50_us": "num?", "wave_trials": "int?",
        "coldivf_trials": "int?", "coldivf_rejected": "int?",
    },
    "batch": {
        "started": "str", "ended": "str", "suite": "str", "suite_sha256": "str", "counts": "dict", "by_system": "dict",
        "resumed": "int", "refs": "dict", "drift_start_pct": "num?", "drift_end_pct": "num?", "flagged": "bool",
        "provenance_changed": "bool", "agents_cpu_after": "dict?", "disk_free_gb": "dict",
    },
}

# What an "ok" line of each kind must have measured.
OK_NEEDS: dict[str, tuple[tuple[str, ...], ...]] = {
    "build": (("seconds",), ("index_bytes",)),
    "warm": (("p50_us", "qps"), ("queries",)),
    "cold": (("total_us",),),
    "write": (("p50_us", "rows_per_s"),),
    "maintain": (("seconds",),),
    "ref": (("wave_us_p50", "coldivf_c0_p50_us"),),
    "batch": (),
}


def blank(kind: str) -> dict:
    """A line of `kind` with every field null, except the few that have a natural empty value."""
    line = {name: None for name in COMMON} | {name: None for name in FIELDS[kind]}
    line.update(schema=SCHEMA, kind=kind, attempt=1, status="ok", variant="", machine={}, provenance={}, gate={},
                extra={})
    return line


def validate(line: dict) -> list[str]:
    """Every problem with `line`, or an empty list when it is a valid line."""
    if not isinstance(line, dict):
        return ["a line must be a JSON object"]

    kind = line.get("kind")

    if kind not in FIELDS:
        return [f"unknown kind {kind!r}"]

    spec = COMMON | FIELDS[kind]
    errors = [f"missing field {name}" for name in spec if name not in line]
    errors += [f"unknown field {name}" for name in line if name not in spec]

    for name, want in spec.items():
        if name not in line:
            continue

        value, nullable = line[name], want.endswith("?")

        if value is None:
            if not nullable:
                errors.append(f"{name} is null")
        elif not TYPES[want.rstrip("?")](value):
            errors.append(f"{name} should be {want.rstrip('?')}, not {type(value).__name__} {value!r:.60}")

    if line.get("schema") != SCHEMA:
        errors.append(f"schema should be {SCHEMA}")

    if line.get("status") not in STATUSES:
        errors.append(f"unknown status {line.get('status')!r}")

    if line.get("status") == "ok":
        for options in OK_NEEDS[kind]:
            if all(line.get(name) is None for name in options):
                errors.append(f"an ok {kind} line needs {' or '.join(options)}")

    try:
        dumps(line)
    except ValueError as e:
        errors.append(str(e))

    return errors


def check(line: dict) -> dict:
    errors = validate(line)

    if errors:
        raise ValueError("; ".join(errors))

    return line


def clean(value: Any) -> Any:
    """`value` as plain JSON values. numpy numbers become Python numbers, tuples become lists, and numbers that aren't
    finite become null."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]

    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        return clean(value.tolist())

    if isinstance(value, float) and not math.isfinite(value):
        return None

    if isinstance(value, Path):
        return str(value)

    return value


def dumps(line: dict) -> str:
    return json.dumps(line, separators=(",", ":"), allow_nan=False, sort_keys=False)


def run_key(identity: dict) -> str:
    """A short stable hash of the fields that name one measurement."""
    text = json.dumps(clean(identity), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()[:20]


def new_run_id() -> str:
    return uuid.uuid4().hex


def utc_now() -> str:
    """The time in Coordinated Universal Time (UTC) to the millisecond, eg. "2026-10-08T05:12:33.123Z"."""
    t = time.time()
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t)) + f".{int(t % 1 * 1000):03d}Z"


def append(path: Path | str, line: dict) -> None:
    """Checks `line`, then appends it to the results file at `path` and fsyncs, so a sleep or a kill loses at most the
    line in flight. A lock keeps two writers from interleaving."""
    check(line)
    data = (dumps(line) + "\n").encode()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)

    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        view = memoryview(data)

        while view:
            view = view[os.write(fd, view):]

        os.fsync(fd)
    finally:
        os.close(fd)


def read(path: Path | str) -> list[dict]:
    """The lines of a results file. A line that doesn't parse, eg. one cut short by a crash, is skipped."""
    path = Path(path)

    if not path.exists():
        return []

    out = []

    with open(path, "rb") as f:
        for raw in f:
            try:
                line = json.loads(raw)
            except ValueError:
                continue

            if isinstance(line, dict):
                out.append(line)

    return out


class Progress:
    """What results files already hold, by run key. A key is done once it has a line with a final status, or once it
    has used up its attempts."""

    def __init__(self, paths: Iterable[Path | str], max_attempts: int = 3):
        self.max_attempts = max_attempts
        self.statuses: dict[str, list[str]] = {}

        for path in paths:
            for line in read(path):
                key = line.get("run_key")

                if isinstance(key, str):
                    self.statuses.setdefault(key, []).append(str(line.get("status")))

    def attempts(self, key: str) -> int:
        return len(self.statuses.get(key, []))

    def done(self, key: str) -> bool:
        seen = self.statuses.get(key, [])
        return any(s in FINAL for s in seen) or len(seen) >= self.max_attempts

    def add(self, key: str, status: str) -> None:
        self.statuses.setdefault(key, []).append(status)


def _run(cmd: list[str], cwd: Path | None = None, timeout: float = 60) -> bytes | None:
    try:
        out = subprocess.run(cmd, cwd=cwd, capture_output=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None

    return out.stdout if out.returncode == 0 else None


def _git(repo: Path, *args: str) -> bytes | None:
    # --no-optional-locks keeps git from rewriting the index, so it never takes the lock from someone else.
    return _run(["git", "-C", str(repo), "--no-optional-locks", *args])


def staged_tree_hash(repo: Path = REPO) -> str | None:
    """The id of the tree that `git write-tree` would write for the staged files, computed without writing anything.
    A commit of the staged changes has this tree, so a line can be matched to it later."""
    out = _git(repo, "ls-files", "-s", "-z")

    if out is None:
        return None

    root: dict = {}

    for entry in out.split(b"\0"):
        if not entry:
            continue

        meta, path = entry.split(b"\t", 1)
        mode, sha, stage = meta.split(b" ")

        if stage != b"0":
            return None

        node = root
        *dirs, name = path.split(b"/")

        for d in dirs:
            node = node.setdefault((d, True), {})

        node[(name, False)] = (mode, bytes.fromhex(sha.decode()))

    def tree(node: dict) -> bytes:
        body = b""

        # Git orders a tree's entries by name, with a folder's name read as if it ended in "/".
        for (name, is_dir), value in sorted(node.items(), key=lambda kv: kv[0][0] + (b"/" if kv[0][1] else b"")):
            if is_dir:
                body += b"40000 " + name + b"\0" + tree(value)
            else:
                mode, sha = value
                body += mode + b" " + name + b"\0" + sha

        return hashlib.sha1(b"tree %d\0" % len(body) + body).digest()

    return tree(root).hex()


def diff_hash(repo: Path = REPO) -> str | None:
    """A hash of everything not staged, i.e. the unstaged diff and every untracked file that isn't ignored."""
    diff = _git(repo, "diff", "--binary", "--no-ext-diff", "--no-color")
    others = _git(repo, "ls-files", "--others", "--exclude-standard", "-z")

    if diff is None or others is None:
        return None

    h = hashlib.sha256(diff)

    for name in sorted(p for p in others.split(b"\0") if p):
        h.update(b"\0untracked\0" + name + b"\0")
        path = repo / name.decode(errors="surrogateescape")

        if path.is_file() and not path.is_symlink():
            h.update(file_sha256(path).encode())

    return h.hexdigest()


def git_state(repo: Path = REPO) -> dict:
    head = _git(repo, "rev-parse", "HEAD")
    tree = staged_tree_hash(repo)
    diff = diff_hash(repo)
    empty = hashlib.sha256(b"").hexdigest()
    return {
        "repo": str(repo),
        "head": head.decode().strip() if head else None,
        "tree_hash": tree,
        "diff_hash": diff,
        "unstaged": diff is not None and diff != empty,
    }


def file_sha256(path: Path | str, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)

    return h.hexdigest()


@lru_cache(maxsize=None)
def pip_freeze_sha256(python: str = sys.executable) -> str | None:
    """A hash of `pip freeze --all` for an interpreter, or of its installed distributions when pip is missing."""
    out = _run([python, "-m", "pip", "freeze", "--all", "--disable-pip-version-check"], timeout=120)

    if out is None:
        code = (
            "import importlib.metadata as m\n"
            "print('\\n'.join(sorted(f\"{d.metadata['Name']}=={d.version}\" for d in m.distributions())))"
        )
        out = _run([python, "-c", code], timeout=120)

    return hashlib.sha256(out).hexdigest() if out is not None else None


def dist_sha256(name: str) -> str | None:
    """A hash of an installed distribution's RECORD file, which lists the hash of every file it installed."""
    try:
        import importlib.metadata as metadata

        record = metadata.distribution(name).read_text("RECORD")
    except Exception:
        return None

    return hashlib.sha256(record.encode()).hexdigest() if record else None


# Environment variables that change how the systems run. MallocLargeCache=0 turns off macOS malloc's cache of freed
# large blocks, which otherwise stays inside a process's footprint, so a line's footprints compare only with lines that
# share its setting.
ENV_KEYS = (
    "OMP_NUM_THREADS", "RAYON_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "LANCE_CPU_THREADS", "LANCE_IO_THREADS", "COLDIVF_DATA", "COLDIVF_BIN", "PYTHONHASHSEED",
    "MallocNanoZone", "MallocLargeCache", "RUSTFLAGS",
)


def env_record(extra: Iterable[str] = ()) -> dict:
    return {k: os.environ[k] for k in (*ENV_KEYS, *extra) if k in os.environ}


SYSCTLS = (
    "hw.model", "machdep.cpu.brand_string", "hw.cpufamily", "hw.nperflevels", "hw.physicalcpu", "hw.logicalcpu",
    "hw.memsize", "hw.pagesize", "hw.cachelinesize", "kern.osversion", "kern.osproductversion", "kern.version",
    "sysctl.proc_translated",
)
PERFLEVEL = ("name", "physicalcpu", "logicalcpu", "l1icachesize", "l1dcachesize", "l2cachesize", "cpusperl2")


def sysctl(names: Iterable[str]) -> dict[str, str]:
    """Values of sysctl names, read in one call. Names this machine doesn't have are left out."""
    names = list(names)

    try:
        out = subprocess.run(["sysctl", *names], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.TimeoutExpired):
        return {}

    values = {}

    for line in out.splitlines():
        name, sep, value = line.partition(": ")

        if sep and name in names:
            values[name] = value.strip()

    return values


def _number(text: str | None) -> int | str | None:
    if text is None:
        return None

    try:
        return int(text)
    except ValueError:
        return text


def _gpu_cores() -> int | None:
    out = _run(["ioreg", "-rc", "AGXAccelerator", "-d", "1"], timeout=30)

    for line in (out or b"").decode(errors="replace").splitlines():
        if '"gpu-core-count"' in line:
            return _number(line.split("=")[-1].strip())

    return None


def _ssd() -> dict | None:
    out = _run(["system_profiler", "SPNVMeDataType", "-json"], timeout=60)

    try:
        items = json.loads(out or b"{}").get("SPNVMeDataType", [])
        drive = items[0]["_items"][0]
    except (ValueError, KeyError, IndexError, TypeError):
        return None

    return {"model": drive.get("device_model") or drive.get("_name"), "capacity": drive.get("size"),
            "capacity_bytes": drive.get("size_in_bytes")}


def _power() -> dict:
    batt = (_run(["pmset", "-g", "batt"]) or b"").decode(errors="replace")
    conf = (_run(["pmset", "-g"]) or b"").decode(errors="replace")
    source = batt.split("'")[1] if batt.count("'") >= 2 else None
    low = None

    for line in conf.splitlines():
        words = line.split()

        if len(words) >= 2 and words[0] == "lowpowermode":
            low = words[1] != "0"
        elif len(words) >= 2 and words[0] == "powermode" and low is None:
            low = words[1] == "1"

    return {"source": source, "low_power": low}


def power() -> dict:
    """The power source, eg. "AC Power", and whether low-power mode is on."""
    return _power()


def _rust() -> dict:
    rustc = _run(["rustc", "-V"])
    config = REPO / ".cargo" / "config.toml"
    return {
        "rustc": rustc.decode().strip() if rustc else None,
        "cargo_config_sha256": file_sha256(config) if config.exists() else None,
        "rustflags": os.environ.get("RUSTFLAGS"),
    }


@lru_cache(maxsize=1)
def machine() -> dict:
    """The machine record of every line. It holds the chip, its core types and their caches, page and line sizes,
    translation status, graphics processor (GPU) cores, the solid-state drive (SSD), the operating system build, power
    state and the Rust toolchain. It leaves out the top-level cache sizes, which report the efficiency cores' values on
    Apple silicon."""
    levels = int(sysctl(["hw.nperflevels"]).get("hw.nperflevels", "0") or 0)
    names = list(SYSCTLS) + [f"hw.perflevel{i}.{k}" for i in range(levels) for k in PERFLEVEL]
    values = sysctl(names)
    get = lambda k: _number(values.get(k))
    return {
        "model": values.get("hw.model"),
        "brand": values.get("machdep.cpu.brand_string"),
        "cpufamily": get("hw.cpufamily"),
        "physicalcpu": get("hw.physicalcpu"),
        "logicalcpu": get("hw.logicalcpu"),
        "memsize": get("hw.memsize"),
        "perflevels": [{k: get(f"hw.perflevel{i}.{k}") for k in PERFLEVEL} for i in range(levels)],
        "pagesize": get("hw.pagesize"),
        "cachelinesize": get("hw.cachelinesize"),
        "translated": get("sysctl.proc_translated"),
        "gpu_cores": _gpu_cores(),
        "ssd": _ssd(),
        "os": {"product_version": values.get("kern.osproductversion"), "build": values.get("kern.osversion"),
               "kernel": values.get("kern.version"), "platform": platform.platform()},
        "power": _power(),
        "rust": _rust(),
    }


def provenance(python: str = sys.executable, wheels: dict[str, str | None] | None = None,
               git: dict | None = None, env: dict | None = None) -> dict:
    """The provenance record of a line. It holds the repo's staged tree hash and unstaged diff hash, a hash of the
    interpreter's installed packages, the hashes of the system's own packages or binaries, and the environment that
    sets threads."""
    return {
        **(git if git is not None else git_state()),
        "python": python,
        "pip_freeze_sha256": pip_freeze_sha256(python),
        "wheel_sha256": wheels or {},
        "env": env if env is not None else env_record(),
    }
