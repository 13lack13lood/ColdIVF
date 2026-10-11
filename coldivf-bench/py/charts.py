"""Charts of ColdIVF against the other systems for the README, and the numbers behind them.

    python charts.py [--suite SUITE.toml] [--quiet DIR] [--out DIR] [--repo DIR] [--data-only]

Two runs feed them. The smoke runs of a suite, results/bench/smoke for suites/smoke-1m.toml, hold every system's
build, warm grid and true-cold (C0) trials at the suite's dataset and prefix. The quiet-window run in results/quiet2
holds ColdIVF's own cold trials, the head-to-head through the driver at matched recall, the lazy-map control, the
AiSAQ-style model of a graph that keeps its codes on disk, and ColdIVF's warm grid and queries per second (QPS).
charts.py picks the lines each chart plots and writes chart-data.json, which holds every plotted number and every number
the README's text cites. Each comes with its source file, dataset, setting, trial count, machine state and the code
that measured it. It then draws each chart as scalable vector graphics (SVG), once for a light page and once for a dark
one, and prints the README's table of headline numbers.

The picks need only the standard library. The drawing needs matplotlib, and MATPLOTLIB below is the version that drew
the committed charts. It draws into a scratch folder and moves the files into place once every chart is drawn, so a
drawing that fails changes nothing. --data-only writes only chart-data.json.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tempfile
import tomllib
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

PY = Path(__file__).resolve().parent
REPO = PY.parents[1]
SUITE = PY.parent / "suites" / "smoke-1m.toml"
QUIET = REPO / "results" / "quiet2"
OUT = REPO / "docs" / "charts"

# The recall@10 the matched chart compares the systems at, the target that pick.py picks settings for.
TARGET = 0.95

# The recall@10 targets that the warm leaderboard ranks the systems at.
WARM_TARGETS = (0.85, 0.90, 0.95)

# The splits whose warm recall picks a matched cell, the tune split that pick.py picks on, else all the queries.
PICK_SPLITS = ("tune", "all")

# The matplotlib that drew the committed charts. Another version draws them too, but its SVGs differ a little.
MATPLOTLIB = "3.11.2"

DATASETS = {
    "cohere-768-cosine": "Cohere 1M, 768 dimensions, cosine, 1,000 queries",
    "cohere10m-768-cosine": "Cohere 10M, 768 dimensions, cosine, 1,000 queries",
}

# The name of each system in the charts, for the index its smoke build makes with the suite's settings.
NAMES = {
    "coldivf": "ColdIVF",
    "faiss-ivf": "faiss IVF RaBitQ",
    "faiss-hnsw": "faiss HNSW SQ8",
    "usearch": "usearch HNSW f16",
    "qdrant-edge": "qdrant-edge",
    "astervec": "AsterVec",
    "diskann": "DiskANN",
    "lancedb": "LanceDB IVF_RQ",
    "paimon": "Paimon IVF-RQ",
    "rabitq-ivf": "RaBitQ IVF",
    "rabitq-hnsw": "RaBitQ HNSW",
    "symphonyqg": "SymphonyQG",
    "turbovec": "turbovec",
    "sqlite-vec": "sqlite-vec",
    "flat": "flat scan",
}

# The order of the warm panels: inverted file (IVF) indexes, then graphs, then scans of every vector.
PANELS = ("coldivf", "faiss-ivf", "lancedb", "paimon", "rabitq-ivf", "faiss-hnsw", "usearch", "qdrant-edge",
          "astervec", "diskann", "rabitq-hnsw", "symphonyqg", "turbovec", "sqlite-vec", "flat")

# The names of the head-to-head cells of the quiet run, by system and config.
CELLS = {
    ("coldivf", "faiss-chained"): "ColdIVF, through the driver",
    ("lancedb", "rq1-default"): "LanceDB IVF_RQ",
    ("lancedb", "pq96-p1024"): "LanceDB IVF_PQ",
    ("faiss-ivf", "vpl160-rq1-shared"): "faiss IVF RaBitQ, mapped",
    ("usearch", "f16-m16"): "usearch HNSW f16, view()",
}

CPU_STATES = {"prewarm": "prewarmed core", "idle": "idle core"}

# The variant of a system that the smoke charts show, where the suite runs it more than one way. ColdIVF's python
# variant searches through its Python module in the adapter's process, as every other system searches through its
# binding, while its variant "" runs the release binaries, for comparing ColdIVF with itself.
SMOKE_VARIANTS = {"coldivf": "python"}

# The quiet run's jobs that time ColdIVF through its own cold binary: the job, the chart group, the read mode and the
# row's label. The read waves are ColdIVF's own path. The rest show how its first query changes with what came before
# it or how it reads, and the controls read the same file another way.
OWN_SCENARIOS = (
    ("a3-m1a", "coldivf", "wave", "ColdIVF, own harness, {state}"),
    ("a4-m1a-dummy", "coldivf", "wave", "ColdIVF after a search of another index, {state}"),
    ("a8-controls", "coldivf", "pool", "ColdIVF with thread-pool reads, {state}"),
    ("a8-controls", "control", "blocking", "ColdIVF's file read whole at open, {state} (control)"),
    ("a5-mapped", "control", "mapped", "ColdIVF's file, lazily mapped, {state} (control)"),
)

# A line of the quiet runner's log written right before a job starts, eg. "23:58:38 a3-m1a: CPU idle 85.16 91.21".
IDLE_LINE = re.compile(r"^\d\d:\d\d:\d\d (\S+): CPU idle ([\d.]+) ([\d.]+)$")


def read_lines(path: Path) -> list[dict]:
    """The JSON objects of a results file, one per line. A line that doesn't parse, eg. one cut short, is skipped."""
    if not path.exists():
        return []

    out = []

    for raw in path.read_text().splitlines():
        try:
            line = json.loads(raw)
        except ValueError:
            continue

        if isinstance(line, dict):
            out.append(line)

    return out


def percentile(values: list[float], p: float) -> float | None:
    """The nearest-rank p-th percentile, the one the quiet run's summary reports."""
    if not values:
        return None

    ordered = sorted(values)
    rank = math.ceil(p * len(ordered) / 100.0 - 1e-9)
    return ordered[min(max(rank, 1), len(ordered)) - 1]


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def span(values: list[float]) -> list[float] | None:
    return [min(values), max(values)] if values else None


def tidy(value):
    """`value` with float noise cut to 6 significant digits, eg. 0.9005999999999982 becomes 0.9006."""
    if isinstance(value, float):
        return float(f"{value:.6g}")

    if isinstance(value, dict):
        return {k: tidy(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [tidy(v) for v in value]

    return value


def rel(path: Path, repo: Path) -> str:
    """`path` relative to the repo when it is inside it."""
    try:
        return str(Path(path).resolve().relative_to(repo.resolve()))
    except ValueError:
        return str(path)


def setting_key(setting: dict) -> str:
    """`setting` as text that is the same for the same settings in any order."""
    return json.dumps(setting, sort_keys=True)


def index_key(line: dict):
    """The index a line names, by its checksum, or by its path, inode, change time and size without one."""
    ident = line.get("index_identity") or {}
    return ident.get("checksum") or (ident.get("path"), ident.get("inode"), ident.get("mtime_ns"), ident.get("bytes"))


def newest_batch(lines: list[dict]) -> list[dict]:
    """The lines of the batch that wrote the newest of `lines`."""
    if not lines:
        return []

    batch = max(lines, key=lambda line: line.get("ts") or "").get("batch_id")
    return [line for line in lines if line.get("batch_id") == batch]


def cell_key(line: dict) -> tuple:
    """What makes a cold cell: the system, its config and variant, the setting, the regime, CPU state and threads."""
    return (line.get("system"), line.get("config_id"), line.get("variant"),
            setting_key(line.get("search_params") or {}), line.get("regime"), line.get("cpu_state"),
            line.get("threads"))


def cell_groups(lines: list[dict]) -> dict[tuple, list[dict]]:
    """The cold lines of `lines` by the cell they belong to."""
    groups = defaultdict(list)

    for line in lines:
        groups[cell_key(line)].append(line)

    return groups


def largest_cell(lines: list[dict]) -> list[dict]:
    """The cell of `lines` with the most trials, ties to the lower median, or no lines."""
    cells = list(cell_groups(lines).values())

    if not cells:
        return []

    return min(cells, key=lambda cell: (-len(cell), percentile([line["total_us"] for line in cell], 50)))


def driver_state(lines: list[dict]) -> dict:
    """The machine state the driver recorded with `lines`: the timing gate of their batches and the 1-minute load."""
    gates = [line.get("gate") or {} for line in lines]
    first = gates[0] if gates else {}
    return {
        "batches": sorted({line.get("batch_id") for line in lines}),
        "gate_mode": first.get("mode"),
        "gate_passed": all(g.get("passed") is True for g in gates) if gates else None,
        "solo": all(g.get("solo") is True for g in gates) if gates else None,
        "cpu_idle_pct": span([g["cpu_idle_pct"] for g in gates if g.get("cpu_idle_pct") is not None]),
        "problems": sorted({p for g in gates for p in g.get("problems") or []}),
        "power": first.get("power"),
        "low_power": first.get("low_power"),
        "load_1": span([line["load_before"][0] for line in lines if line.get("load_before")]),
    }


def code_of(lines: list[dict], repo: Path | None = None) -> dict | list[dict] | None:
    """The code that measured `lines`. For the driver's lines it is the system's version, the hashes of the wheels or
    ColdIVF binaries it ran, the folder those binaries came from when the run named one, the commit a binary says it
    was built from, and the commit of the harness and whether its tree had changes. For the cold binary's own lines it
    is the commit the binary was built from, whether that tree had changes and when it was built. It is one record when
    every line agrees, else each distinct one."""
    records = []

    for line in lines:
        provenance = line.get("provenance")
        extra = line.get("extra") or {}

        if isinstance(provenance, dict):
            folder = (provenance.get("env") or {}).get("COLDIVF_BIN") if line.get("system") == "coldivf" else None
            record = {"system_version": line.get("system_version"), "wheel_sha256": provenance.get("wheel_sha256"),
                      "bin": rel(Path(folder), repo) if folder and repo else folder,
                      "binary_commit": extra.get("commit"), "binary_dirty": extra.get("dirty"),
                      "harness_head": provenance.get("head"), "harness_unstaged": provenance.get("unstaged")}
        else:
            record = {"commit": line.get("commit"), "dirty": line.get("dirty"), "built": line.get("built")}

        record = {k: v for k, v in record.items() if v is not None}

        if record and record not in records:
            records.append(record)

    return records[0] if len(records) == 1 else records or None


def smoke_pick(lines: list[dict], system: str, dataset: str, n: int, variant: str = "") -> dict:
    """One system's smoke lines at `dataset` and `n` of its `variant`: the warm settings and the C0 trials of its
    newest batch of each, and the build of the index they searched. Older batches, lines that aren't ok and dummy
    builds are left out."""
    ok = [line for line in lines if line.get("system") == system and line.get("status") == "ok"
          and line.get("dataset") == dataset and line.get("n") == n and (line.get("variant") or "") == variant]
    warm = newest_batch([line for line in ok if line.get("kind") == "warm" and line.get("threads") == 1
                         and line.get("p50_us") is not None])
    cold = largest_cell(newest_batch([line for line in ok if line.get("kind") == "cold" and line.get("regime") == "c0"
                                      and line.get("total_us") is not None]))
    searched = cold or warm
    build = None

    if searched:
        key = index_key(searched[0])
        builds = [line for line in ok if line.get("kind") == "build" and index_key(line) == key]
        build = max(builds, key=lambda line: line.get("ts") or "") if builds else None

    return {"system": system, "warm": sorted(warm, key=lambda line: line["p50_us"]), "cold": cold, "build": build}


def cold_summary(lines: list[dict]) -> dict:
    """The first-query times of one cell's cold trials, with the setting, recall and memory they share."""
    times = [line["total_us"] for line in lines]
    recall = [line["recall10"] for line in lines if line.get("recall10") is not None]
    deltas = [line["footprint_delta"] for line in lines if line.get("footprint_delta") is not None]
    first = lines[0]
    return {
        "config_id": first.get("config_id"),
        "setting": first.get("search_params") or {},
        "regime": first.get("regime"),
        "cpu_state": first.get("cpu_state"),
        "threads": first.get("threads"),
        "trials": len(lines),
        "p50_us": percentile(times, 50),
        "p95_us": percentile(times, 95),
        "p99_us": percentile(times, 99),
        "min_us": min(times),
        "max_us": max(times),
        "slowest_us": sorted(times)[-3:],
        "recall10_trials": mean(recall),
        "footprint_delta": ({"median": percentile(deltas, 50), "min": min(deltas), "max": max(deltas)}
                            if deltas else None),
    }


def warm_at(lines: list[dict], system: str, config_id: str | None, setting: dict,
            splits: tuple[str, ...] = ()) -> dict | None:
    """The newest ok warm line of `system` at `setting` and 1 thread, from `config_id` unless it is None. It takes the
    first split of `splits` that has one, and any split when none does."""
    found = [line for line in lines if line.get("kind") == "warm" and line.get("status") == "ok"
             and line.get("system") == system and line.get("threads") == 1
             and (config_id is None or line.get("config_id") == config_id)
             and (line.get("search_params") or {}) == setting and line.get("recall10") is not None]

    for split in splits:
        mine = [line for line in found if line.get("split") == split]

        if mine:
            found = mine
            break

    return max(found, key=lambda line: line.get("ts") or "") if found else None


def own_warm(lines: list[dict]) -> list[dict]:
    """The warm binary's latency lines at 1 thread, eg. b3-warm's, in the shape of the driver's warm lines, so that one
    lookup finds either."""
    return [{"kind": "warm", "status": "ok", "system": "coldivf", "config_id": None, "threads": 1,
             "search_params": {"nprobe": line.get("nprobe"), "rerank": line.get("rerank")},
             "split": line.get("split"), "ts": line.get("time"), "recall10": line.get("recall10"),
             "p50_us": line.get("p50_us"), "queries": line.get("queries"), "passes": line.get("passes"),
             "bytes_scanned": None, "commit": line.get("commit"), "dirty": line.get("dirty"),
             "built": line.get("built")}
            for line in lines if line.get("threads") == 1 and line.get("p50_us") is not None]


def matched_cells(cold: list[dict], warm_of) -> tuple[list[tuple[list[dict], dict, dict]], list[dict]]:
    """The cell at matched recall of each system, config, CPU state and thread count in `cold`. Of the C0 cells whose
    setting reaches TARGET warm, it keeps those with the most trials, the main cells a verdict needs, and takes the
    cheapest of them as pick.py does, the lowest warm p50 with ties to fewer bytes scanned. `warm_of(line, splits)`
    gives the warm line of a cell's setting from the first of `splits` that has one. Each chosen cell comes with the
    warm line that picked it and the one its row shows, which is on the split its trials searched when there is one.
    The groups with no cell that reaches TARGET come apart, with the best warm recall they reached."""
    groups = defaultdict(list)

    for key, cell in cell_groups(cold).items():
        system, config_id, variant, _, regime, cpu_state, threads = key
        split = cell[0].get("split")
        picked = warm_of(cell[0], PICK_SPLITS)
        shown = warm_of(cell[0], (split, "all") if split else ("all",))
        groups[(system, config_id, variant, regime, cpu_state, threads)].append((cell, picked, shown))

    chosen, left_out = [], []

    for (system, config_id, variant, _, cpu_state, threads), cells in groups.items():
        reach = [c for c in cells if c[1] is not None and c[1]["recall10"] >= TARGET]

        if not reach:
            recalls = [c[1]["recall10"] for c in cells if c[1] is not None]
            left_out.append({"system": system, "config_id": config_id, "variant": variant, "cpu_state": cpu_state,
                             "threads": threads, "best_recall10_warm": max(recalls) if recalls else None})
            continue

        most = max(len(c[0]) for c in reach)
        chosen.append(min((c for c in reach if len(c[0]) == most),
                          key=lambda c: (c[1].get("p50_us") or math.inf, c[1].get("bytes_scanned") or 0.0,
                                         percentile([line["total_us"] for line in c[0]], 50))))

    return chosen, left_out


def warm_fields(shown: dict, picked: dict, source: dict, repo: Path) -> dict:
    """The warm numbers a matched row shows. They are the recall and p50 of its setting, the split and queries they
    are over, the file they come from and the code that measured them, plus the recall that picked the cell when a
    line of another split picked it."""
    out = {"recall10_warm": shown.get("recall10"), "recall10_warm_queries": shown.get("queries"),
           "recall10_warm_split": shown.get("split"), "warm_p50_us": shown.get("p50_us"),
           "recall_source": rel(source[id(shown)], repo), "recall_code": code_of([shown], repo)}

    if picked is not shown:
        out |= {"recall10_pick": picked.get("recall10"), "recall10_pick_split": picked.get("split")}

    return out


def firsts(lines: list[dict]) -> list[dict]:
    """The first query of each of the cold binary's trials that started with none of the index cached."""
    return [line for line in lines if line.get("seq") == 0 and not line.get("pages_resident_at_start")]


def own_cells(lines: list[dict]) -> list[dict]:
    """The cold binary's trials by mode, CPU state and setting, as the quiet run's summary groups them."""
    groups = defaultdict(list)

    for line in firsts(lines):
        groups[(line.get("mode"), line.get("variant"), line.get("nprobe"), line.get("rerank"))].append(line)

    out = []

    for (mode, variant, nprobe, rerank), rows in sorted(groups.items(), key=lambda kv: str(kv[0])):
        times = [line["total_us"] for line in rows]
        deltas = [line["footprint_delta"] for line in rows if line.get("footprint_delta") is not None]
        out.append({
            "mode": mode, "cpu_state": variant, "setting": {"nprobe": nprobe, "rerank": rerank}, "trials": len(rows),
            "p50_us": percentile(times, 50), "p95_us": percentile(times, 95), "p99_us": percentile(times, 99),
            "slowest_us": sorted(times)[-3:],
            "recall10_trials": mean([line["recall10"] for line in rows if line.get("recall10") is not None]),
            "footprint_delta": percentile(deltas, 50), "dataset": rows[0].get("dataset"), "n": rows[0].get("n"),
            "load_1": span([line["load_1"] for line in rows if line.get("load_1") is not None]),
            "power": rows[0].get("power"), "low_power": rows[0].get("low_power"), "code": code_of(rows),
        })

    return out


def runner_idle(text: str) -> dict[str, list[float]]:
    """Each job's two CPU idle samples from the quiet runner's log, taken right before the job started."""
    out = {}

    for raw in text.splitlines():
        found = IDLE_LINE.match(raw.strip())

        if found:
            out[found.group(1)] = [float(found.group(2)), float(found.group(3))]

    return out


def quiet_state(job: str, idle: dict, load_1, power, low_power) -> dict:
    return {"run": "quiet window", "job": job, "runner_cpu_idle_pct": idle.get(job), "load_1": load_1,
            "power": power, "low_power": low_power}


def knob(settings: list[dict]) -> str:
    """The settings a warm grid sweeps, eg. "nprobe 16 to 256", or "one setting"."""
    keys = [k for k in (settings[0] if settings else {}) if len({json.dumps(s.get(k)) for s in settings}) > 1]

    if not keys:
        return "one setting"

    parts = []

    for k in keys:
        values = sorted(s[k] for s in settings)
        parts.append(f"{k} {values[0]} to {values[-1]}")

    return ", ".join(parts)


def machine_record(lines: list[dict]) -> dict:
    """The machine the driver's lines recorded: the chip, memory, operating system and drive."""
    for line in lines:
        m = line.get("machine")

        if isinstance(m, dict) and m.get("model"):
            return {"model": m.get("model"), "chip": m.get("brand"), "memory_bytes": m.get("memsize"),
                    "cores": [{"name": p.get("name"), "count": p.get("physicalcpu")} for p in m.get("perflevels", [])],
                    "os": (m.get("os") or {}).get("product_version"), "ssd": (m.get("ssd") or {}).get("model")}

    return {}


def baseline(row: dict, rows: list[dict]) -> dict | None:
    """The ColdIVF row through the driver that `row` is measured against, the fastest one in its CPU state, else the
    fastest in any."""
    ours = sorted((r for r in rows if r["group"] == "coldivf" and r["harness"] == "driver"), key=lambda r: r["p50_us"])
    same = [r for r in ours if r["cpu_state"] == row.get("cpu_state")] or ours
    return {k: same[0][k] for k in ("label", "setting", "cpu_state", "trials", "p50_us", "source")} if same else None


def cold_matched(quiet: Path, repo: Path) -> list[dict]:
    """The two charts of the cold first query at matched recall from the quiet run: ColdIVF in each of its setups, and
    ColdIVF through the driver against one row per other system, with ColdIVF's warm numbers there."""
    idle = runner_idle((quiet / "summary.md").read_text() if (quiet / "summary.md").exists() else "")
    b3 = read_lines(quiet / "b3-warm.jsonl")
    own = own_warm(b3)
    driver_lines = [(path, line) for path in sorted(quiet.glob("driver/*/*.jsonl")) for line in read_lines(path)]
    source = {id(line): path for path, line in driver_lines} | {id(line): quiet / "b3-warm.jsonl" for line in own}
    cold = [line for _, line in driver_lines if line.get("kind") == "cold" and line.get("status") == "ok"
            and line.get("regime") == "c0" and line.get("total_us") is not None]
    warm = [line for _, line in driver_lines if line.get("kind") == "warm"]

    def warm_of(line: dict, splits: tuple[str, ...]) -> dict | None:
        setting = line.get("search_params") or {}
        found = warm_at(warm, line["system"], line.get("config_id"), setting, splits)

        if found is None and line["system"] == "coldivf":
            found = warm_at(own, "coldivf", None, setting, splits)

        return found

    chosen, left_out = matched_cells(cold, warm_of)
    picked_ids = {id(cell[0]) for cell, _, _ in chosen}
    others = []

    for cell in cell_groups(cold).values():
        if id(cell[0]) in picked_ids:
            continue

        s = cold_summary(cell)
        split = cell[0].get("split")
        shown = warm_of(cell[0], (split, "all") if split else ("all",)) or {}
        others.append({"system": cell[0]["system"], "config_id": s["config_id"], "setting": s["setting"],
                       "cpu_state": s["cpu_state"], "threads": s["threads"], "trials": s["trials"],
                       "p50_us": s["p50_us"], "p99_us": s["p99_us"], "recall10_warm": shown.get("recall10"),
                       "warm_p50_us": shown.get("p50_us"), "source": rel(source[id(cell[0])], repo)})

    rows = []

    for cell, picked, shown in chosen:
        s = cold_summary(cell)
        system = cell[0]["system"]
        state = CPU_STATES.get(s["cpu_state"], s["cpu_state"])
        label = CELLS.get((system, s["config_id"]), f"{NAMES.get(system, system)} {s['config_id']}")
        rows.append({
            "label": label if s["cpu_state"] in (None, "prewarm") else f"{label}, {state}",
            "group": "coldivf" if system == "coldivf" else "system", "harness": "driver", "system": system,
            "config_id": s["config_id"], "setting": s["setting"], "dataset": cell[0].get("dataset"),
            "n": cell[0].get("n"), "regime": s["regime"], "cpu_state": s["cpu_state"], "threads": s["threads"],
            "trials": s["trials"], "p50_us": s["p50_us"], "p95_us": s["p95_us"], "p99_us": s["p99_us"],
            "slowest_us": s["slowest_us"], "recall10_trials": s["recall10_trials"],
            **warm_fields(shown, picked, source, repo), "modeled": False,
            "source": rel(source[id(cell[0])], repo), "machine_state": driver_state(cell), "code": code_of(cell, repo),
        })

    for job, group, mode, name in OWN_SCENARIOS:
        for c in own_cells(read_lines(quiet / f"{job}.jsonl")):
            if c["mode"] != mode:
                continue

            shown = warm_at(own, "coldivf", None, c["setting"])

            if shown is None or shown["recall10"] < TARGET:
                left_out.append({"system": "coldivf", "job": job, "mode": mode, "cpu_state": c["cpu_state"],
                                 "setting": c["setting"], "best_recall10_warm": (shown or {}).get("recall10")})
                continue

            label = name.format(state=CPU_STATES.get(c["cpu_state"], c["cpu_state"]))
            rows.append({
                "label": label, "group": group, "harness": "own", "job": job, "system": "coldivf", "config_id": None,
                "setting": c["setting"], "dataset": c["dataset"], "n": c["n"], "regime": "c0", "mode": c["mode"],
                "cpu_state": c["cpu_state"], "threads": 1, "trials": c["trials"], "p50_us": c["p50_us"],
                "p95_us": c["p95_us"], "p99_us": c["p99_us"], "slowest_us": c["slowest_us"],
                "recall10_trials": c["recall10_trials"], **warm_fields(shown, shown, source, repo), "modeled": False,
                "source": rel(quiet / f"{job}.jsonl", repo),
                "machine_state": quiet_state(job, idle, c["load_1"], c["power"], c["low_power"]), "code": c["code"],
            })

    for w in read_lines(quiet / "a7-aisaq.jsonl"):
        if (w.get("rounds") or 0) <= 1:
            continue

        rows.append({
            "label": f"AiSAQ-style graph, {w['rounds']} rounds (modeled)", "group": "model", "harness": "model",
            "system": None, "config_id": None,
            "setting": {"rounds": w["rounds"], "ranges": w.get("ranges"), "page": w.get("page")},
            "dataset": None, "n": None, "regime": "c0", "cpu_state": None, "threads": 1, "trials": w.get("trials"),
            "p50_us": w.get("p50_us"), "p95_us": None, "p99_us": w.get("p99_us"), "round_p50_us": w.get("round_p50_us"),
            "file_bytes": w.get("size"), "recall10_trials": None, "recall10_warm": None,
            "recall10_warm_queries": None, "modeled": True, "source": rel(quiet / "a7-aisaq.jsonl", repo),
            "recall_source": None,
            "machine_state": quiet_state("a7-aisaq", idle, [w.get("load_1")] * 2, w.get("power"), w.get("low_power")),
            "code": code_of([w]),
        })

    # A control is measured against the read waves that its own job interleaved with it, at its setting and CPU state,
    # and every other row against ColdIVF through the driver.
    waves = {(job, c["cpu_state"], setting_key(c["setting"])): c
             for job in ("a5-mapped", "a8-controls") for c in own_cells(read_lines(quiet / f"{job}.jsonl"))
             if c["mode"] == "wave"}

    for r in rows:
        if r["group"] == "control":
            w = waves.get((r["job"], r["cpu_state"], setting_key(r["setting"])))
            r["against"] = None if w is None else {
                "label": f"ColdIVF, own harness, read waves, {CPU_STATES.get(w['cpu_state'], w['cpu_state'])}",
                "setting": w["setting"], "cpu_state": w["cpu_state"], "trials": w["trials"], "p50_us": w["p50_us"],
                "source": rel(quiet / f"{r['job']}.jsonl", repo),
            }
        else:
            r["against"] = baseline(r, rows)

        r["p50_ratio"] = r["p50_us"] / r["against"]["p50_us"] if r["against"] else None

    measured = sorted((r for r in rows if not r["modeled"]), key=lambda r: r["p50_us"])
    modeled = sorted((r for r in rows if r["modeled"]), key=lambda r: r["p50_us"])
    datasets = sorted({r["dataset"] for r in measured if r["dataset"]})
    ours = baseline({"cpu_state": "prewarm"}, rows)
    point = (ours or next((r for r in measured if r["group"] == "coldivf"), {})).get("setting")

    # The head-to-head keeps one row per system, its fastest config at the target, all timed the same way through the
    # driver. The configs it drops stay in other_cells.
    fastest = {}

    for r in measured:
        if r["harness"] == "driver":
            fastest.setdefault(r["system"], r)

    head = list(fastest.values())
    dropped = [r for r in measured if r["harness"] == "driver" and r not in head]
    others += [{k: r.get(k) for k in ("system", "config_id", "setting", "cpu_state", "threads", "trials", "p50_us",
                                       "p99_us", "recall10_warm", "warm_p50_us", "source")} for r in dropped]
    common = {
        "dataset": datasets[0] if len(datasets) == 1 else datasets,
        "n": measured[0]["n"] if measured else None,
        "unit": "microseconds",
        "target_recall10": TARGET,
    }
    pick_note = (f"Each row runs at a setting whose recall@10 reaches {TARGET:.2f} warm. Of those timed cold, it is "
                 "the one with the most trials, and of those with as many, the cheapest warm, the lowest p50 and then "
                 "the fewest bytes scanned, as pick.py ranks settings.")
    head_to_head = {
        "id": "cold-head-to-head",
        "title": "Cold first query against the other systems, recall@10 near 0.95",
        **common,
        "legend": ["ColdIVF", "Other systems"],
        "footer": "Every row ran interleaved through the same driver in one quiet window. Each trial is a fresh "
                  "process with the index evicted from the cache, on a prewarmed core that spins for 150 ms before the "
                  "query, and each system shows its fastest config. The other systems join this chart as their matched "
                  "runs finish.",
        "notes": [
            pick_note,
            "Each system has one row, its fastest config at the target. Every row ran interleaved through one driver, "
            "each trial a fresh process with the index evicted, on a prewarmed core.",
            "Each p50_ratio is the row's p50 over ColdIVF's through the driver.",
            "models are reads-only models, which the chart doesn't plot. The AiSAQ-style model times only the reads "
            "of a graph that keeps its codes on disk, i.e. dependent rounds of 4 random 16 KiB reads from a 1 GiB "
            "file, with no compute, for an index ~5 to 15 times ColdIVF's size.",
            "coldivf_warm is ColdIVF's warm p50 and its queries per second at the setting of its row through the "
            "driver, on the quiet run's index.",
            "other_cells are the cells the driver timed that the chart doesn't plot, and left_out the systems and "
            "cells with no setting that reaches the target.",
        ],
        "rows": head,
        "models": modeled,
        "other_cells": sorted(others, key=lambda c: (c["system"], c["config_id"] or "", c["p50_us"])),
        "left_out": left_out,
        "coldivf_warm": quiet_warm(b3, point, quiet, repo, idle) if point else None,
    }
    scenarios = {
        "id": "cold-coldivf",
        "title": "ColdIVF's cold first query in each setup, recall@10 near 0.95",
        **common,
        "legend": ["ColdIVF", "Controls, ColdIVF's file read another way"],
        "footer": "A prewarmed core spins for 150 ms before the query, and an idle core sleeps 200 ms first, so its "
                  "query can start on an efficiency core. The driver runs a fresh process per trial with the index "
                  "evicted, and the own harness is ColdIVF's cold binary. After a search of another index, the "
                  "process has first searched a 50K index of its own. The lazy map searches ColdIVF's file through a "
                  "memory map that faults pages in, at the control's setting rather than its cheapest one.",
        "notes": [
            pick_note,
            "Every row is ColdIVF's index, timed through the driver or ColdIVF's own cold binary, with the read path "
            "and the state before the query that its label names.",
            "Each p50_ratio is the row's p50 over ColdIVF's through the driver, and a control's over the read waves "
            "that its own job interleaved with it, at the same setting and CPU state.",
        ],
        "rows": [r for r in measured if r["system"] == "coldivf"],
    }
    return [scenarios, head_to_head]


def quiet_warm(b3: list[dict], setting: dict, quiet: Path, repo: Path, idle: dict) -> dict | None:
    """ColdIVF's warm p50 at 1 thread and its queries per second at each thread count, at `setting` in the quiet run."""
    at = [w for w in b3 if w.get("nprobe") == setting.get("nprobe") and w.get("rerank") == setting.get("rerank")]
    latency = [w for w in at if w.get("threads") == 1 and w.get("p50_us") is not None]
    qps = sorted((w for w in at if w.get("qps") is not None), key=lambda w: w.get("threads") or 0)

    if not latency and not qps:
        return None

    w = max(latency, key=lambda w: w.get("time") or "") if latency else {}
    first = w or qps[0]
    return {
        "setting": setting, "dataset": first.get("dataset"), "n": first.get("n"), "recall10": first.get("recall10"),
        "queries": first.get("queries"), "split": first.get("split"), "p50_us": w.get("p50_us"),
        "p99_us": w.get("p99_us"), "passes": w.get("passes"),
        "qps": [{"threads": q.get("threads"), "qps": q["qps"], "seconds": q.get("seconds"), "runs": q.get("qps_runs")}
                for q in qps],
        "source": rel(quiet / "b3-warm.jsonl", repo),
        "machine_state": quiet_state("b3-warm", idle, span([x["load_1"] for x in at if x.get("load_1") is not None]),
                                     first.get("power"), first.get("low_power")),
        "code": code_of(([w] if w else []) + qps),
    }


def interpolate(points: list[dict], recall: float) -> float | None:
    """The p50 of the grid `points` at `recall`, linear in the log of the latency between the two points around it,
    or None outside the grid's recall."""
    ordered = sorted((p for p in points if p.get("recall10") is not None), key=lambda p: p["recall10"])

    for a, b in zip(ordered, ordered[1:] or ordered):
        if a["recall10"] <= recall <= b["recall10"]:
            if b["recall10"] == a["recall10"]:
                return min(a["p50_us"], b["p50_us"])

            f = (recall - a["recall10"]) / (b["recall10"] - a["recall10"])
            return math.exp((1 - f) * math.log(a["p50_us"]) + f * math.log(b["p50_us"]))

    return None


def against_coldivf(curve: dict, ours: dict) -> dict:
    """Each point of `curve` against ColdIVF's p50 at the same recall, for the points inside the recall of ColdIVF's
    grid. The ratio is the point's p50 over ColdIVF's. ratio_less_call takes the median time of one call on a
    1,000-vector index out of the point's p50 first. That call holds the binding's per-call cost and a small search,
    so it is left out for a system whose call takes longer than one of its own searches."""
    points = [p for p in curve["points"] if p.get("recall10") is not None]
    calls = [p["overhead_us"] for p in points if p.get("overhead_us") is not None]
    usable = bool(calls) and max(calls) < min(p["p50_us"] for p in points)
    # ColdIVF's own call, once it runs through its binding, comes out of its time too. Its binaries pay none.
    mine = [p["overhead_us"] for p in ours["points"] if p.get("overhead_us")]
    own = max(mine) if mine and max(mine) < min(p["p50_us"] for p in ours["points"]) else 0.0
    out = []

    for p in sorted(points, key=lambda p: p["recall10"]):
        base = interpolate(ours["points"], p["recall10"])

        if base is None:
            continue

        call = p.get("overhead_us")
        out.append({"setting": p["setting"], "recall10": p["recall10"], "p50_us": p["p50_us"],
                    "coldivf_p50_us": base, "ratio": p["p50_us"] / base,
                    "ratio_less_call": (p["p50_us"] - call) / (base - own) if usable and call is not None
                    else None})

    return {"call_us": span(calls), "call_longer_than_a_search": bool(calls) and not usable, "points": out}


def at_recall(points: list[dict], target: float) -> dict | None:
    """A warm grid's p50 at recall@10 `target`. Inside the grid it is interpolated, linear in the log of the latency
    between the two settings around it. Where every setting is above `target`, it is the fastest of them, with that
    setting and its recall. It is None when no setting reaches `target`."""
    between = interpolate(points, target)

    if between is not None:
        return {"p50_us": between, "between": True}

    reach = [p for p in points if (p.get("recall10") or 0.0) >= target]

    if not reach:
        return None

    fastest = min(reach, key=lambda p: p["p50_us"])
    return {"p50_us": fastest["p50_us"], "between": False, "setting": fastest["setting"],
            "recall10": fastest["recall10"]}


def leaderboard(curves: list[dict]) -> tuple[list[dict], list[dict]]:
    """The warm leaderboard: at each of WARM_TARGETS, the systems whose grid reaches it, fastest first, and the ones
    whose grid doesn't. A system whose grid reaches none of the targets is left off it, and comes apart with its best
    recall."""
    best = {c["system"]: max((p["recall10"] for p in c["points"] if p.get("recall10") is not None), default=None)
            for c in curves}
    on = [c for c in curves if best[c["system"]] is not None and best[c["system"]] >= min(WARM_TARGETS)]
    left_out = [{"system": c["system"], "label": c["label"], "best_recall10": best[c["system"]]}
                for c in curves if c not in on]
    boards = []

    for target in WARM_TARGETS:
        rows, short = [], []

        for c in on:
            found = at_recall(c["points"], target)

            if found is None:
                short.append({"system": c["system"], "label": c["label"], "best_recall10": best[c["system"]]})
                continue

            calls = [p["overhead_us"] for p in c["points"] if p.get("overhead_us") is not None]
            usable = bool(calls) and max(calls) < min(p["p50_us"] for p in c["points"])
            rows.append({"system": c["system"], "label": c["label"], **found,
                         "p50_less_call_us": found["p50_us"] - max(calls) if usable else None})

        boards.append({"recall10": target, "rows": sorted(rows, key=lambda r: r["p50_us"]), "not_reached": short})

    return boards, left_out


def smoke_charts(suite_path: Path, repo: Path) -> tuple[dict, dict, dict, list[str], dict]:
    """Charts (b), (c) and (d) from the suite's smoke runs, the systems with no lines, and the machine."""
    suite = tomllib.loads(suite_path.read_text())
    dataset, n = suite["dataset"], int(suite["n"])
    root = Path(suite.get("results_root", "results/bench"))
    root = (root if root.is_absolute() else repo / root) / suite["phase"]
    names = [s["name"] for s in suite.get("system", [])]
    order = [s for s in PANELS if s in names] + [s for s in names if s not in PANELS]
    curves, points, memory, missing, everything = [], [], [], [], []

    for system in order:
        path = root / f"{system}.jsonl"
        lines = read_lines(path)
        everything += lines
        pick = smoke_pick(lines, system, dataset, n, SMOKE_VARIANTS.get(system, ""))
        label = NAMES.get(system, system)
        source = rel(path, repo)

        if not pick["warm"] and not pick["cold"]:
            missing.append(system)
            continue

        if pick["warm"]:
            settings = [w.get("search_params") or {} for w in pick["warm"]]
            curves.append({
                "system": system, "label": label, "dataset": dataset, "n": n, "knob": knob(settings),
                "binding": pick["warm"][0].get("binding"), "source": source,
                "machine_state": driver_state(pick["warm"]), "code": code_of(pick["warm"], repo),
                "points": [{"setting": w.get("search_params") or {}, "p50_us": w["p50_us"], "p99_us": w.get("p99_us"),
                            "qps": w.get("qps"), "recall10": w.get("recall10"), "queries": w.get("queries"),
                            "passes": w.get("passes"), "overhead_us": w.get("overhead_us")} for w in pick["warm"]],
            })

        if not pick["cold"]:
            continue

        c = cold_summary(pick["cold"])
        at = warm_at(pick["warm"], system, c["config_id"], c["setting"])
        threads = f" ({c['threads']} threads)" if (c["threads"] or 1) > 1 else ""
        points.append({
            "system": system, "label": label + threads, "dataset": dataset, "n": n, "config_id": c["config_id"],
            "setting": c["setting"], "regime": c["regime"], "cpu_state": c["cpu_state"], "threads": c["threads"],
            "trials": c["trials"], "p50_us": c["p50_us"], "min_us": c["min_us"], "max_us": c["max_us"],
            "recall10_warm": (at or {}).get("recall10"), "recall10_warm_queries": (at or {}).get("queries"),
            "recall10_trials": c["recall10_trials"], "source": source, "machine_state": driver_state(pick["cold"]),
            "code": code_of(pick["cold"], repo),
        })
        warm_line = at or (max(pick["warm"], key=lambda w: w.get("footprint") or 0) if pick["warm"] else None)
        extra = (warm_line or {}).get("extra") or {}
        since_open = extra.get("footprint_since_open")
        build = pick["build"] or {}
        memory.append({
            "system": system, "label": label, "dataset": dataset, "n": n, "source": source,
            "cold": {"setting": c["setting"], "trials": c["trials"], "footprint_delta": c["footprint_delta"]},
            "warm": None if warm_line is None else {
                "setting": warm_line.get("search_params") or {},
                "process_bytes": since_open if since_open is not None else warm_line.get("footprint"),
                "process_source": "footprint_since_open" if since_open is not None else "footprint, the whole process",
                "footprint": warm_line.get("footprint"), "resident_file_bytes": warm_line.get("resident_file_bytes"),
            },
            "build": None if not build else {
                "peak_footprint": build.get("peak_footprint"), "threads": build.get("threads"),
                "seconds": build.get("seconds"), "batch_id": build.get("batch_id"),
            },
            "disk": None if not build else {
                "index_bytes": build.get("index_bytes"), "n": build.get("n"),
                "bytes_per_vector": build["index_bytes"] / build["n"] if build.get("index_bytes") else None,
                "exact_f32_stored": build.get("exact_f32_stored"),
                "stored_vector_bytes": build.get("stored_vector_bytes"),
            },
            "machine_state": driver_state(pick["cold"] + ([build] if build else [])),
            "code": code_of(pick["cold"] + ([warm_line] if warm_line else []) + ([build] if build else []), repo),
        })

    ours = next((c for c in curves if c["system"] == "coldivf"), None)

    for c in curves:
        if ours is not None and c is not ours:
            c["against_coldivf"] = against_coldivf(c, ours)

    queries = fmt_range([p["queries"] for c in curves for p in c["points"]])
    busy = busy_note([c["machine_state"] for c in curves], [c["system"] == "coldivf" for c in curves])
    build_threads = fmt_range([(r["build"] or {}).get("threads") for r in memory])
    boards, left_out = leaderboard(curves)
    targets = ", ".join(f"{v:.2f}" for v in WARM_TARGETS)
    warm = {
        "id": "warm-leaderboard", "title": "Fastest warm query at three recall targets", "dataset": dataset, "n": n,
        "unit": "microseconds",
        "notes": [f"The smoke grids run at 1 thread over {queries} queries of {dataset}'s tune split, each index "
                  "read into memory first.", busy,
                  binding_note(curves) + " Each point's overhead_us is the median time of one call on a "
                  "1,000-vector index, which holds that cost and a small search.",
                  "against_coldivf compares each point with ColdIVF's p50 at the same recall, linear in the log of the "
                  "latency between the two settings of ColdIVF's grid around it. ratio_less_call takes that call's "
                  "time out of the point's p50 first, and is left out where the call takes longer than one of the "
                  "system's own searches.",
                  f"boards ranks the systems at recall@10 {targets} by warm p50, read off each grid as against_coldivf "
                  "does. Where every setting of a grid is above a target, it takes the fastest, with between false. "
                  "not_reached are the systems whose grid doesn't reach that target, and left_out those whose grid "
                  "reaches none. p50_less_call_us takes the call's time out first, where it is shorter than every "
                  "search of the grid."],
        "curves": curves,
        "boards": boards,
        "left_out": left_out,
    }
    cold = {
        "id": "cold-smoke", "title": "Cold first query against recall at each system's smoke setting",
        "dataset": dataset, "n": n, "unit": "microseconds",
        "notes": ["Recall isn't matched. Each system runs one setting near its defaults.",
                  "Each trial is a fresh process with the index evicted (C0), on a prewarmed core.", busy,
                  f"Recall is the setting's warm recall@10 over {queries} tune queries. The trials search test "
                  "queries."],
        "points": points,
    }
    mem = {
        "id": "memory", "title": "Memory at 1M vectors", "dataset": dataset, "n": n, "unit": "bytes",
        "notes": ["Cold is the footprint the open and the first query add to the trial's process, the median of its "
                  "C0 trials.",
                  "Warm is at the cold setting, after reading every index file. It is the process memory the index "
                  f"adds, {process_note(memory)}, plus the index's pages in the page cache.",
                  f"Build is the peak footprint of the build process, on {build_threads} threads.",
                  "Disk is the index's bytes over its vectors."],
        "rows": memory,
    }
    for chart in (warm, cold, mem):
        chart["notes"] = [note for note in chart["notes"] if note]

    return warm, cold, mem, missing, machine_record(everything)


def binding_note(curves: list[dict]) -> str:
    """How the warm times hold each binding's per-call cost, by whether ColdIVF ran through its Python module too."""
    ours = next((c for c in curves if c["system"] == "coldivf"), {})

    if ours.get("binding") == "python":
        return ("Every system runs through its Python binding, in the adapter's process, and the times are raw, so "
                "each holds its binding's per-call cost.")

    return ("The times of the Python adapters are raw, so they include each binding's per-call cost, which ColdIVF's "
            "binary doesn't pay.")


def process_note(rows: list[dict]) -> str:
    """Where the warm process memory of each system in the memory chart comes from."""
    whole = [r["label"] for r in rows if r["warm"] and r["warm"]["process_source"] != "footprint_since_open"]

    if not whole:
        return "from each process's footprint since the open, which leaves out the interpreter"

    return (f"from the footprint since the open, which leaves out the interpreter, and for {', '.join(whole)} the "
            "whole process")


def busy_note(states: list[dict], ours: list[bool]) -> str:
    """How idle the machine was when each system's batch started, from the gate the driver recorded, with ColdIVF's
    batches apart from the others'."""
    theirs = [v for s, o in zip(states, ours) if not o for v in (s.get("cpu_idle_pct") or [])]
    mine = [v for s, o in zip(states, ours) if o for v in (s.get("cpu_idle_pct") or [])]
    note = (f"The smoke cells checked the harness, not the machine. Other jobs ran beside most of them, at "
            f"{min(theirs):.0f}% to {max(theirs):.0f}% idle CPU when each batch started") if theirs else ""
    return note + (f", and ColdIVF's refreshed cells at {min(mine):.0f}%." if mine and note else "." if note else "")


def chart_data(suite: Path = SUITE, quiet: Path = QUIET, repo: Path = REPO) -> dict:
    """Every number the charts plot and the README's text cites, with where it came from."""
    warm, cold, mem, missing, machine = smoke_charts(suite, repo)
    scenarios, head_to_head = cold_matched(quiet, repo)
    return tidy({
        "about": "Every number the README's charts plot and its text cites, written by coldivf-bench/py/charts.py. "
                 "The results are preliminary, and the full head-to-head runs will replace them.",
        "preliminary": True,
        "machine": machine,
        "drawn_with": None,
        "datasets": dict(DATASETS),
        "smoke_suite": rel(suite, repo),
        "missing_systems": missing,
        "charts": [scenarios, head_to_head, warm, cold, mem],
    })


def fmt_ms(us: float) -> str:
    """Microseconds as milliseconds with ~3 significant digits, eg. "6.26 ms" or "654 ms"."""
    v = us / 1000

    if v >= 100:
        return f"{v:,.0f} ms"

    return f"{v:.1f} ms" if v >= 10 else f"{v:.2f} ms"


def fmt_bytes(b: float) -> str:
    """Bytes in decimal units with 3 significant digits, eg. "5.54 MB" or "954 MB"."""
    for unit, size in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if b >= size:
            v = b / size
            return f"{v:.3g} {unit}" if v < 100 else f"{v:.0f} {unit}"

    return f"{b:.0f} B"


def fmt_setting(row: dict) -> str:
    """A row's setting in words, eg. "nprobe 96, rerank 35", or the reads of a modeled row."""
    s = row["setting"]

    if row.get("modeled"):
        return f"{s['rounds']} rounds of {s['ranges']} × {s['page'] // 1024} KiB reads"

    return ", ".join(f"{k} {v}" for k, v in s.items()) or "no setting"


def fmt_range(values: list[float], unit: str = "") -> str:
    """The span of `values` in words, eg. "5" when every value is 5, or "30 to 300"."""
    values = [v for v in values if v is not None]

    if not values:
        return "an unknown number of"

    lo, hi = min(values), max(values)
    return f"{lo:g}{unit}" if lo == hi else f"{lo:g} to {hi:g}{unit}"


def fmt_count(values: list[int]) -> str:
    """The span of counts in words with thousands commas, eg. "1,000", or "500 to 1,000"."""
    values = [v for v in values if v is not None]

    if not values:
        return "an unknown number of"

    lo, hi = min(values), max(values)
    return f"{lo:,}" if lo == hi else f"{lo:,} to {hi:,}"


def fmt_us(us: float) -> str:
    return f"{us:.0f} µs" if us < 1000 else fmt_ms(us)


def millions(n: int) -> str:
    return f"{n / 1e6:g}M"


def headline_table(data: dict, chart_id: str = "cold-head-to-head") -> str:
    """A README table: each row of a matched chart, its setting, recall, times and ratio."""
    rows = next(c for c in data["charts"] if c["id"] == chart_id)["rows"]
    out = ["| System | Setting | Recall@10 | Trials | Cold p50 | Cold p95 | Cold p99 | p50 Against ColdIVF |",
           "|---|---|---|---|---|---|---|---|"]

    for r in rows:
        recall = f"{r['recall10_warm']:.3f}" if r.get("recall10_warm") is not None else "-"
        p95 = fmt_ms(r["p95_us"]) if r.get("p95_us") is not None else "-"
        ratio = f"{r['p50_ratio']:.2f}x" if r.get("p50_ratio") is not None else "-"
        out.append(f"| {r['label']} | {fmt_setting(r)} | {recall} | {r['trials']} | {fmt_ms(r['p50_us'])} | {p95} | "
                   f"{fmt_ms(r['p99_us'])} | {ratio} |")

    return "\n".join(out)


# Each chart's colors on a light and a dark page. ColdIVF is blue, the system a panel is about is orange, and
# everything else is gray, so the one series that matters stands out.
THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e1e0d9", "axis": "#c3c2b7",
              "coldivf": "#2a78d6", "coldivf_tint": "#86b6ef", "focus": "#eb6834", "rest": "#898781",
              "rest_tint": "#c3c2b7", "context": "#d4d3cc"},
    "dark": {"surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7", "grid": "#2c2c2a", "axis": "#383835",
             "coldivf": "#3987e5", "coldivf_tint": "#184f95", "focus": "#d95926", "rest": "#898781",
             "rest_tint": "#52514e", "context": "#4a4a46"},
}

# The page width and margin in points. GitHub scales a README image to its column, ~880 pixels wide.
WIDTH = 756.0
MARGIN = 24.0


@lru_cache(maxsize=1)
def _mpl() -> SimpleNamespace:
    """The parts of matplotlib the drawing uses, imported on first use so the picks run without it."""
    import matplotlib
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.path import Path as MPath
    from matplotlib.ticker import FixedLocator, MaxNLocator, NullLocator
    from matplotlib.transforms import IdentityTransform, blended_transform_factory

    class Bar(Patch):
        """A horizontal bar from x0 to x1 at y, `height` points thick, square at x0 and rounded by `radius` points at
        x1, its data end. Its outline is found in display space at draw time. `gap` moves its start right by that many
        points, the surface gap after the segment it continues."""

        def __init__(self, x0, x1, y, height, radius=4.0, gap=0.0, **kwargs):
            super().__init__(**kwargs)
            self.span = (x0, x1, y, height, radius, gap)

        def get_path(self):
            x0, x1, y, height, radius, gap = self.span
            (a, mid), (b, _) = self.axes.transData.transform([(x0, y), (x1, y)])
            scale = self.figure.dpi / 72.0
            half = height * scale / 2
            a = min(a + gap * scale, b)
            r = max(0.0, min(radius * scale, half, b - a))
            verts = [(a, mid - half), (b - r, mid - half), (b, mid - half), (b, mid - half + r), (b, mid + half - r),
                     (b, mid + half), (b - r, mid + half), (a, mid + half), (a, mid - half)]
            codes = [MPath.MOVETO, MPath.LINETO, MPath.CURVE3, MPath.CURVE3, MPath.LINETO, MPath.CURVE3, MPath.CURVE3,
                     MPath.LINETO, MPath.CLOSEPOLY]
            return MPath(verts, codes)

        def get_transform(self):
            return IdentityTransform()

    scratch = Figure(figsize=(1, 1), dpi=72)
    FigureCanvasAgg(scratch)
    return SimpleNamespace(matplotlib=matplotlib, Figure=Figure, Canvas=FigureCanvasAgg, Line2D=Line2D, Patch=Patch,
                           Bar=Bar, FixedLocator=FixedLocator, MaxNLocator=MaxNLocator, NullLocator=NullLocator,
                           blended=blended_transform_factory, scratch=scratch)


def _rc(t: dict) -> dict:
    return {
        "svg.fonttype": "none", "svg.hashsalt": "coldivf-charts", "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 9.5, "text.color": t["ink"],
        "axes.facecolor": t["surface"], "figure.facecolor": t["surface"], "savefig.facecolor": t["surface"],
        "axes.edgecolor": t["axis"], "axes.linewidth": 0.8, "axes.labelcolor": t["ink2"], "axes.labelsize": 9,
        "xtick.color": t["axis"], "ytick.color": t["axis"], "xtick.labelcolor": t["ink2"],
        "ytick.labelcolor": t["ink2"], "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.labelcolor": t["ink2"],
        "legend.fontsize": 8.5, "legend.frameon": False, "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        # Hinted glyphs at 72 dots per inch come out up to 7% narrower or wider than a browser draws them, which breaks
        # the wrapping and the label placement, so text is measured unhinted.
        "text.hinting": "no_hinting",
    }


def text_width(s: str, size: float, weight: str = "normal") -> float:
    """The width of `s` in points, as matplotlib lays it out at 72 dots per inch."""
    m = _mpl()
    text = m.scratch.text(0, 0, s, size=size, weight=weight)
    width = text.get_window_extent(m.scratch.canvas.get_renderer()).width
    text.remove()
    return width


def wrap(text: str, size: float, width: float) -> list[str]:
    """`text` broken at spaces into lines of at most `width` points."""
    lines, line = [], ""

    for word in text.split():
        trial = f"{line} {word}" if line else word

        if line and text_width(trial, size) > width:
            lines.append(line)
            line = word
        else:
            line = trial

    return lines + ([line] if line else [])


class Page:
    """A figure laid out in points from its top left corner, with the title, subtitle and footnotes every chart has.
    `body` is the height left for the chart between the subtitle and the footnotes, which starts at `top`."""

    def __init__(self, t: dict, title: str, subtitle: list[str], body: float, footer: list[str] = ()):
        m = _mpl()
        self.t = t
        width = (WIDTH - 2 * MARGIN) * 0.97
        sub = [line for para in subtitle for line in wrap(para, 9, width)]
        foot = [line for para in footer for line in wrap(para, 8, width)]
        self.top = 48 + 13.5 * len(sub)
        self.bottom = self.top + body
        self.w, self.h = WIDTH, self.bottom + 12 * len(foot) + 16
        self.fig = m.Figure(figsize=(self.w / 72, self.h / 72), dpi=72)
        m.Canvas(self.fig)
        self.text(MARGIN, 28, title, size=14, weight="bold", color=t["ink"])

        for i, line in enumerate(sub):
            self.text(MARGIN, 47 + 13.5 * i, line, size=9, color=t["ink2"])

        for i, line in enumerate(foot):
            self.text(MARGIN, self.bottom + 10 + 12 * i, line, size=8, color=t["ink2"])

    def text(self, x: float, y: float, s: str, **kwargs):
        kwargs.setdefault("va", "baseline")
        return self.fig.text(x / self.w, 1 - y / self.h, s, **kwargs)

    def axes(self, left: float, top: float, width: float, height: float):
        """An axes whose box starts `left` and `top` points from the figure's top left corner."""
        ax = self.fig.add_axes((left / self.w, 1 - (top + height) / self.h, width / self.w, height / self.h))
        ax.set_facecolor(self.t["surface"])

        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

        ax.tick_params(length=3, width=0.8)
        return ax

    def anchor(self, x: float, y: float) -> tuple[float, float]:
        return x / self.w, 1 - y / self.h

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(path, format="svg", metadata={"Date": None})


def _log_limits(values: list[float], pad: float = 1.6) -> tuple[float, float]:
    return min(values) / pad, max(values) * pad


def _log_ticks(ax, lo: float, hi: float, axis: str = "x", fmt=lambda v: f"{v:g}") -> None:
    """Ticks at each power of ten between lo and hi, labeled with `fmt`."""
    m = _mpl()
    ticks = [10.0 ** e for e in range(math.floor(math.log10(lo)), math.ceil(math.log10(hi)) + 1)]
    ticks = [v for v in ticks if lo <= v <= hi]
    target = ax.xaxis if axis == "x" else ax.yaxis
    target.set_major_locator(m.FixedLocator(ticks))
    target.set_minor_locator(m.NullLocator())
    (ax.set_xticklabels if axis == "x" else ax.set_yticklabels)([fmt(v) for v in ticks])


def _grid(ax, t: dict, axis: str = "x") -> None:
    ax.grid(True, axis=axis, color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def _dot(ax, x: float, y: float, color: str, t: dict, size: float, hollow: bool = False, z: float = 3) -> None:
    """A dot at (x, y) in `color` with a ring of the surface around it, or a ring of `color` when it is hollow."""
    face, edge, width = (t["surface"], color, 1.6) if hollow else (color, t["surface"], 1.75)
    ax.plot([x], [y], marker="o", markersize=size, markerfacecolor=face, markeredgecolor=edge, markeredgewidth=width,
            linestyle="none", zorder=z)


def _row_detail(r: dict) -> str:
    parts = [fmt_setting(r)]

    if r.get("recall10_warm") is not None:
        parts.append(f"recall {r['recall10_warm']:.3f}")

    threads = f", {r['threads']} threads" if (r.get("threads") or 1) > 1 else ""
    parts.append(f"{r['trials']} trials{threads}")
    return " · ".join(parts)


def draw_cold_matched(data: dict, chart: dict, t: dict, path: Path) -> None:
    m = _mpl()
    rows = chart["rows"]
    measured = [r for r in rows if not r["modeled"]]
    modeled = [r for r in rows if r["modeled"]]
    slots = measured + ([None] if modeled else []) + modeled
    row_h, left, plot_w = 31.0, 300.0, 290.0
    dataset = chart["dataset"] if isinstance(chart["dataset"], str) else ", ".join(chart["dataset"])
    recalls = [r["recall10_warm"] for r in measured if r.get("recall10_warm") is not None]
    reach = (f"Each row runs at a setting picked to reach recall@10 {chart['target_recall10']:.2f} warm, and its "
             f"warm recall is {min(recalls):.3f} to {max(recalls):.3f} over "
             f"{fmt_count([r.get('recall10_warm_queries') for r in measured])} queries. ") if recalls else ""
    page = Page(t, f"{chart['title']} (preliminary)", [
        f"{millions(chart['n'])} vectors of {dataset} ({DATASETS.get(dataset, dataset)}), on one "
        f"{data['machine'].get('chip', 'machine')} in a quiet window. Each dot is the p50 of "
        f"{fmt_range([r['trials'] for r in measured])} trials, and its line runs to the p99. {reach}Log scale."
    ], 34 + row_h * len(slots) + 42, [chart["footer"]])
    top = page.top + 34
    ax = page.axes(left, top, plot_w, row_h * len(slots))
    lo = 10 ** math.floor(math.log10(min(r["p50_us"] for r in rows) / 1.2))
    hi = 10 ** math.ceil(math.log10(max(r["p99_us"] for r in rows)))
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(len(slots) - 0.5, -0.5)
    ax.yaxis.set_major_locator(m.NullLocator())
    ax.spines["left"].set_visible(False)
    _log_ticks(ax, lo, hi, fmt=lambda v: f"{v / 1000:,g}")
    ax.set_xlabel("Cold first query (ms), log scale", labelpad=6)
    _grid(ax, t)
    rowpos = m.blended(ax.transAxes, ax.transData)

    for i, r in enumerate(slots):
        if r is None:
            ax.axhline(i, color=t["axis"], linewidth=0.8)
            ax.text(-0.02, i, "Modeled, reads only", transform=rowpos, ha="right", va="center", size=8.5,
                    color=t["ink2"], style="italic")
            continue

        ours = r["group"] == "coldivf"
        color = t["coldivf"] if ours else t["rest"]
        weight = "bold" if ours else "normal"
        ax.axhline(i, color=t["grid"], linewidth=0.6, zorder=0)
        ax.plot([r["p50_us"], r["p99_us"]], [i, i], color=color, linewidth=1.5, zorder=2, solid_capstyle="butt")
        ax.plot([r["p99_us"]], [i], marker="|", markersize=9, markeredgewidth=1.5, color=color, zorder=2)
        _dot(ax, r["p50_us"], i, color, t, 10, hollow=r["modeled"])
        ax.text(-0.02, i - 0.1, r["label"], transform=rowpos, ha="right", va="bottom", size=9.5, color=t["ink"],
                weight=weight)
        ax.text(-0.02, i + 0.1, _row_detail(r), transform=rowpos, ha="right", va="top", size=8, color=t["ink2"])
        ax.text(1 + 14 / plot_w, i, fmt_ms(r["p50_us"]), transform=rowpos, ha="left", va="center", size=9.5,
                color=t["ink"], weight=weight)
        ax.text(1 + 76 / plot_w, i, f"p99 {fmt_ms(r['p99_us'])}", transform=rowpos, ha="left", va="center",
                size=8.5, color=t["ink2"])

    dot = {"marker": "o", "markersize": 8, "linestyle": "none"}
    ours_label, rest_label = chart["legend"]
    handles = [m.Line2D([], [], markerfacecolor=t["coldivf"], markeredgecolor=t["surface"], label=ours_label, **dot),
               m.Line2D([], [], markerfacecolor=t["rest"], markeredgecolor=t["surface"], label=rest_label, **dot)]

    if modeled:
        handles.append(m.Line2D([], [], markerfacecolor=t["surface"], markeredgecolor=t["rest"], markeredgewidth=1.6,
                                label="Modeled", **dot))

    handles.append(m.Line2D([], [], color=t["rest"], marker="|", markersize=9, markeredgewidth=1.5, linewidth=1.5,
                            label="Line to the p99"))
    page.fig.legend(handles=handles, loc="upper left", ncol=4, handlelength=1.6, columnspacing=1.8,
                    borderaxespad=0, borderpad=0, bbox_to_anchor=page.anchor(MARGIN, page.top + 4))
    page.save(path)


def draw_warm_leaderboard(data: dict, chart: dict, t: dict, path: Path) -> None:
    m = _mpl()
    boards, curves = chart["boards"], chart["curves"]
    points = [p for c in curves for p in c["points"]]
    calls = [p["overhead_us"] for c in curves for p in c["points"] if p.get("overhead_us")]
    busy = busy_note([c["machine_state"] for c in curves], [c["system"] == "coldivf" for c in curves])
    foot = []

    if chart["left_out"]:
        foot.append(f"Left off, since no setting of its grid reaches recall@10 {boards[0]['recall10']:.2f}: "
                    + ", ".join(f"{g['label']}, whose best is {g['best_recall10']:.3f}" for g in chart["left_out"])
                    + ".")

    if calls:
        foot.append(f"{binding_note(curves)} One call on a 1,000-vector index takes them {fmt_us(min(calls))} to "
                    f"{fmt_us(max(calls))}.")

    if busy:
        foot.append(busy)

    longest = max(len(b["rows"]) for b in boards)
    every = [r["p50_us"] for b in boards for r in b["rows"]]
    lo, hi = min(every) / 1.5, max(every) * 1.5
    row_h, label_w, plot_w, value_w, gap = 16.0, 100.0, 88.0, 42.0, 6.0
    panel_w = label_w + plot_w + value_w
    short = [wrap(f"Not reached in its grid: {', '.join(r['label'] for r in b['not_reached'])}.", 7.5, panel_w - 4)
             if b["not_reached"] else [] for b in boards]
    grid = [len(c["points"]) for c in curves if len(c["points"]) > 1]
    page = Page(t, f"{chart['title']} at {millions(chart['n'])} vectors (preliminary)", [
        f"First {millions(chart['n'])} rows of {chart['dataset']} ({DATASETS.get(chart['dataset'], '')}). Each panel "
        "ranks the systems by warm p50 at one recall@10, read off each system's grid of "
        f"{fmt_range(grid)} settings between the two around it, linear in the log of the latency. A system whose "
        "every setting is above a target shows its fastest, marked *. Smoke grids at 1 thread over "
        f"{fmt_range([p['queries'] for p in points])} queries, {fmt_range([p['passes'] for p in points])} timed "
        "passes each, with every index read into memory first. The three panels share one log scale."
    ], 30 + row_h * longest + 36 + 10 * max(len(lines) for lines in short) + 4, [" ".join(foot)])

    for b, board in enumerate(boards):
        x0 = MARGIN + b * (panel_w + gap)
        ax = page.axes(x0 + label_w, page.top + 30, plot_w, row_h * longest)
        ax.set_xscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(longest - 0.5, -0.5)
        ax.yaxis.set_major_locator(m.NullLocator())
        ax.spines["left"].set_visible(False)
        _log_ticks(ax, lo, hi, fmt=lambda v: f"{v / 1000:g}")
        ax.tick_params(axis="x", labelsize=7)
        ax.set_xlabel("ms, log scale", size=7.5, labelpad=3)
        _grid(ax, t)
        page.text(x0, page.top + 18, f"Recall@10 {board['recall10']:.2f}", size=10, weight="bold", color=t["ink"])
        pos = m.blended(ax.transAxes, ax.transData)

        for i, r in enumerate(board["rows"]):
            ours = r["system"] == "coldivf"
            color = t["coldivf"] if ours else t["rest"]
            ink = t["coldivf"] if ours else t["ink"]
            weight = "bold" if ours else "normal"
            ax.plot([lo, r["p50_us"]], [i, i], color=color, linewidth=2 if ours else 1.2, zorder=2,
                    solid_capstyle="butt")
            ax.plot([r["p50_us"]], [i], marker="o", markersize=6 if ours else 4.5, color=color,
                    markeredgecolor=t["surface"], markeredgewidth=0.8, zorder=3)
            ax.text(-label_w / plot_w, i, f"{i + 1}", transform=pos, ha="left", va="center", size=7.5,
                    color=t["ink2"])
            ax.text(-0.04, i, r["label"], transform=pos, ha="right", va="center", size=8, color=ink, weight=weight)
            ax.text(1 + value_w / plot_w, i, fmt_us(r["p50_us"]) + ("" if r["between"] else "*"), transform=pos,
                    ha="right", va="center", size=8, color=ink, weight=weight)

        for k, line in enumerate(short[b]):
            page.text(x0, page.top + 30 + row_h * longest + 40 + 10 * k, line, size=7.5, color=t["ink2"])

    page.save(path)


def _place_labels(ax, points: list[tuple], obstacles: list[tuple]) -> None:
    """Puts each point's label beside it where it overlaps no other label, point or whisker, trying 8 spots in turn,
    and takes the spot with the least overlap when none is clear. `obstacles` are boxes in display space."""
    renderer = ax.figure.canvas.get_renderer()
    spots = [(7, 0, "left", "center"), (-7, 0, "right", "center"), (0, 8, "center", "bottom"),
             (0, -8, "center", "top"), (6, 6, "left", "bottom"), (6, -6, "left", "top"), (-6, 6, "right", "bottom"),
             (-6, -6, "right", "top")]
    frame = ax.get_window_extent(renderer)
    taken = list(obstacles)

    def overlap(a, b):
        w = min(a[2], b[2]) - max(a[0], b[0])
        h = min(a[3], b[3]) - max(a[1], b[1])
        return w * h if w > 0 and h > 0 else 0.0

    for x, y, label, color, weight in points:
        dx, dy = ax.transData.transform((x, y))
        own = (dx - 6, dy - 6, dx + 6, dy + 6)
        best = None

        for ox, oy, ha, va in spots:
            text = ax.annotate(label, (x, y), xytext=(ox, oy), textcoords="offset points", ha=ha, va=va, size=8.5,
                               color=color, weight=weight)
            e = text.get_window_extent(renderer)
            area = (e.x0 - 1.5, e.y0 - 1.5, e.x1 + 1.5, e.y1 + 1.5)
            cost = sum(overlap(area, b) for b in taken if b != own)
            cost += 1e6 if area[0] < frame.x0 or area[2] > frame.x1 or area[1] < frame.y0 or area[3] > frame.y1 else 0

            if best is None or cost < best[0]:
                if best is not None:
                    best[1].remove()

                best = (cost, text, area)
            else:
                text.remove()

            if cost == 0:
                break

        taken.append(best[2])


def draw_cold_smoke(data: dict, chart: dict, t: dict, path: Path) -> None:
    points = [p for p in chart["points"] if p["recall10_warm"] is not None]
    busy = busy_note([p["machine_state"] for p in points], [p["system"] == "coldivf" for p in points])
    threads = sorted({p["threads"] for p in points if p["threads"]})
    foot = []

    if len(threads) > 1:
        many = [p["label"].split(" (")[0] for p in points if p["threads"] == max(threads)]
        foot.append(f"{', '.join(many)} ran on {max(threads)} threads, its default pool, and the rest on "
                    f"{min(threads)}.")

    if busy:
        foot.append(busy)

    page = Page(t, f"{chart['title']} (preliminary)", [
        f"First {millions(chart['n'])} rows of {chart['dataset']} ({DATASETS.get(chart['dataset'], '')}). Each trial "
        f"is a fresh process with the index evicted, {fmt_range([p['trials'] for p in points])} trials per system, "
        "on a prewarmed core. Recall isn't matched. Each system runs one setting near its defaults, and the recall is "
        f"that setting's warm recall@10 over {fmt_range([p['recall10_warm_queries'] for p in points])} tune queries. "
        "Dots are the median, whiskers span the trials. Log scale."
    ], 330, [" ".join(foot)])
    left, width, height = 70.0, WIDTH - 70 - 30, 284.0
    ax = page.axes(left, page.top + 8, width, height)
    xs = [p["recall10_warm"] for p in points]
    lo, hi = _log_limits([v for p in points for v in (p["min_us"], p["max_us"])], 1.5)
    ax.set_yscale("log")
    ax.set_ylim(lo, hi)
    ax.set_xlim(math.floor(min(xs) * 50) / 50 - 0.01, 1.01)
    _log_ticks(ax, lo, hi, axis="y", fmt=lambda v: f"{v / 1000:,g}")
    _grid(ax, t, "both")
    ax.set_xlabel("Recall@10 of the setting, warm", labelpad=6)
    ax.set_ylabel("Cold first query (ms), log scale", labelpad=6)
    labels, obstacles = [], []

    for p in sorted(points, key=lambda p: p["system"] == "coldivf"):
        ours = p["system"] == "coldivf"
        color = t["coldivf"] if ours else t["rest"]
        ax.plot([p["recall10_warm"]] * 2, [p["min_us"], p["max_us"]], color=color, linewidth=1.2, zorder=2,
                solid_capstyle="butt")
        ax.plot([p["recall10_warm"]], [p["p50_us"]], marker="o", markersize=9 if ours else 7.5, color=color,
                markeredgecolor=t["surface"], markeredgewidth=2, zorder=4 if ours else 3, linestyle="none")
        (x, y0), (_, y1) = ax.transData.transform([(p["recall10_warm"], p["min_us"]), (p["recall10_warm"],
                                                                                       p["max_us"])])
        obstacles += [(x - 6, y0 - 6, x + 6, y1 + 6)]
        labels.append((p["recall10_warm"], p["p50_us"], p["label"], t["ink"] if ours else t["ink2"],
                       "bold" if ours else "normal"))

    _place_labels(ax, sorted(labels, key=lambda q: q[4] != "bold"), obstacles)
    page.save(path)


def draw_memory(data: dict, chart: dict, t: dict, path: Path) -> None:
    m = _mpl()
    rows = chart["rows"]
    row_h, label_w, plot_w, gap_x = 14.5, 104.0, 196.0, 70.0
    panel_h = row_h * len(rows)
    exact = [r["label"] for r in rows if (r["disk"] or {}).get("exact_f32_stored")]
    page = Page(t, f"{chart['title']} (preliminary)", [
        f"First {millions(chart['n'])} rows of {chart['dataset']}, from the smoke runs. Cold is the median "
        f"of {fmt_range([r['cold']['trials'] for r in rows])} trials. Warm is at the cold setting, after every index "
        f"file was read. Warm process memory comes {process_note(rows)}. Builds ran on "
        f"{fmt_range([(r['build'] or {}).get('threads') for r in rows])} threads."
    ], 2 * (28 + panel_h + 30) + 6, [f"The indexes that keep every float32 row on disk: {', '.join(exact)}."]
                                     if exact else [])
    panels = [
        ("Added by the open and the first cold query", "log scale",
         lambda r: ((r["cold"]["footprint_delta"] or {}).get("median"), None), True, fmt_bytes),
        ("Warm: process memory plus cached index pages", "",
         lambda r: ((r["warm"] or {}).get("process_bytes"), (r["warm"] or {}).get("resident_file_bytes")), False,
         fmt_bytes),
        ("Peak footprint while building", "", lambda r: ((r["build"] or {}).get("peak_footprint"), None), False,
         fmt_bytes),
        ("Index bytes per vector on disk", "", lambda r: ((r["disk"] or {}).get("bytes_per_vector"), None), False,
         lambda v: f"{v:,.0f} B"),
    ]

    for i, (title, note, value, log, fmt) in enumerate(panels):
        r_i, k = divmod(i, 2)
        left = MARGIN + label_w + k * (label_w + plot_w + gap_x)
        top = page.top + 26 + r_i * (panel_h + 58)
        ax = page.axes(left, top, plot_w, panel_h)
        # A log axis can't show 0, so a log panel leaves out what measured nothing.
        have = sorted(((r, *value(r)) for r in rows if value(r)[0] is not None and (value(r)[0] > 0 or not log)),
                      key=lambda q: q[1] + (q[2] or 0))
        totals = [a + (b or 0) for _, a, b in have]

        if not have or max(totals) <= 0:
            ax.axis("off")
            ax.text(0, 1, f"{title}: no lines", transform=ax.transAxes, size=9, color=t["ink2"], va="top")
            continue

        ax.set_ylim(len(have) - 0.5, -0.5)
        ax.yaxis.set_major_locator(m.FixedLocator(range(len(have))))
        ax.set_yticklabels([r["label"] for r, _, _ in have])
        ax.tick_params(axis="y", length=0, pad=5)
        ax.spines["left"].set_visible(False)

        for tick, (r, _, _) in zip(ax.get_yticklabels(), have):
            ours = r["system"] == "coldivf"
            tick.set_color(t["ink"] if ours else t["ink2"])
            tick.set_fontweight("bold" if ours else "normal")
            tick.set_fontsize(8.5)

        if log:
            ax.set_xscale("log")
            lo = 10 ** math.floor(math.log10(min(totals)))
            hi = 10 ** math.ceil(math.log10(max(totals)))
            ax.set_xlim(lo, hi)
            _log_ticks(ax, lo, hi, fmt=fmt)
        else:
            ax.set_xlim(0, max(totals) * 1.25)
            ax.xaxis.set_major_locator(m.MaxNLocator(4))
            ax.xaxis.set_major_formatter(lambda v, _, fmt=fmt: fmt(v) if v else "0")

        # A bar's length only means something from 0, so the log panel marks each value with a dot on a row line.
        _grid(ax, t, "both" if log else "x")

        for j, (r, a, b) in enumerate(have):
            ours = r["system"] == "coldivf"
            solid, tint = (t["coldivf"], t["coldivf_tint"]) if ours else (t["rest"], t["rest_tint"])

            if log:
                _dot(ax, a, j, solid, t, 7)
            elif b:
                ax.add_artist(m.Bar(0.0, a, j, 9, radius=0, facecolor=solid, edgecolor="none"))
                ax.add_artist(m.Bar(a, a + b, j, 9, gap=2, facecolor=tint, edgecolor="none"))
            else:
                ax.add_artist(m.Bar(0.0, a, j, 9, facecolor=solid, edgecolor="none"))

            ax.annotate(f"{fmt(a)} + {fmt(b)}" if b else fmt(a), (a + (b or 0), j), xytext=(6 if log else 4, 0),
                        textcoords="offset points", va="center", size=7.5, color=t["ink"] if ours else t["ink2"],
                        annotation_clip=False)

        ax.text(-label_w / plot_w, 1 + 12 / panel_h, title, transform=ax.transAxes, size=9.5, weight="bold",
                color=t["ink"], va="bottom")

        if note:
            ax.text(1, 1 + 12 / panel_h, note, transform=ax.transAxes, size=8, color=t["ink2"], va="bottom",
                    ha="right")

        if i == 1:
            handles = [m.Patch(facecolor=t["rest"], label="process memory"),
                       m.Patch(facecolor=t["rest_tint"], label="cached index pages")]
            ax.legend(handles=handles, loc="upper right", handlelength=1.2, borderaxespad=0.3)

    page.save(path)


DRAW = {"cold-coldivf": draw_cold_matched, "cold-head-to-head": draw_cold_matched,
        "warm-leaderboard": draw_warm_leaderboard, "cold-smoke": draw_cold_smoke, "memory": draw_memory}


def draw(data: dict, out: Path) -> list[Path]:
    """Every chart in both themes, as <id>.svg for a light page and <id>-dark.svg for a dark one."""
    m = _mpl()
    written = []

    for name, t in THEMES.items():
        with m.matplotlib.rc_context(_rc(t)):
            for chart in data["charts"]:
                if not (chart.get("rows") or chart.get("points") or chart.get("curves")):
                    print(f"no lines for the {chart['id']} chart, so it isn't drawn", file=sys.stderr)
                    continue

                path = out / chart["files"][name]
                DRAW[chart["id"]](data, chart, t, path)
                written.append(path)

    return written


def publish(data: dict, out: Path, charts: bool = True) -> list[Path]:
    """Draws every chart, unless `charts` is false, and writes chart-data.json, all into a scratch folder in `out`,
    then moves each file into place. A drawing that fails leaves `out` as it was. It returns the files it wrote."""
    version = _mpl().matplotlib.__version__ if charts else None

    if version is not None and version != MATPLOTLIB:
        print(f"drawing with matplotlib {version}, while the committed charts were drawn with {MATPLOTLIB}, so the "
              "SVGs may differ a little", file=sys.stderr)

    data["drawn_with"] = {"matplotlib": version} if version else None
    out.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=out, prefix=".charts-") as scratch:
        scratch = Path(scratch)
        drawn = draw(data, scratch) if charts else []

        (scratch / "chart-data.json").write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
        written = []

        for path in drawn + [scratch / "chart-data.json"]:
            os.replace(path, out / path.name)
            written.append(out / path.name)

    return written


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--suite", type=Path, default=SUITE, help="the smoke suite, whose dataset, n and systems it reads")
    p.add_argument("--quiet", type=Path, default=QUIET, help="the quiet-window run's results folder")
    p.add_argument("--out", type=Path, default=OUT, help="where chart-data.json and the charts go")
    p.add_argument("--repo", type=Path, default=REPO, help="the root that paths in chart-data.json are relative to")
    p.add_argument("--data-only", action="store_true", help="write chart-data.json without drawing")
    args = p.parse_args(argv)
    data = chart_data(args.suite, args.quiet, args.repo)

    for chart in data["charts"]:
        chart["files"] = {name: f"{chart['id']}.svg" if name == "light" else f"{chart['id']}-{name}.svg"
                          for name in THEMES}

    if data["missing_systems"]:
        print(f"no smoke lines for {', '.join(data['missing_systems'])}", file=sys.stderr)

    for gone in next(c for c in data["charts"] if c["id"] == "cold-head-to-head")["left_out"]:
        print(f"the matched chart leaves out {gone}, since no cell reaches recall@10 {TARGET} warm", file=sys.stderr)

    try:
        publish(data, args.out, charts=not args.data_only)
    except ImportError as e:
        print(f"wrote nothing, since drawing needs matplotlib: {e}", file=sys.stderr)
        return 1

    if args.data_only:
        print("wrote only chart-data.json, so the charts may not match it until they are drawn", file=sys.stderr)

    print(headline_table(data, "cold-head-to-head"))
    print()
    print(headline_table(data, "cold-coldivf"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
