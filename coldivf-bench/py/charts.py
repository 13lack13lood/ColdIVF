"""Charts of ColdIVF against the other systems for the README, and the numbers behind them.

    python charts.py [--quiet DIR] [--h2h SCALE=SUITE.toml]... [--check SCALE=SUITE.toml]...
                     [--out DIR] [--repo DIR] [--data-only]

Three runs feed them. The head-to-heads at 1M and 10M vectors, results/bench/bench-1m for suites/bench-1m.toml and
results/bench/bench-10m for suites/bench-10m.toml, hold each system's build, tune grid, true-cold (C0) screen and its
picks for each recall target timed on the test split, which headtohead.py reads. Each scale's charts are named by it,
eg. cold-1m-95 and memory-10m, and a same-batch check of a scale, eg. suites/bench-1m-same-batch.toml, which times
ColdIVF's module and its closest rival in one batch, joins that scale's cold chart at 0.95. The quiet-window run in
results/quiet2 holds ColdIVF's cold trials on Cohere 1M in each setup, through the driver and through its own cold
binary, the controls that read its file another way, and its warm grid. charts.py picks the lines each chart plots and
writes chart-data.json, which holds every plotted number and every number the README's text cites. Each comes with
its source file, dataset, setting, trial count, machine state and the code that measured it. It then draws each chart
as scalable vector graphics (SVG) for a dark page, and prints the README's tables.

The quiet-window picks need only the standard library, and the head-to-head ones numpy too, through the driver. The
drawing needs matplotlib, and MATPLOTLIB below is the version that drew the committed charts. It draws into a scratch
folder and moves the files into place once every chart is drawn, so a drawing that fails changes nothing. --data-only
writes only chart-data.json.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tempfile
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

PY = Path(__file__).resolve().parent
REPO = PY.parents[1]
QUIET = REPO / "results" / "quiet2"
OUT = REPO / "docs" / "charts"

# The recall@10 the cold-coldivf chart compares ColdIVF's setups at, the target that pick.py picks settings for.
TARGET = 0.95

# The splits whose warm recall picks a matched cell, the tune split that pick.py picks on, else all the queries.
PICK_SPLITS = ("tune", "all")

# The matplotlib that drew the committed charts. Another version draws them too, but its SVGs differ a little.
MATPLOTLIB = "3.11.2"

DATASETS = {
    "cohere-768-cosine": "Cohere 1M, 768 dimensions, cosine, 1,000 queries",
    "cohere10m-768-cosine": "Cohere 10M, 768 dimensions, cosine, 1,000 queries",
}

# The names of ColdIVF's cells through the driver in the quiet run, by config.
CELLS = {"faiss-chained": "ColdIVF, through the driver"}

CPU_STATES = {"prewarm": "prewarmed core", "idle": "idle core"}

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
    """`value` with float noise cut to 7 significant digits, eg. 0.9005999999999982 becomes 0.9006. Seven keep every
    count the tables print to the unit, eg. 11,543.45 queries per second stays 11,543, not 11,544."""
    if isinstance(value, float):
        return float(f"{value:.7g}")

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


def cold_summary(lines: list[dict]) -> dict:
    """The first-query times of one cell's cold trials, with the setting and recall they share."""
    times = [line["total_us"] for line in lines]
    recall = [line["recall10"] for line in lines if line.get("recall10") is not None]
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
        "slowest_us": sorted(times)[-3:],
        "recall10_trials": mean(recall),
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


def matched_cells(cold: list[dict], warm_of) -> list[tuple[list[dict], dict, dict]]:
    """The cell at matched recall of each system, config, CPU state and thread count in `cold`. Of the C0 cells whose
    setting reaches TARGET warm, it keeps those with the most trials, the main cells a verdict needs, and takes the
    cheapest of them as pick.py does, the lowest warm p50 with ties to fewer bytes scanned. `warm_of(line, splits)`
    gives the warm line of a cell's setting from the first of `splits` that has one. Each chosen cell comes with the
    warm line that picked it and the one its row shows, which is on the split its trials searched when there is one.
    A group with no cell that reaches TARGET is left out."""
    groups = defaultdict(list)

    for key, cell in cell_groups(cold).items():
        system, config_id, variant, _, regime, cpu_state, threads = key
        split = cell[0].get("split")
        picked = warm_of(cell[0], PICK_SPLITS)
        shown = warm_of(cell[0], (split, "all") if split else ("all",))
        groups[(system, config_id, variant, regime, cpu_state, threads)].append((cell, picked, shown))

    chosen = []

    for cells in groups.values():
        reach = [c for c in cells if c[1] is not None and c[1]["recall10"] >= TARGET]

        if not reach:
            continue

        most = max(len(c[0]) for c in reach)
        chosen.append(min((c for c in reach if len(c[0]) == most),
                          key=lambda c: (c[1].get("p50_us") or math.inf, c[1].get("bytes_scanned") or 0.0,
                                         percentile([line["total_us"] for line in c[0]], 50))))

    return chosen


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


def cold_matched(quiet: Path, repo: Path) -> dict:
    """The chart of ColdIVF's cold first query at matched recall in each of its setups, from the quiet run: through the
    driver, through its own cold binary, and the controls that read its file another way."""
    idle = runner_idle((quiet / "summary.md").read_text() if (quiet / "summary.md").exists() else "")
    own = own_warm(read_lines(quiet / "b3-warm.jsonl"))
    driver_lines = [(path, line) for path in sorted(quiet.glob("driver/*/coldivf.jsonl")) for line in read_lines(path)]
    source = {id(line): path for path, line in driver_lines} | {id(line): quiet / "b3-warm.jsonl" for line in own}
    cold = [line for _, line in driver_lines if line.get("kind") == "cold" and line.get("status") == "ok"
            and line.get("regime") == "c0" and line.get("total_us") is not None]
    warm = [line for _, line in driver_lines if line.get("kind") == "warm"]

    def warm_of(line: dict, splits: tuple[str, ...]) -> dict | None:
        setting = line.get("search_params") or {}
        found = warm_at(warm, "coldivf", line.get("config_id"), setting, splits)
        return found if found is not None else warm_at(own, "coldivf", None, setting, splits)

    rows = []

    for cell, picked, shown in matched_cells(cold, warm_of):
        s = cold_summary(cell)
        state = CPU_STATES.get(s["cpu_state"], s["cpu_state"])
        label = CELLS.get(s["config_id"], f"ColdIVF {s['config_id']}")
        rows.append({
            "label": label if s["cpu_state"] in (None, "prewarm") else f"{label}, {state}",
            "group": "coldivf", "harness": "driver", "system": "coldivf",
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

    rows.sort(key=lambda r: r["p50_us"])
    datasets = sorted({r["dataset"] for r in rows if r["dataset"]})
    return {
        "id": "cold-coldivf",
        "title": "ColdIVF's cold first query in each setup, recall@10 near 0.95",
        "dataset": datasets[0] if len(datasets) == 1 else datasets,
        "n": rows[0]["n"] if rows else None,
        "unit": "microseconds",
        "target_recall10": TARGET,
        "legend": ["ColdIVF", "Controls, ColdIVF's file read another way"],
        "footer": "A prewarmed core spins for 150 ms before the query, and an idle core sleeps 200 ms first, so its "
                  "query can start on an efficiency core. The driver runs a fresh process per trial with the index "
                  "evicted, and the own harness is ColdIVF's cold binary. After a search of another index, the "
                  "process has first searched a 50K index of its own. The lazy map searches ColdIVF's file through a "
                  "memory map that faults pages in, at the control's setting rather than its cheapest one.",
        "notes": [
            f"Each row runs at a setting whose recall@10 reaches {TARGET:.2f} warm. Of those timed cold, it is the one "
            "with the most trials, and of those with as many, the cheapest warm, the lowest p50 and then the fewest "
            "bytes scanned, as pick.py ranks settings.",
            "Every row is ColdIVF's index, timed through the driver or ColdIVF's own cold binary, with the read path "
            "and the state before the query that its label names.",
            "Each p50_ratio is the row's p50 over ColdIVF's through the driver, and a control's over the read waves "
            "that its own job interleaved with it, at the same setting and CPU state.",
        ],
        "rows": rows,
    }


# The head-to-head suites, relative to the repo, whose results the head-to-head charts read, by the scale that names
# their charts, eg. cold-1m-95 and memory-10m, and the targets the cold and warm charts show.
H2H_SUITES = {
    "1m": Path("coldivf-bench") / "suites" / "bench-1m.toml",
    "10m": Path("coldivf-bench") / "suites" / "bench-10m.toml",
}
H2H_TARGETS = (0.95, 0.90)

# The rows of each dataset, so the charts of a suite that searches fewer of them say it takes the first rows.
DATASET_ROWS = {"cohere10m-768-cosine": 10_000_000}

# A short name for each of the head-to-head's systems, for charts whose labels have little room.
SHORT = {
    "ColdIVF": "ColdIVF", "ColdIVF, binaries": "ColdIVF binaries",
    "faiss IVF as shipped, 640 per list": "faiss IVF", "faiss IVF, parallel, 640 per list": "faiss parallel",
    "faiss IVF, batched reads, 640 per list": "faiss batched",
    "faiss IVF, on-disk lists, 640 per list": "faiss on-disk",
    "faiss IVF, HNSW quantizer, 640 per list": "faiss graph route",
    "faiss IVF, batched reads, 2,560 per list": "faiss batched 2560",
    "faiss IVF, on-disk lists, 2,560 per list": "faiss on-disk 2560",
    "faiss FastScan + SQ8": "faiss FastScan", "faiss HNSW32-SQ8": "faiss HNSW SQ8", "LanceDB IVF_RQ": "LanceDB IVF_RQ",
    "Paimon 4-bit": "Paimon 4-bit", "Paimon 8-bit": "Paimon 8-bit", "RaBitQ-Library IVF 5-bit": "RaBitQ IVF 5-bit",
    "RaBitQ-Library IVF 9-bit": "RaBitQ IVF 9-bit", "usearch HNSW i8": "usearch HNSW i8",
    "turbovec 4-bit": "turbovec", "turbovec 4-bit, calibrated": "turbovec calibrated",
}

# The short names of faiss IVF's setups by variant. A setup on faiss's own k-means, whose config id starts with vpl,
# adds its vectors per list, while those on ColdIVF's partition go without.
FAISS_SHORT = {"shipped": "faiss IVF", "parallel": "faiss parallel", "batched": "faiss batched",
               "ondisk": "faiss on-disk", "graph": "faiss graph route"}


def short_label(r: dict) -> str:
    """A head-to-head row's name for charts whose labels have little room, eg. "faiss batched 2560"."""
    if r["label"] in SHORT:
        return SHORT[r["label"]]

    if r["system"] == "faiss-ivf" and r.get("variant") in FAISS_SHORT:
        found = re.match(r"vpl(\d+)", r.get("config_id") or "")
        return FAISS_SHORT[r["variant"]] + (f" {found[1]}" if found else "")

    return r["label"].replace("RaBitQ-Library ", "RaBitQ ")


# The name of each system of the head-to-head, for a chart that shows it whatever configuration its row is.
SYSTEM_NAMES = {
    "coldivf": "ColdIVF", "faiss-ivf": "faiss IVF", "faiss-fastscan": "faiss FastScan", "faiss-hnsw": "faiss HNSW SQ8",
    "lancedb": "LanceDB IVF_RQ", "paimon": "Paimon", "rabitq-ivf": "RaBitQ IVF", "turbovec": "turbovec",
    "usearch": "usearch HNSW i8", "rabitq-hnsw": "RaBitQ HNSW", "qdrant-edge": "qdrant-edge", "diskann": "DiskANN",
    "astervec": "AsterVec", "symphonyqg": "SymphonyQG", "sqlite-vec": "sqlite-vec", "flat": "flat scan",
}

# How a warm line on 6 threads ran its queries side by side, by headtohead.py's name for it, in the order the
# footnotes of the queries per second (QPS) charts list them.
METHODS = ("batch call", "processes", "threads")


def listed(names: list[str]) -> str:
    """Names in words, eg. "a", "a and b" or "a, b and c"."""
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + f" and {names[-1]}"


def one_per_system(rows: list[dict], key) -> list[dict]:
    """One row per system, in the order of `rows`, so that a chart compares each system once. ColdIVF's is its Python
    module's, which runs as every other system runs through its binding, or its binaries' when the module has no row.
    Every other system's is the configuration with the lowest `key`."""
    module = any(r["system"] == "coldivf" and r.get("variant") == "python" for r in rows)
    best = {}

    for r in rows:
        if r["system"] == "coldivf" and module and r.get("variant") != "python":
            continue

        if r["system"] not in best or key(r) < key(best[r["system"]]):
            best[r["system"]] = r

    kept = {id(r) for r in best.values()}
    return [r for r in rows if id(r) in kept]


def every_setting(setting: dict) -> str:
    """A setting in words with every search parameter, the ones its whole grid shares too, by the names the tables use,
    eg. "nprobe 48, k_factor 4, qb 4, quantizer_ef 128", or "none" for a system without search settings."""
    import headtohead

    return ", ".join(f"{headtohead.SETTING_NAMES.get(k, k)} {v}" for k, v in setting.items()) or "none"


def h2h_cold_row(r: dict, folder: Path, repo: Path) -> dict:
    """One system's row of a head-to-head cold chart, in the shape of the cold-coldivf chart's rows, with every search
    parameter of its setting, as the warm and QPS charts name it."""
    trials, against = r["_trials"], r.get("against") or {}
    first = trials[0] if trials else {}
    return {
        "label": r["label"], "short": short_label(r),
        "group": "coldivf" if r["system"] == "coldivf" else "system", "harness": "driver", "system": r["system"],
        "variant": r["variant"], "config_id": r["config_id"], "setting": r["setting"], "dataset": first.get("dataset"),
        "n": first.get("n"), "regime": "c0", "cpu_state": first.get("cpu_state"), "threads": first.get("threads"),
        "setting_text": every_setting(r["setting"]), "trials": r["trials"], "queries": r["queries"],
        "p50_us": r["c0_p50_us"], "p95_us": r["c0_p95_us"],
        "p99_us": r["c0_p99_us"], "open_p50_us": r["open_p50_us"], "recall10_warm": r["recall10"],
        "recall10_warm_queries": r["warm_queries"], "recall10_warm_split": "test",
        "c0_exact_p50_us": (r.get("c0_exact") or {}).get("p50_us"), "p50_ratio": against.get("c0_p50", 1.0),
        "paired_ratio": against.get("c0_paired"), "exact_ratio": against.get("c0_exact"), "modeled": False,
        "source": rel(folder / f"{r['system']}.jsonl", repo), "machine_state": driver_state(trials),
        "code": code_of(trials, repo),
    }


def h2h_warm_row(r: dict, folder: Path, repo: Path) -> dict:
    """One system's row of a head-to-head warm chart, in the shape of the cold charts' rows: the p50 and p99 of its
    warm line at 1 thread on the test split, with every search parameter of its setting."""
    lines = [x for x in r["_warm"] if x.get("threads") == 1]
    first = lines[0] if lines else {}
    return {
        "label": r["label"], "short": short_label(r),
        "group": "coldivf" if r["system"] == "coldivf" else "system", "harness": "driver", "system": r["system"],
        "variant": r["variant"], "config_id": r["config_id"], "setting": r["setting"], "dataset": first.get("dataset"),
        "n": first.get("n"), "mode": r["warm_mode"], "threads": 1, "setting_text": every_setting(r["setting"]),
        "queries": r["warm_queries"], "passes": r["warm_passes"], "p50_us": r["warm_p50_us"],
        "p99_us": r["warm_p99_us"], "recall10_warm": r["recall10"], "recall10_warm_queries": r["warm_queries"],
        "recall10_warm_split": "test", "p50_ratio": (r.get("against") or {}).get("warm_p50", 1.0),
        "invalid": [t for t in r["warm_invalid"] if t == 1], "source": rel(folder / f"{r['system']}.jsonl", repo),
        "machine_state": driver_state(lines), "code": code_of(lines, repo),
    }


def h2h_qps_rows(entry: dict, folder: Path, repo: Path) -> list[dict]:
    """A head-to-head QPS panel's rows: each system's queries per second on 6 threads at its point for the target, on
    the test split, with every search parameter of its setting and how its queries ran side by side."""
    rows = []

    for r in entry["rows"]:
        if r["qps"].get(6) is not None:
            lines = [x for x in r["_warm"] if x.get("threads") == 6]
            rows.append({"system": r["system"], "variant": r["variant"], "label": short_label(r),
                         "full_label": r["label"], "config_id": r["config_id"], "qps": r["qps"][6],
                         "setting": r["setting"], "recall10": r["recall10"], "mode": r["warm_mode"],
                         "method": (r.get("qps_method") or {}).get(6),
                         "invalid": [t for t in r["warm_invalid"] if t == 6],
                         "setting_text": every_setting(r["setting"]),
                         "source": rel(folder / f"{r['system']}.jsonl", repo), "machine_state": driver_state(lines),
                         "code": code_of(lines, repo)})

    return sorted(rows, key=lambda x: x["qps"], reverse=True)


def h2h_memory_row(r: dict, target: float | None, folder: Path, repo: Path) -> dict:
    """One system's row of a head-to-head memory chart."""
    return {
        "system": r["system"], "variant": r["variant"], "label": short_label(r),
        "full_label": r["label"], "target_recall10": target, "source": rel(folder / f"{r['system']}.jsonl", repo),
        "setting_text": r["setting_text"], "recall10": r["recall10"],
        "cold": {"setting": r["setting"], "trials": r["trials"], "process": r["cold_process"], "pages": r["cold_pages"],
                 "total": r["cold_total"]},
        "warm": {"setting": r["setting"], "mode": r["warm_mode"], "process_bytes": r["warm_footprint"],
                 "process_source": "footprint, the whole process", "resident_file_bytes": r["warm_cached"]},
        "build": {"peak_footprint": r["build_peak"], "seconds": r["build_s"],
                  "cluster_seconds": r.get("build_cluster_s"), "shared_file": r["shared_file"],
                  "shared_partition": r["shared_partition"], "malloc_cache": r.get("build_malloc_cache"),
                  "stored_vector_bytes": r.get("stored_vector_bytes")},
        "disk": {"index_bytes": r["index_bytes"], "n": r["_trials"][0].get("n") if r["_trials"] else None,
                 "bytes_per_vector": r["bytes_per_vector"], "exact_f32_stored": None},
        "machine_state": driver_state(r["_trials"] + r["_warm"]), "code": code_of(r["_trials"] + r["_warm"], repo),
    }


def method_note(boards: list[dict]) -> str:
    """How the systems of a QPS chart ran their queries side by side on 6 threads, by name."""
    by = defaultdict(list)

    for b in boards:
        for r in b["rows"]:
            if r["label"] not in by[r.get("method")]:
                by[r.get("method")].append(r["label"])

    def verb(names: list[str]) -> str:
        return "searches" if len(names) == 1 else "search"

    parts = []

    if by.get("batch call"):
        parts.append(f"{listed(by['batch call'])} {verb(by['batch call'])} through a call for a batch of queries that "
                     "releases Python's lock.")

    rest = [(by[m], "processes" if m == "processes" else "threads in one process") for m in METHODS[1:] if by.get(m)]

    if rest:
        text = f"{listed(rest[0][0])} {verb(rest[0][0])} from 6 {rest[0][1]}"
        parts.append(text + (f", and {listed(rest[1][0])} from 6 {rest[1][1]}." if len(rest) > 1 else "."))

    if "ColdIVF" in by.get("threads", []):
        parts.append("ColdIVF's module has no batch call.")

    return " ".join(parts)


def warm_reads_note(names_by_mode: dict[str, list[str]]) -> str:
    """The warm chart's sentences on the systems whose warm mode leaves part of the index on disk, by mode: LanceDB's
    "index", which caches only its index files, and DiskANN's "pq", which keeps its product quantization (PQ) codes in
    memory and reads every sector from the disk, for an index larger than memory."""
    parts = []

    for mode, names in names_by_mode.items():
        one = len(names) == 1
        its = "its" if one else "their"

        if mode == "index":
            parts.append(f"{listed(names)} {'caches' if one else 'cache'} only {its} index files, so {its} rows are "
                         f"read from {its} table.")
        elif mode == "pq":
            parts.append(f"{listed(names)} {'keeps' if one else 'keep'} only {its} PQ codes in memory and "
                         f"{'reads' if one else 'read'} every sector from the disk, since {its} index doesn't fit in "
                         "memory.")
        else:
            parts.append(f"{listed(names)} {'runs' if one else 'run'} in warm mode {mode}.")

    return " ".join(parts)


def warm_left_note(rows: list[dict]) -> str:
    """The memory chart's sentences on what each row's warm mode leaves on disk, eg. LanceDB's table."""
    parts = []

    for r in rows:
        name, mode = r["full_label"].split(" ")[0], r["warm"]["mode"]

        if mode == "index" and r["build"]["stored_vector_bytes"]:
            parts.append(f"{name}'s warm runs cache only its index files and leave its "
                         f"{fmt_bytes(r['build']['stored_vector_bytes'])} table on disk.")
        elif mode == "pq" and r["disk"]["index_bytes"]:
            parts.append(f"{name}'s warm runs keep only its PQ codes in memory and read the sectors of its "
                         f"{fmt_bytes(r['disk']['index_bytes'])} index from the disk.")

    return " ".join(parts)


def malloc_note(rows: list[dict]) -> str:
    """Which builds of a memory chart ran with macOS malloc's cache of freed large blocks off, which keeps freed blocks
    out of their peaks, and which with it on."""
    off = [r["label"] for r in rows if r["build"]["peak_footprint"] is not None and r["build"]["malloc_cache"] == "off"]
    on = [r["label"] for r in rows if r["build"]["peak_footprint"] is not None and r["build"]["malloc_cache"] == "on"]

    if off and not on:
        return ("Every build ran with macOS malloc's cache of freed large blocks off, so no build peak holds freed "
                "memory.")

    if off and on:
        whose = f"{off[0]}'s build" if len(off) == 1 else f"The builds of {listed(off)}"
        return (f"{whose} ran with macOS malloc's cache of freed large blocks off, and the other builds with it on, "
                "which keeps freed blocks inside their peaks.")

    return ""


def h2h_charts(suite_path: Path, repo: Path, scale: str = "10m") -> tuple[list[dict], dict]:
    """A head-to-head's charts from its suite's results, by headtohead.py's picks: the cold first query and the warm
    query at 1 thread at recall@10 0.95 and 0.90, queries per second at each target, and memory, named by `scale`, eg.
    cold-1m-95 and warm-1m-90. It returns them with the machine their lines recorded."""
    import driver
    import headtohead

    suite = driver.load_suite(suite_path)
    folder = suite.results_root / suite.phase
    res = headtohead.Results(folder)
    data = headtohead.collect(suite, res)
    # ColdIVF shows once, through its Python module as every other system runs through its binding. Its binaries'
    # rows, a control for comparing ColdIVF with itself, stay in the tables.
    ours = headtohead.find(suite.systems, headtohead.REFERENCE)

    def shown(r: dict) -> bool:
        return ours is None or r["system"] != ours.name or r.get("variant") == ours.variant

    for entry in [*data["targets"].values(), {"rows": data["hand"]}]:
        entry["rows"][:] = [r for r in entry["rows"] if shown(r)]
        entry.get("not_reached", [])[:] = [m for m in entry.get("not_reached", []) if shown(m)]

    machine = machine_record(res.lines)
    chip = machine.get("chip", "machine")
    n, dataset = suite.n, suite.dataset
    whole = DATASET_ROWS.get(dataset)
    taken = f"The first {millions(n)} rows" if whole and n < whole else f"{millions(n)} vectors"
    about = (f"{taken} of {dataset} ({DATASETS.get(dataset, dataset)}), on one {chip} with "
             f"{(machine.get('memory_bytes') or 0) / 2 ** 30:.0f} GB of memory.")
    charts = []
    systems = {r["system"] for t in headtohead.TARGETS for r in data["targets"][t]["rows"]}
    systems |= {m["system"] for t in headtohead.TARGETS for m in data["targets"][t]["not_reached"]}

    def missing(rows: list[dict]) -> list[dict]:
        """The systems with no row in a chart, since no configuration of theirs reaches its target."""
        have = {r["system"] for r in rows}
        return [{"system": s, "label": SYSTEM_NAMES.get(s, s)}
                for s in sorted(systems - have, key=lambda s: SYSTEM_NAMES.get(s, s))]

    for t in H2H_TARGETS:
        entry = data["targets"][t]
        rows = one_per_system([h2h_cold_row(r, folder, repo) for r in entry["rows"] if r["c0_p50_us"] is not None],
                              lambda r: r["p50_us"])
        recalls = [r["recall10_warm"] for r in rows if r["recall10_warm"] is not None]
        queries = [r["queries"] for r in rows if r["queries"]]
        # A cell of fewer trials than first queries times only some of them, from an offset of its own, so two such
        # cells may share few queries.
        subsets = (" Rows with fewer queries time a subset of them, which differs from system to system, so the "
                   "ratios paired by query compare only the queries both timed.") \
            if queries and min(queries) < max(queries) else ""
        charts.append({
            "id": f"cold-{scale}-{round(t * 100)}", "kind": "h2h-cold",
            "title": f"Cold first query at {millions(n)} vectors, recall@10 {t:.2f}",
            "dataset": dataset, "n": n, "unit": "microseconds", "target_recall10": t, "preliminary": False,
            "legend": ["ColdIVF", "Other systems"],
            "subtitle": f"{about} Each dot is the p50 of {fmt_range([r['trials'] for r in rows])} true-cold trials, "
                        "each a fresh process with the index evicted from the cache, and its line runs to the p99. "
                        f"Each system runs at its pick for recall@10 {t:.2f}, whose recall over the "
                        f"{fmt_count([r['recall10_warm_queries'] for r in rows])} test queries is "
                        f"{min(recalls):.3f} to {max(recalls):.3f}. Log scale.",
            "footer": "ColdIVF runs through its Python module, as every other system runs through its binding, and "
                      "each trial first searches a small index of its own config. Indexes that load whole at open, "
                      "and the in-memory graphs, read most of their index before the first query. A system's trials "
                      "time at most 50 distinct test queries, so a p99 shows the spread over repeats of them."
                      + subsets,
            "notes": ["Each row is headtohead.py's pick for the target, pick.py's verdict rule on the tune split, "
                      "measured on the test split.",
                      "p50_ratio is the row's C0 p50 over ColdIVF's at the same target, paired_ratio the median of "
                      "per-query ratios over the queries both timed with its 95% bootstrap interval, and exact_ratio "
                      "the ratio of the two p50s at exactly the target, where both have cells around it."],
            "rows": rows, "models": [], "not_reached": missing(rows), "reference": entry["reference"],
        })

        if t == 0.95:
            # ColdIVF's 0.95 setting timed again in every rival's batch, through its binaries, and on a fresh clone.
            charts[-1]["controls"] = [c for c in data["controls"] if c["kind"] == "cold"]

    for t in H2H_TARGETS:
        entry = data["targets"][t]
        rows = one_per_system(sorted((h2h_warm_row(r, folder, repo) for r in entry["rows"]
                                      if r["warm_p50_us"] is not None), key=lambda r: r["p50_us"]),
                              lambda r: r["p50_us"])
        recalls = [r["recall10_warm"] for r in rows if r["recall10_warm"] is not None]
        partial = defaultdict(list)

        for r in rows:
            if r["mode"] not in (None, "whole"):
                partial[r["mode"]].append(r["label"])

        reads = f" {warm_reads_note(partial)}" if partial else ""
        charts.append({
            "id": f"warm-{scale}-{round(t * 100)}", "kind": "h2h-warm",
            "title": f"Warm query at {millions(n)} vectors, recall@10 {t:.2f}",
            "dataset": dataset, "n": n, "unit": "microseconds", "target_recall10": t, "preliminary": False,
            "legend": ["ColdIVF", "Other systems"],
            "subtitle": f"{about} Each dot is the p50 of the {fmt_count([r['queries'] for r in rows])} test queries "
                        "at 1 thread, with every index read into memory first, and its line runs to the p99. Both are "
                        f"the median of {fmt_range([r['passes'] for r in rows])} passes over the queries.{reads} "
                        f"Each system runs at its pick for recall@10 {t:.2f}, whose recall over those queries is "
                        f"{min(recalls):.3f} to {max(recalls):.3f}. Log scale.",
            "footer": "Every system runs through its Python binding, so each time holds that binding's per-call cost.",
            "notes": ["Each row is headtohead.py's pick for the target, pick.py's verdict rule on the tune split, "
                      "measured on the test split. Of a system's configurations it is the one with the lowest warm "
                      "p50, which can differ from the one its cold chart shows, and ColdIVF's is its Python module's.",
                      "p50_ratio is the row's warm p50 over ColdIVF's at the same target."],
            "rows": rows, "not_reached": missing(rows), "reference": entry["reference"],
        })

    def board(t: float) -> dict:
        rows = one_per_system(h2h_qps_rows(data["targets"][t], folder, repo), lambda r: -r["qps"])
        return {"recall10": t, "rows": rows, "not_reached": missing(rows)}

    # Only the targets ColdIVF reaches, since a panel without ColdIVF compares nothing the README is about (user,
    # 2026-10-11), eg. recall@10 0.99 at 10M.
    reached = [t for t in headtohead.TARGETS if any(r["system"] == "coldivf" for r in data["targets"][t]["rows"])]
    boards = [board(t) for t in reached]
    # The panels stand one above the other, so each fills the page's width.
    label_w, value_w = 276.0, 60.0
    charts.append({
        "id": f"qps-{scale}", "kind": "h2h-qps", "title": f"Queries per second at {millions(n)} vectors, 6 threads",
        "unit": "queries per second", "dataset": dataset, "n": n, "preliminary": False,
        "not_reached_text": "Not reached",
        "layout": {"label_w": label_w, "plot_w": WIDTH - 2 * MARGIN - label_w - value_w, "value_w": value_w},
        "subtitle": f"{about} Each panel ranks the systems by queries per second on 6 threads, the M3 Pro's "
                    "performance cores, at their pick for one recall@10 target, the median of 3 runs of 10 s each. "
                    "Higher is faster. The panels share one log scale.",
        "footer": method_note(boards),
        "boards": boards,
    })
    rows, seen = [], set()
    shown = {r["label"] for c in charts if c["kind"] == "h2h-cold" for r in c["rows"]}

    for t in (0.95, 0.90):
        for r in data["targets"][t]["rows"]:
            if r["label"] in shown and r["system"] not in seen:
                seen.add(r["system"])
                rows.append(h2h_memory_row(r, t, folder, repo))

    # A system that reaches no target shows its cell of most recall.
    hand = []

    for r in sorted(data["hand"], key=lambda r: -(r["recall10"] or 0.0)):
        if r["system"] not in seen:
            seen.add(r["system"])
            hand.append(short_label(r))
            rows.append(h2h_memory_row(r, None, folder, repo))

    held = (f", and {listed(hand)} at {'its setting' if len(hand) == 1 else 'their settings'} of most recall, "
            f"since {'it reaches' if len(hand) == 1 else 'they reach'} neither") if hand else ""
    leave = warm_left_note(rows)
    charts.append({
        "id": f"memory-{scale}", "kind": "h2h-memory", "title": f"Memory at {millions(n)} vectors", "dataset": dataset,
        "n": n, "unit": "bytes", "preliminary": False,
        "subtitle": f"{about} Each system at its pick for recall@10 0.95, or 0.90 where it doesn't reach 0.95{held}. "
                    "Cold is the median of its trials of what the open and the first query add to the process, plus "
                    "the index's pages they bring into the page cache. Warm is the process plus the cached pages of "
                    "its index files, with every file read first.",
        "footer": " ".join(x for x in ("Pages an index copies into its own memory at open count once, in the process.",
                                       leave, "Setups that search another setup's file show that file's build.",
                                       malloc_note(rows)) if x),
        "panels": "memory", "rows": rows,
    })
    # The same rows' build and disk panels, which the README shows as an image of their own.
    charts.append({
        "id": f"build-{scale}", "kind": "h2h-memory", "title": f"Build and index size at {millions(n)} vectors",
        "dataset": dataset, "n": n, "unit": "bytes", "preliminary": False,
        "subtitle": f"{about} Each system's build peak, the most memory its build process held, and its index's bytes "
                    "per vector on disk, for the setup its memory row shows.",
        "footer": malloc_note(rows), "panels": "build", "rows": rows,
    })
    return charts, machine


# A check of a head-to-head, by scale: a suite that times ColdIVF's module and its closest rival in one batch, with
# ColdIVF's binaries as the control that ties the batch to the head-to-head's, since the head-to-head times each in a
# batch of its own.
H2H_CHECKS = {"1m": Path("coldivf-bench") / "suites" / "bench-1m-same-batch.toml"}


def same_batch(suite_path: Path, repo: Path) -> dict | None:
    """A same-batch check's cells: each one's C0 p50 and p99 over its ok trials, and each other cell against ColdIVF's
    module, the ratio of the p50s and the median of per-query ratios with its 95% interval. It also gives the test
    recall of each warm line the check ran, and the drift references at the batch's start and end. It is None until the
    check's cells have trials."""
    import driver
    import headtohead

    suite = driver.load_suite(suite_path)
    folder = suite.results_root / suite.phase
    res = headtohead.Results(folder)
    cells, every = [], []

    for system in suite.systems:
        for c in system.cold:
            trials = [x for x in res.cold(system, c.build, c.params, c.split) if x.get("status") == "ok"]

            if not trials:
                continue

            total = [x["total_us"] for x in trials]
            w1 = res.warm(system, c.build, c.params, c.split, 1) or {}
            every += trials
            cells.append({"label": headtohead.label(system.name, system.variant, c.build.config_id),
                          "system": system.name, "variant": system.variant, "config_id": c.build.config_id,
                          "setting": dict(c.params), "setting_text": headtohead.setting_text(c.params),
                          "trials": len(trials), "queries": len({x["query_id"] for x in trials}),
                          "p50_us": headtohead.median(total), "p99_us": headtohead.pct(total, 99),
                          "recall10": w1.get("recall10"), "_trials": trials,
                          "source": rel(folder / f"{system.name}.jsonl", repo)})

    module = next((c for c in cells if c["system"] == "coldivf" and c["variant"] == "python"), None)

    if module is None:
        return None

    for c in cells:
        if c is not module:
            c["against"] = {"c0_p50": c["p50_us"] / module["p50_us"],
                            "c0_paired": headtohead.paired(c["_trials"], module["_trials"])}

    refs = [{k: x.get(k) for k in ("position", "wave_us_p50", "coldivf_c0_p50_us", "ref_drift_pct")}
            for x in schema_lines(folder / "ref.jsonl")]
    return {"suite": rel(suite_path, repo), "folder": rel(folder, repo), "machine_state": driver_state(every),
            "code": code_of(every, repo), "refs": refs,
            "cells": [{k: v for k, v in c.items() if not k.startswith("_")} for c in cells]}


def schema_lines(path: Path) -> list[dict]:
    """The lines of a results file, or none when it isn't there."""
    import schema

    return schema.read(path) if path.exists() else []


def chart_data(quiet: Path = QUIET, repo: Path = REPO, h2h: dict | None = None, checks: dict | None = None) -> dict:
    """Every number the charts plot and the README's text cites, with where it came from. The head-to-head charts come
    from the suites `h2h`, by the scale that names their charts, by default the repo's bench-1m.toml and bench-10m.toml,
    and each is left out when its suite isn't there. The same-batch check of a scale in `checks`, by default the repo's
    H2H_CHECKS, joins its cold chart at 0.95. The chart of ColdIVF in each setup comes from the quiet run in `quiet`.
    The machine is the first one the head-to-heads' lines recorded."""
    scenarios = cold_matched(quiet, repo)
    suites = {scale: Path(path) for scale, path in (h2h or {s: repo / p for s, p in H2H_SUITES.items()}).items()}
    checks = {s: repo / p for s, p in H2H_CHECKS.items()} if checks is None else checks
    big, machine, used = [], {}, {}

    for scale, path in suites.items():
        if not path.exists():
            continue

        made, found = h2h_charts(path, repo, scale)
        check = Path(checks[scale]) if scale in checks else None
        checked = same_batch(check, repo) if check is not None and check.exists() else None

        if checked is not None:
            next(c for c in made if c["id"] == f"cold-{scale}-95")["same_batch"] = checked

        big += made
        machine = machine or found
        used[scale] = rel(path, repo)

    return tidy({
        "about": "Every number the README's charts plot and its text cites, written by coldivf-bench/py/charts.py. "
                 "The 1M and 10M head-to-head charts come from their suites' results through headtohead.py, and the "
                 "chart of ColdIVF in each setup on Cohere 1M from the quiet run in results/quiet2.",
        "preliminary": not big,
        "machine": machine,
        "drawn_with": None,
        "datasets": dict(DATASETS),
        "h2h_suites": used,
        "charts": [*big, scenarios],
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
    """A row's setting in words, eg. "nprobe 96, rerank 35". A row that names its setting in words already, eg. by the
    names the head-to-head's tables use, keeps those words."""
    if row.get("setting_text"):
        return row["setting_text"]

    return ", ".join(f"{k} {v}" for k, v in row["setting"].items()) or "no setting"


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


def millions(n: int) -> str:
    return f"{n / 1e6:g}M"


def headline_table(data: dict, chart_id: str) -> str:
    """A README table of a cold chart at matched recall, eg. cold-coldivf: each row's setting, recall, times and
    ratio."""
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


def h2h_table(data: dict, chart_id: str) -> str:
    """A README table of a 10M cold chart: each row's setting, test recall, trials, cold times and its ratios to
    ColdIVF, the plain one and the one paired by query."""
    rows = next(c for c in data["charts"] if c["id"] == chart_id)["rows"]
    out = ["| System | Setting | Recall@10 | Trials (Queries) | Cold p50 | Cold p99 | p50 Against ColdIVF "
           "| Paired by Query |", "|---|---|---|---|---|---|---|---|"]

    for r in rows:
        pair = r.get("paired_ratio")
        paired_text = f"{pair['median']:.2f}x ({pair['low']:.2f} to {pair['high']:.2f})" if pair else "-"
        out.append(f"| {r['label']} | {fmt_setting(r)} | {r['recall10_warm']:.3f} | {r['trials']} ({r['queries']}) "
                   f"| {fmt_ms(r['p50_us'])} | {fmt_ms(r['p99_us'])} | {r['p50_ratio']:.2f}x | {paired_text} |")

    return "\n".join(out)


def chart_of(data: dict, chart_id: str) -> dict:
    return next(c for c in data["charts"] if c["id"] == chart_id)


def warm_ms(us: float) -> str:
    """A warm time in milliseconds, with 3 decimals under 1 ms, eg. "0.348 ms" or "2.10 ms"."""
    return f"{us / 1000:.3f} ms" if us < 1000 else fmt_ms(us)


def bold(text: str, on: bool) -> str:
    return f"**{text}**" if on and text else text


def h2h_warm_table(data: dict, scale: str) -> str:
    """A README table of a head-to-head's warm charts: each system's warm p50 and its queries per second on 6 threads
    at each target, as the charts show them, and how its 6-thread queries ran side by side. The rows follow the warm
    p50 at the highest target. A cell from another configuration than the row's own names it."""
    warm = [c for c in data["charts"] if c.get("kind") == "h2h-warm" and c["id"].startswith(f"warm-{scale}-")]
    qps = chart_of(data, f"qps-{scale}")
    # The warm charts show fixed targets and the QPS panels those ColdIVF reaches, so the table takes every one of both.
    targets = sorted({c["target_recall10"] for c in warm} | {b["recall10"] for b in qps["boards"]})
    cells = defaultdict(dict)

    # A warm chart's row names its system in full and a QPS panel's in short, so each cell holds both names.
    for c in warm:
        for r in c["rows"]:
            cells[r["system"]][(c["target_recall10"], "p50_us")] = {"label": r["short"], "full_label": r["label"],
                                                                    "p50_us": r["p50_us"]}

    for b in qps["boards"]:
        for r in b["rows"]:
            cells[r["system"]][(b["recall10"], "qps")] = r

    def order(system: str) -> tuple:
        p50 = [(t, cells[system][(t, "p50_us")]["p50_us"]) for t in reversed(targets) if (t, "p50_us") in cells[system]]
        return (-p50[0][0], p50[0][1]) if p50 else (0.0, math.inf)

    head = " | ".join(f"Warm p50 at {t:.2f} | QPS, 6 Threads, at {t:.2f}" for t in targets)
    out = [f"| System | {head} | QPS Method |", "|---|" + "---|" * (2 * len(targets) + 1)]

    for system in sorted(cells, key=order):
        own = next(cells[system][(t, v)] for t in reversed(targets) for v in ("p50_us", "qps")
                   if (t, v) in cells[system])
        ours = system == "coldivf"
        row = [bold(own["full_label"], ours)]

        for t in targets:
            for value in ("p50_us", "qps"):
                r = cells[system].get((t, value))

                if r is None:
                    reached = any((t, v) in cells[system] for v in ("p50_us", "qps"))
                    row.append("-" if reached else "not reached")
                    continue

                text = warm_ms(r["p50_us"]) if value == "p50_us" else f"{r['qps']:,.0f}"
                other = f" ({r['label']})" if r["full_label"] != own["full_label"] else ""
                row.append(bold(text, ours) + other)

        methods = list(dict.fromkeys(cells[system][(t, "qps")].get("method") or "-" for t in targets
                                     if (t, "qps") in cells[system]))
        row.append(", ".join(methods) or "-")
        out.append("| " + " | ".join(row) + " |")

    return "\n".join(out)


def size(b: float | None) -> str:
    return fmt_bytes(b) if b is not None else "-"


def h2h_memory_table(data: dict, scale: str) -> str:
    """A README table of a head-to-head's memory chart: what each system's open and first cold query add, its warm
    process and cached pages, and its build's time, peak and index size. ColdIVF comes first, then the others by their
    cold memory. A build marked * reuses ColdIVF's partition, so its time leaves out the clustering."""
    rows = chart_of(data, f"memory-{scale}")["rows"]
    out = ["| System | Open + First Query, Process and Pages | Warm: Process + Cached Pages | Build Time | Build Peak "
           "| Index on Disk |", "|---|---|---|---|---|---|"]

    for r in sorted(rows, key=lambda r: (r["system"] != "coldivf", r["cold"]["total"] or math.inf)):
        ours = r["system"] == "coldivf"
        target = r["target_recall10"]
        at = "" if target == 0.95 else f", at {target:.2f}" if target else ", at its setting of most recall"
        cached = r["warm"]["resident_file_bytes"]
        warm = (f"{size(r['warm']['process_bytes'])} + {size(cached)}" if cached
                else size(r["warm"]["process_bytes"]))
        b = r["build"]
        star = "*" if b["shared_partition"] and not ours else ""
        seconds = b["seconds"]
        took = "-" if seconds is None else f"{seconds:.0f} s{star}" if seconds < 60 else f"{seconds / 60:.1f} min{star}"
        cells = [bold(r["full_label"], ours) + at, size(r["cold"]["total"]), warm, took, size(b["peak_footprint"]),
                 size(r["disk"]["index_bytes"])]
        out.append("| " + " | ".join(bold(c, ours) if i else c for i, c in enumerate(cells)) + " |")

    return "\n".join(out)


# Each chart's colors, for a dark page. ColdIVF is blue and everything else is gray, so the one series that matters
# stands out.
THEME = {"surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7", "grid": "#2c2c2a", "axis": "#383835",
         "coldivf": "#3987e5", "coldivf_tint": "#184f95", "rest": "#898781", "rest_tint": "#52514e"}

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
    """A figure laid out in points from its top left corner, with the title every chart has. The README describes each
    chart beside it, so the image holds no description. `body` is the height left for the chart, which starts at
    `top`."""

    def __init__(self, t: dict, title: str, body: float):
        m = _mpl()
        self.t = t
        self.top = 48
        self.bottom = self.top + body
        self.w, self.h = WIDTH, self.bottom + 16
        self.fig = m.Figure(figsize=(self.w / 72, self.h / 72), dpi=72)
        m.Canvas(self.fig)
        self.text(MARGIN, 28, title, size=14, weight="bold", color=t["ink"])

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


def _log_ticks(ax, lo: float, hi: float, fmt=lambda v: f"{v:g}") -> None:
    """Ticks on the x axis at each power of ten between lo and hi, labeled with `fmt`."""
    m = _mpl()
    ticks = [10.0 ** e for e in range(math.floor(math.log10(lo)), math.ceil(math.log10(hi)) + 1)]
    ticks = [v for v in ticks if lo <= v <= hi]
    ax.xaxis.set_major_locator(m.FixedLocator(ticks))
    ax.xaxis.set_minor_locator(m.NullLocator())
    ax.set_xticklabels([fmt(v) for v in ticks])


def _grid(ax, t: dict, axis: str = "x") -> None:
    ax.grid(True, axis=axis, color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def _dot(ax, x: float, y: float, color: str, t: dict, size: float, z: float = 3) -> None:
    """A dot at (x, y) in `color` with a ring of the surface around it."""
    ax.plot([x], [y], marker="o", markersize=size, markerfacecolor=color, markeredgecolor=t["surface"],
            markeredgewidth=1.75, linestyle="none", zorder=z)


def _row_detail(r: dict) -> str:
    """The line under a row's label: its setting, its recall and the trials or, for a warm row, the queries it timed."""
    parts = [fmt_setting(r)]

    if r.get("recall10_warm") is not None:
        parts.append(f"recall {r['recall10_warm']:.3f}")

    if r.get("trials") is not None:
        threads = f", {r['threads']} threads" if (r.get("threads") or 1) > 1 else ""
        parts.append(f"{r['trials']} trials{threads}")
    elif r.get("queries") is not None:
        parts.append(f"{r['queries']:,} queries")

    return " · ".join(parts)


def draw_matched(data: dict, chart: dict, t: dict, path: Path, axis: str = "Cold first query (ms), log scale",
                 fmt=fmt_ms) -> None:
    """A dot at each row's p50 with a line to its p99, on a log scale labeled `axis`. The two times, written by `fmt`,
    stand in columns on the right under one header."""
    m = _mpl()
    rows = chart["rows"]
    row_h, left, plot_w = 31.0, 300.0, 290.0
    page = Page(t, chart["title"] + (" (preliminary)" if chart.get("preliminary", False) else ""),
                34 + row_h * len(rows) + 42)
    top = page.top + 34
    ax = page.axes(left, top, plot_w, row_h * len(rows))
    lo = 10 ** math.floor(math.log10(min(r["p50_us"] for r in rows) / 1.2))
    hi = 10 ** math.ceil(math.log10(max(r["p99_us"] or r["p50_us"] for r in rows)))
    # The axis's length in points per power of ten, to tell when a p99 falls under its dot.
    per = plot_w / math.log10(hi / lo)
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.yaxis.set_major_locator(m.NullLocator())
    ax.spines["left"].set_visible(False)
    _log_ticks(ax, lo, hi, fmt=lambda v: f"{v / 1000:,g}")
    ax.set_xlabel(axis, labelpad=6)
    _grid(ax, t)
    rowpos = m.blended(ax.transAxes, ax.transData)

    for x, name in ((14, "p50"), (76, "p99")):
        page.text(left + plot_w + x, top - 3, name, size=8.5, color=t["ink2"])

    for i, r in enumerate(rows):
        ours = r["group"] == "coldivf"
        color = t["coldivf"] if ours else t["rest"]
        weight = "bold" if ours else "normal"
        ax.axhline(i, color=t["grid"], linewidth=0.6, zorder=0)

        # A row without a p99, eg. of a warm line that recorded only its p50, shows only its dot, and so does one whose
        # p99 lands within 6 pt of its p50, where the dot would hide the tick. Its column still gives the p99.
        if r["p99_us"] is not None and per * math.log10(r["p99_us"] / r["p50_us"]) > 6:
            ax.plot([r["p50_us"], r["p99_us"]], [i, i], color=color, linewidth=1.5, zorder=2, solid_capstyle="butt")
            ax.plot([r["p99_us"]], [i], marker="|", markersize=9, markeredgewidth=1.5, color=color, zorder=2)

        _dot(ax, r["p50_us"], i, color, t, 10)
        ax.text(-0.02, i - 0.1, r["label"], transform=rowpos, ha="right", va="bottom", size=9.5, color=t["ink"],
                weight=weight)
        ax.text(-0.02, i + 0.1, _row_detail(r), transform=rowpos, ha="right", va="top", size=8, color=t["ink2"])
        ax.text(1 + 14 / plot_w, i, fmt(r["p50_us"]), transform=rowpos, ha="left", va="center", size=9.5,
                color=t["ink"], weight=weight)
        ax.text(1 + 76 / plot_w, i, fmt(r["p99_us"]) if r["p99_us"] is not None else "-", transform=rowpos,
                ha="left", va="center", size=8.5, color=t["ink2"])

    dot = {"marker": "o", "markersize": 8, "linestyle": "none"}
    ours_label, rest_label = chart["legend"]
    handles = [m.Line2D([], [], markerfacecolor=t["coldivf"], markeredgecolor=t["surface"], label=ours_label, **dot),
               m.Line2D([], [], markerfacecolor=t["rest"], markeredgecolor=t["surface"], label=rest_label, **dot),
               m.Line2D([], [], color=t["rest"], marker="|", markersize=9, markeredgewidth=1.5, linewidth=1.5,
                        label="Line to the p99")]
    page.fig.legend(handles=handles, loc="upper left", ncol=3, handlelength=1.6, columnspacing=1.8,
                    borderaxespad=0, borderpad=0, bbox_to_anchor=page.anchor(MARGIN, page.top + 4))
    page.save(path)


def draw_warm_matched(data: dict, chart: dict, t: dict, path: Path) -> None:
    """A warm chart, drawn as the cold ones are, with its times in milliseconds as the README's warm table writes
    them, eg. "0.348 ms"."""
    draw_matched(data, chart, t, path, "Warm query at 1 thread (ms), log scale", warm_ms)


def draw_leaderboard(data: dict, chart: dict, t: dict, path: Path) -> None:
    """A QPS chart's panels, one per recall target, one above the other on one log scale. Each ranks the systems by
    queries per second, with each one's full name, as the warm charts name it, and its setting and recall under it."""
    m = _mpl()
    boards = chart["boards"]
    every = [r["qps"] for b in boards for r in b["rows"]]
    lo, hi = min(every) / 1.5, max(every) * 1.5
    layout = chart["layout"]
    row_h, gap = 25.0, 20.0
    label_w, plot_w, value_w = layout["label_w"], layout["plot_w"], layout["value_w"]
    gone = chart["not_reached_text"]
    short = [wrap(f"{gone}: {', '.join(r['label'] for r in b['not_reached'])}.", 7.5, label_w + plot_w + value_w)
             if b["not_reached"] else [] for b in boards]
    # Each panel's height: its title, its rows, the axis under them and the systems that don't reach its target.
    heights = [30 + row_h * len(b["rows"]) + 36 + 10 * len(lines) for b, lines in zip(boards, short)]
    page = Page(t, chart["title"] + (" (preliminary)" if chart.get("preliminary", False) else ""),
                sum(heights) + gap * (len(boards) - 1) + 4)
    y = page.top

    for b, board in enumerate(boards):
        rows = board["rows"]
        ax = page.axes(MARGIN + label_w, y + 30, plot_w, row_h * len(rows))
        ax.set_xscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.yaxis.set_major_locator(m.NullLocator())
        ax.spines["left"].set_visible(False)
        _log_ticks(ax, lo, hi, fmt=lambda v: f"{v:,.0f}")
        ax.tick_params(axis="x", labelsize=7.5)
        ax.set_xlabel("queries per second, log scale", size=8, labelpad=3)
        _grid(ax, t)
        page.text(MARGIN, y + 18, f"Recall@10 {board['recall10']:.2f}", size=10, weight="bold", color=t["ink"])
        pos = m.blended(ax.transAxes, ax.transData)

        for i, r in enumerate(rows):
            ours = r["system"] == "coldivf"
            color = t["coldivf"] if ours else t["rest"]
            ink = t["coldivf"] if ours else t["ink"]
            weight = "bold" if ours else "normal"
            recall = f" · recall {r['recall10']:.3f}" if r.get("recall10") is not None else ""
            ax.plot([lo, r["qps"]], [i, i], color=color, linewidth=2 if ours else 1.2, zorder=2,
                    solid_capstyle="butt")
            ax.plot([r["qps"]], [i], marker="o", markersize=6 if ours else 4.5, color=color,
                    markeredgecolor=t["surface"], markeredgewidth=0.8, zorder=3)
            # The rank stands just before the name, which ends at the plot.
            name_w = text_width(r["full_label"], 8.5, weight)
            ax.text((-8 - name_w - 6) / plot_w, i - 0.1, f"{i + 1}", transform=pos, ha="right", va="bottom", size=7.5,
                    color=t["ink2"])
            ax.text(-8 / plot_w, i - 0.1, r["full_label"], transform=pos, ha="right", va="bottom", size=8.5, color=ink,
                    weight=weight)
            ax.text(-8 / plot_w, i + 0.1, fmt_setting(r) + recall, transform=pos, ha="right", va="top", size=7.5,
                    color=t["ink2"])
            ax.text(1 + value_w / plot_w, i, f"{r['qps']:,.0f}", transform=pos, ha="right", va="center", size=8.5,
                    color=ink, weight=weight)

        for k, line in enumerate(short[b]):
            page.text(MARGIN, y + 30 + row_h * len(rows) + 40 + 10 * k, line, size=7.5, color=t["ink2"])

        y += heights[b] + gap

    page.save(path)


def draw_memory(data: dict, chart: dict, t: dict, path: Path) -> None:
    """A memory chart's panels: with `panels` "memory", the cold and the warm memory side by side under one legend, and
    with "build", the build's peak and the index's bytes per vector."""
    m = _mpl()
    rows = chart["rows"]
    row_h, label_w, plot_w, gap_x = 14.5, 104.0, 196.0, 70.0
    panel_h = row_h * len(rows)

    def cold(r: dict) -> tuple:
        """A row's process memory and the index pages its open and first query brought into the page cache."""
        return r["cold"]["process"], r["cold"]["pages"]

    every = [
        ("Cold: the open and the first query", "", cold, False, fmt_bytes),
        ("Warm: every index file read first", "",
         lambda r: (r["warm"]["process_bytes"], r["warm"]["resident_file_bytes"]), False, fmt_bytes),
        ("Build: the peak while building", "", lambda r: (r["build"]["peak_footprint"], None), False, fmt_bytes),
        ("Disk: index bytes per vector", "", lambda r: (r["disk"]["bytes_per_vector"], None), False,
         lambda v: f"{v:,.0f} B"),
    ]
    which = chart["panels"]
    panels = every[:2] if which == "memory" else every[2:]
    # One legend sits above the panels that split memory into the process and the index's pages.
    legend_h = 20.0 if which == "memory" else 0.0
    page = Page(t, chart["title"] + (" (preliminary)" if chart.get("preliminary", False) else ""),
                legend_h + math.ceil(len(panels) / 2) * (28 + panel_h + 30) + 6)

    if legend_h:
        handles = [m.Patch(facecolor=t["rest"], label="Process memory"),
                   m.Patch(facecolor=t["rest_tint"], label="Index pages in the page cache"),
                   m.Patch(facecolor=t["coldivf"], label="ColdIVF")]
        page.fig.legend(handles=handles, loc="upper left", ncol=3, handlelength=1.2, columnspacing=1.8,
                        borderaxespad=0, borderpad=0, bbox_to_anchor=page.anchor(MARGIN, page.top + 2))

    for i, (title, note, value, log, fmt) in enumerate(panels):
        r_i, k = divmod(i, 2)
        left = MARGIN + label_w + k * (label_w + plot_w + gap_x)
        top = page.top + legend_h + 26 + r_i * (panel_h + 58)
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

    page.save(path)


# How each chart is drawn, by its id, or by its kind for the head-to-head's charts of every scale.
DRAW = {"cold-coldivf": draw_matched, "h2h-cold": draw_matched, "h2h-warm": draw_warm_matched,
        "h2h-qps": draw_leaderboard, "h2h-memory": draw_memory}


def draw(data: dict, out: Path) -> list[Path]:
    """Every chart for a dark page, as <id>-dark.svg."""
    m = _mpl()
    written = []

    with m.matplotlib.rc_context(_rc(THEME)):
        for chart in data["charts"]:
            if not (chart.get("rows") or chart.get("boards")):
                print(f"no lines for the {chart['id']} chart, so it isn't drawn", file=sys.stderr)
                continue

            path = out / chart["files"]["dark"]
            DRAW[chart.get("kind") or chart["id"]](data, chart, THEME, path)
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
    p.add_argument("--quiet", type=Path, default=QUIET, help="the quiet-window run's results folder")
    p.add_argument("--h2h", action="append", default=None, metavar="SCALE=SUITE.toml",
                   help="a head-to-head suite and the scale that names its charts, eg. 1m=suite.toml, once per suite, "
                        "by default the repo's bench-1m.toml and bench-10m.toml")
    p.add_argument("--check", action="append", default=None, metavar="SCALE=SUITE.toml",
                   help="a same-batch check of a scale's head-to-head, by default the repo's H2H_CHECKS")
    p.add_argument("--out", type=Path, default=OUT, help="where chart-data.json and the charts go")
    p.add_argument("--repo", type=Path, default=REPO, help="the root that paths in chart-data.json are relative to")
    p.add_argument("--data-only", action="store_true", help="write chart-data.json without drawing")
    args = p.parse_args(argv)

    def scales(values: list[str] | None) -> dict | None:
        if values is None:
            return None

        pairs = [v.split("=", 1) for v in values]

        if any(len(x) != 2 for x in pairs):
            p.error("each --h2h and --check takes SCALE=SUITE.toml")

        return {scale: Path(path) for scale, path in pairs}

    data = chart_data(args.quiet, args.repo, scales(args.h2h), scales(args.check))

    for chart in data["charts"]:
        chart["files"] = {"dark": f"{chart['id']}-dark.svg"}

    try:
        publish(data, args.out, charts=not args.data_only)
    except ImportError as e:
        print(f"wrote nothing, since drawing needs matplotlib: {e}", file=sys.stderr)
        return 1

    if args.data_only:
        print("wrote only chart-data.json, so the charts may not match it until they are drawn", file=sys.stderr)

    for scale in data["h2h_suites"]:
        for chart in data["charts"]:
            if chart.get("kind") == "h2h-cold" and chart["id"].startswith(f"cold-{scale}-"):
                print(h2h_table(data, chart["id"]))
                print()

        print(h2h_warm_table(data, scale))
        print()
        print(h2h_memory_table(data, scale))
        print()

    print(headline_table(data, "cold-coldivf"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
