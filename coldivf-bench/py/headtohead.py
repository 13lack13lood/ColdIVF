"""The head-to-head of a benchmark phase: each system's operating point at each recall target, by pick.py's rule, with
its warm, true-cold (C0) and memory numbers on the test split, and how it compares with ColdIVF.

    python headtohead.py [--suite SUITE.toml] [--reference NAME:VARIANT] [--json OUT.json] [--md OUT.md]

For each build with a warm grid on the tune split, the point for a target is pick.py's verdict. Of the 3 cheapest
settings by warm 1-thread p50 whose tune recall@10 reaches the target, it is the one with the lowest C0 p50 over its
tune trials, or a cheaper one whose 95% interval for that median overlaps it, or the cheapest when no screen ran. Its
numbers come from the test split: recall@10 and p50 from the 1-thread warm line, queries per second (QPS) at 6 and 12
threads with how those queries ran side by side, through the library's call for a batch of queries, one process per
thread or threads in one process, and the C0 trials of that setting. A build that reaches no target shows the cells the
suite holds for it.

Memory follows one rule for every system, what the process holds plus the pages of the index's own files in the page
cache. A cold trial's memory is the footprint that the open and the first query add, plus the file pages they brought
in. A range the system copies into its own memory at open counts once, in the footprint. A warm line's memory is the
process's footprint plus the cached pages of the index's files, in the warm mode the line names: "whole", with every
file read through first, or one a system defines, eg. LanceDB's "index", which leaves its table on disk. macOS malloc's
cache of freed large blocks stays inside a footprint, so each build's peak comes with whether that cache was on.

ColdIVF's numbers against the others are those of the reference, by default its Python module, which runs in the
adapter's process as every other system's binding does. Each comparison gives the ratio of C0 p50s at the same target
and the median of per-query ratios over the test queries both timed, with a 95% bootstrap interval. It also gives each
system's C0 p50 at exactly the target, linear in the log of the latency between the two of its cells whose test recall
lies around it. Control builds, whose config_id ends in -vs-NAME, -clone or -vmcheck, come apart.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import driver
import pick
import schema

SUITE = Path(__file__).resolve().parent.parent / "suites" / "bench-10m.toml"
TARGETS = (0.90, 0.95, 0.99)
REFERENCE = "coldivf:python"
RESAMPLES = 10_000
CONTROL = re.compile(r"-(vs-.+|clone|vmcheck)$")

# A system's name in the tables, by system and variant. A faiss IVF build also names its partition.
NAMES = {
    ("coldivf", "python"): "ColdIVF",
    ("coldivf", ""): "ColdIVF, binaries",
    ("faiss-ivf", "shipped"): "faiss IVF as shipped",
    ("faiss-ivf", "parallel"): "faiss IVF, parallel",
    ("faiss-ivf", "batched"): "faiss IVF, batched reads",
    ("faiss-ivf", "ondisk"): "faiss IVF, on-disk lists",
    ("faiss-ivf", "graph"): "faiss IVF, HNSW quantizer",
    ("faiss-fastscan", "fastscan"): "faiss FastScan + SQ8",
    ("faiss-hnsw", ""): "faiss HNSW32-SQ8",
    ("lancedb", ""): "LanceDB IVF_RQ",
    ("paimon", ""): "Paimon",
    ("rabitq-ivf", ""): "RaBitQ-Library IVF",
    ("rabitq-hnsw", ""): "RaBitQ-Library HNSW",
    ("usearch", ""): "usearch HNSW",
    ("turbovec", ""): "turbovec 4-bit",
    ("qdrant-edge", ""): "qdrant-edge",
    ("diskann", ""): "DiskANN",
    ("astervec", ""): "AsterVec",
    ("symphonyqg", ""): "SymphonyQG",
    ("sqlite-vec", ""): "sqlite-vec",
    ("flat", ""): "flat scan",
}
LABELS = {
    ("paimon", "", "b4"): "Paimon 4-bit", ("paimon", "", "b8"): "Paimon 8-bit",
    ("turbovec", "", "b4"): "turbovec 4-bit", ("turbovec", "", "b4-cal"): "turbovec 4-bit, calibrated",
}
# A build's name from its config id, for the systems whose builds differ in it, eg. cp640-b9 for RaBitQ-Library's IVF
# at 9 bits and f16-m16 for usearch with 16-bit float rows, so no two builds of a phase share a name.
CONFIG_LABELS = (
    ("rabitq-ivf", re.compile(r"cp\d+-b(\d+)"), "RaBitQ-Library IVF {}-bit"),
    ("rabitq-hnsw", re.compile(r"b(\d+)"), "RaBitQ-Library HNSW {}-bit"),
    ("usearch", re.compile(r"(i8|f16|f32)-m\d+"), "usearch HNSW {}"),
)
# The partition in a faiss IVF build's config id, eg. cp640-rq1 for ColdIVF's at 640 vectors per list and vpl2560-rq1
# for faiss's own at 2,560.
PARTITION = re.compile(r"(?:cp|vpl)(\d+)-rq1(?:-hnsw32)?")

# Search settings by the names the tables use, and those every setting of a grid shares, which they leave out.
SETTING_NAMES = {"refine_factor": "refine", "expansion_search": "expansion", "ef_search": "ef"}
CONSTANT_SETTINGS = {"qb", "prefetch_threads", "quantizer_ef"}


def pct(values: list[float], q: float) -> float | None:
    """The q-th percentile by nearest rank, eg. the 99th of 100 values is the 99th smallest, and of 30 the largest."""
    v = sorted(values)
    return v[max(0, min(len(v) - 1, math.ceil(q / 100 * len(v)) - 1))] if v else None


def median(values: list[float]) -> float | None:
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def key(params: dict | None) -> str:
    return pick.canonical(params or {})


def label(system: str, variant: str, config_id: str) -> str:
    if (system, variant, config_id) in LABELS:
        return LABELS[(system, variant, config_id)]

    for owner, pattern, text in CONFIG_LABELS:
        found = pattern.fullmatch(config_id)

        if system == owner and found:
            return text.format(found[1])

    name = NAMES.get((system, variant), f"{system}:{variant}" if variant else system)
    found = PARTITION.fullmatch(config_id)
    return f"{name}, {int(found[1]):,} per list" if system == "faiss-ivf" and found else name


def setting_text(params: dict | None) -> str:
    """A setting in words, eg. "nprobe 192, rerank 39", or "none" for a system without search settings."""
    parts = [f"{SETTING_NAMES.get(k, k)} {v}" for k, v in (params or {}).items() if k not in CONSTANT_SETTINGS]
    return ", ".join(parts) or "none"


def first_query_pages(line: dict) -> int | None:
    """The bytes of the index's own files that a cold trial's open and first query left in the page cache, besides the
    ranges the system copies into its own memory, which its footprint already holds. The faiss adapters measure these
    pages right after the first query. For every other system they are the bytes that the open and the first query
    read from disk, since every read here goes through the page cache, less the loaded ranges' cached bytes, which the
    open reads whole."""
    extra = line.get("extra") or {}
    loaded = int(extra.get("resident_loaded_bytes_after") or 0)
    after_first = extra.get("resident_after_first")

    if isinstance(after_first, dict) and after_first:
        return max(0, int(sum(after_first.values())) - loaded)

    if line.get("disk_bytes") is None:
        return None

    return max(0, int(line["disk_bytes"]) - loaded)


def qps_method(line: dict | None) -> str | None:
    """How a warm line on several threads ran its queries side by side: "batch call" through the library's own call for
    a batch of queries, "processes" with one process per thread, or "threads" with that many threads in one process."""
    if not line:
        return None

    if line.get("batch_api"):
        return "batch call"

    return "processes" if (line.get("processes") or 1) > 1 else "threads"


def malloc_large_cache(line: dict | None) -> str | None:
    """Whether macOS malloc's cache of freed large blocks was "on" or "off" in the process a line measured. The cache
    keeps freed blocks inside the footprint, so footprints compare only under one setting. It is the MallocLargeCache
    the line recorded, or else how its job ran: the machine's job gate turns the cache off for a normal job and leaves
    it on for the solo job, which holds every slot. A line from before the variable was recorded, of a batch that saw
    no job gate, gives None."""
    if not line:
        return None

    env = (line.get("provenance") or {}).get("env") or {}

    if "MallocLargeCache" in env:
        return "off" if env["MallocLargeCache"] == "0" else "on"

    gate = line.get("gate") or {}

    if not gate.get("locks_seen"):
        return None

    return "on" if gate.get("solo") else "off"


def cold_memory(line: dict) -> dict:
    """A cold trial's memory: the footprint its open and first query added, the file pages they brought in, and both."""
    process, pages = line.get("footprint_delta"), first_query_pages(line)
    total = process + pages if process is not None and pages is not None else None
    return {"process": process, "pages": pages, "total": total}


def per_query(trials: list[dict]) -> dict[int, float]:
    """The median C0 time of each query over the trials that timed it, in microseconds."""
    times = defaultdict(list)

    for x in trials:
        times[int(x["query_id"])].append(float(x["total_us"]))

    return {q: statistics.median(v) for q, v in times.items()}


def paired(theirs: list[dict], ours: list[dict], resamples: int = RESAMPLES, seed: int = 1) -> dict | None:
    """Their C0 time over ours, query by query, over the test queries both cells timed: the median of the per-query
    ratios with its 95% bootstrap interval, from `resamples` resamples of those queries, their least and most, and the
    number of queries on which theirs is faster."""
    a, b = per_query(theirs), per_query(ours)
    shared = sorted(set(a) & set(b))

    if not shared:
        return None

    ratios = [a[q] / b[q] for q in shared]
    rng = random.Random(seed)
    boots = sorted(statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(resamples))
    return {"queries": len(shared), "median": statistics.median(ratios), "low": boots[int(0.025 * resamples)],
            "high": boots[int(math.ceil(0.975 * resamples)) - 1], "min": min(ratios), "max": max(ratios),
            "faster": sum(1 for r in ratios if r < 1)}


def at_exactly(points: list[dict], target: float) -> dict | None:
    """The C0 p50 at recall@10 exactly `target`, linear in the log of the latency between the point of most recall below
    it and the point of least recall at or above it, from `points` with "recall10", "p50_us" and "setting". A point at
    exactly the target gives its own p50. It is None unless points lie on both sides or one is at the target."""
    below = [p for p in points if p["recall10"] < target]
    above = [p for p in points if p["recall10"] >= target]
    at = [p for p in above if p["recall10"] == target]

    if at:
        a = min(at, key=lambda p: p["p50_us"])
        return {"p50_us": a["p50_us"], "between": [{k: a[k] for k in ("setting", "recall10", "p50_us")}]}

    if not below or not above:
        return None

    a = max(below, key=lambda p: (p["recall10"], -p["p50_us"]))
    b = min(above, key=lambda p: (p["recall10"], p["p50_us"]))
    f = (target - a["recall10"]) / (b["recall10"] - a["recall10"])
    p50 = math.exp((1 - f) * math.log(a["p50_us"]) + f * math.log(b["p50_us"]))
    return {"p50_us": p50, "between": [{k: p[k] for k in ("setting", "recall10", "p50_us")} for p in (a, b)]}


class Results:
    """Every line of a phase's results folder, apart from its batch and drift reference lines."""

    def __init__(self, folder: Path) -> None:
        self.lines = []

        for path in sorted(Path(folder).glob("*.jsonl")):
            if path.name not in ("batch.jsonl", "ref.jsonl"):
                self.lines += schema.read(path)

        self.by_build = defaultdict(list)

        for x in self.lines:
            self.by_build[(x.get("system"), x.get("variant") or "", x.get("config_id"), x.get("n"))].append(x)

    def of(self, system: driver.System, build: driver.Build) -> list[dict]:
        return self.by_build[(system.name, system.variant, build.config_id, build.n)]

    def warm(self, system: driver.System, build: driver.Build, params: dict, split: str, threads: int) -> dict | None:
        """The latest ok warm line of the setting, else the latest invalid one, marked by "invalid", since swap-ins or
        compressions elsewhere on the machine during its timed passes made it invalid."""
        found = [x for x in self.of(system, build) if x.get("kind") == "warm" and x.get("split") == split
                 and x.get("threads") == threads and key(x.get("search_params")) == key(params)]
        ok = [x for x in found if x.get("status") == "ok"]

        if ok:
            return ok[-1]

        flagged = [x for x in found if x.get("status") == "invalid"]
        return dict(flagged[-1], invalid=True) if flagged else None

    def cold(self, system: driver.System, build: driver.Build, params: dict, split: str) -> list[dict]:
        """The C0 trials of the setting with a time, of every status."""
        return [x for x in self.of(system, build) if x.get("kind") == "cold" and x.get("split") == split
                and x.get("regime") == "c0" and x.get("total_us") is not None
                and key(x.get("search_params")) == key(params)]

    def build_line(self, system: driver.System, build: driver.Build) -> dict | None:
        found = [x for x in self.of(system, build) if x.get("kind") == "build" and x.get("status") == "ok"]
        return found[-1] if found else None


def picks_for(res: Results, system: driver.System, build: driver.Build,
              targets: tuple[float, ...] = TARGETS) -> dict[float, tuple[pick.Point | None, str]]:
    """{target: (point or None, note)} by pick.py's verdict rule, or the cheapest setting where no screen ran."""
    grid = pick.points(res.of(system, build), system, build)
    out = {}

    if not grid:
        return out

    for t in targets:
        found = pick.cheapest(grid, t, 3)

        if not found:
            best = max(grid, key=lambda p: (p.recall10, -p.p50_us))
            out[t] = (None, f"not reached, best tune recall@10 {best.recall10:.4f} at {setting_text(best.params)}")
            continue

        if all(len(p.cold_us) >= 10 for p in found):
            (verdict,) = pick.verdict(grid, [t], 3, 10, False)
            chosen, note = verdict.cells[0][0], "screened"
        else:
            chosen, note = found[0], "the cheapest warm, with no screen"

        # The setting the suite holds a test cell for is the one measured. A pick made by an earlier rule, the fastest
        # cold alone, can differ from the median-interval rule's within the noise.
        held = [p for p in found if pick.held(system, build.config_id, p, "test")]

        if held and chosen not in held:
            chosen, note = held[0], "screened, picked by the fastest-cold rule"

        out[t] = (chosen, note)

    return out


def owner_of(systems: list[driver.System], system: driver.System, build: driver.Build):
    """The system and build that made the index a build searches, itself unless it names another's index."""
    if build.prebuilt is None:
        return system, build

    for s in systems:
        for b in s.builds:
            if b.prebuilt is None and b.out == build.prebuilt:
                return s, b

    return None


def summarize(res: Results, systems: list[driver.System], system: driver.System, build: driver.Build, params: dict,
              note: str) -> dict:
    """Every number of one system's setting on the test split."""
    w = {t: res.warm(system, build, params, "test", t) for t in (1, 6, 12)}
    trials = [x for x in res.cold(system, build, params, "test") if not (x.get("extra") or {}).get("clone")]
    ok = [x for x in trials if x.get("status") == "ok"]
    total = [x["total_us"] for x in ok]
    memory = [cold_memory(x) for x in ok]
    owner = owner_of(systems, system, build)
    b = res.build_line(*owner) if owner is not None else None
    w1 = w[1] or {}
    return {
        "label": label(system.name, system.variant, build.config_id), "system": system.name,
        "variant": system.variant, "config_id": build.config_id, "setting": dict(params),
        "setting_text": setting_text(params), "note": note,
        "recall10": w1.get("recall10"), "recall1": w1.get("recall1"), "recall100": w1.get("recall100"),
        "warm_queries": w1.get("queries"), "warm_passes": w1.get("passes"),
        "warm_mode": w1.get("mode"), "warm_p50_us": w1.get("p50_us"), "warm_p99_us": w1.get("p99_us"),
        "qps": {t: (w[t] or {}).get("qps") for t in (1, 6, 12)},
        "qps_method": {t: qps_method(w[t]) for t in (6, 12)},
        "warm_invalid": [t for t in (1, 6, 12) if w[t] and w[t].get("invalid")],
        "warm_footprint": w1.get("footprint"), "warm_cached": w1.get("resident_file_bytes"),
        "warm_total": w1.get("memory_total"),
        "trials": len(ok), "queries": len({x["query_id"] for x in ok}),
        "status": dict(Counter(x.get("status") for x in trials)),
        "c0_p50_us": median(total), "c0_p95_us": pct(total, 95), "c0_p99_us": pct(total, 99),
        "open_p50_us": median([x.get("open_us") for x in ok]),
        "disk_p50": median([x.get("disk_bytes") for x in ok]),
        "cold_process": median([m["process"] for m in memory]), "cold_pages": median([m["pages"] for m in memory]),
        "cold_total": median([m["total"] for m in memory]),
        "build_s": (b or {}).get("seconds"), "build_cluster_s": (b or {}).get("cluster_seconds"),
        "build_peak": (b or {}).get("peak_footprint"),
        "build_malloc_cache": malloc_large_cache(b), "index_bytes": (b or {}).get("index_bytes"),
        "stored_vector_bytes": (b or {}).get("stored_vector_bytes"),
        "bytes_per_vector": (b or {})["index_bytes"] / build.n if (b or {}).get("index_bytes") else None,
        "shared_file": build.prebuilt is not None and owner is not None and owner[0] is not system,
        "shared_partition": build.config_id.startswith("cp"),
        "_trials": ok, "_warm": [x for x in w.values() if x is not None],
    }


def measured_points(res: Results, system: driver.System, build: driver.Build) -> list[dict]:
    """Each test-split C0 cell of the build with a test recall: its setting, test recall@10 and C0 p50."""
    out = []

    for c in system.cold:
        if c.build is not build or c.split != "test" or c.regime != "c0" or c.fresh_clone:
            continue

        w1 = res.warm(system, build, c.params, "test", 1)
        total = [x["total_us"] for x in res.cold(system, build, c.params, "test") if x.get("status") == "ok"]

        if w1 and w1.get("recall10") is not None and total:
            out.append({"setting": dict(c.params), "recall10": w1["recall10"], "p50_us": statistics.median(total)})

    return out


def find(systems: list[driver.System], spec: str) -> driver.System | None:
    name, _, variant = spec.partition(":")
    return next((s for s in systems if s.name == name and s.variant == variant), None)


def collect(suite: driver.Suite, res: Results, reference: str = REFERENCE) -> dict:
    """The rows of every target, the builds that reach no target with their hand cells, and the controls."""
    systems = suite.systems
    ours = find(systems, reference)
    targets = {t: {"rows": [], "not_reached": []} for t in TARGETS}
    hand, points = [], {}

    for system in systems:
        for build in system.builds:
            if build.n != suite.n or CONTROL.search(build.config_id):
                continue

            picks = picks_for(res, system, build)

            if not picks:
                continue

            name = label(system.name, system.variant, build.config_id)
            points[name] = measured_points(res, system, build)

            for t, (p, note) in picks.items():
                if p is None:
                    targets[t]["not_reached"].append({"label": name, "system": system.name,
                                                      "variant": system.variant, "note": note})
                else:
                    targets[t]["rows"].append(summarize(res, systems, system, build, p.params, note))

            cells = [c for c in system.cold if c.build is build and c.split == "test" and c.regime == "c0"]

            if cells and all(p is None for p, _ in picks.values()):
                note = "a cell of the suite, which reaches no target"
                hand += [summarize(res, systems, system, build, c.params, note) for c in cells]

    for t, entry in targets.items():
        rows = entry["rows"]
        base = next((r for r in rows if ours is not None and r["system"] == ours.name
                     and r["variant"] == ours.variant), None)
        base_exact = at_exactly(points.get(base["label"], []), t) if base else None

        for r in rows:
            r["c0_exact"] = at_exactly(points.get(r["label"], []), t)

            if base is None or r is base:
                continue

            r["against"] = {
                "c0_p50": r["c0_p50_us"] / base["c0_p50_us"] if r["c0_p50_us"] and base["c0_p50_us"] else None,
                "c0_paired": paired(r["_trials"], base["_trials"]),
                "c0_exact": r["c0_exact"]["p50_us"] / base_exact["p50_us"] if r["c0_exact"] and base_exact else None,
                "warm_p50": r["warm_p50_us"] / base["warm_p50_us"]
                if r["warm_p50_us"] and base["warm_p50_us"] else None,
                "qps6": r["qps"][6] / base["qps"][6] if r["qps"][6] and base["qps"][6] else None,
                "cold_memory": r["cold_total"] / base["cold_total"] if r["cold_total"] and base["cold_total"] else None,
                "warm_memory": r["warm_total"] / base["warm_total"] if r["warm_total"] and base["warm_total"] else None,
            }

        rows.sort(key=lambda r: (r["c0_p50_us"] is None, r["c0_p50_us"] or 0.0))
        entry["reference"] = base["label"] if base else None

    return {"targets": targets, "hand": hand, "controls": controls(suite, res)}


def controls(suite: driver.Suite, res: Results) -> list[dict]:
    """The control builds' cells and warm lines: ColdIVF's cell in each rival's batch, the fresh-clone cells and the
    warm lines measured again with the swap check."""
    out = []

    for system in suite.systems:
        for build in system.builds:
            if not CONTROL.search(build.config_id):
                continue

            for c in system.cold:
                if c.build is not build:
                    continue

                trials = [x for x in res.cold(system, build, c.params, c.split) if x.get("status") == "ok"]
                total = [x["total_us"] for x in trials]
                out.append({"kind": "cold", "label": label(system.name, system.variant, build.config_id),
                            "config_id": build.config_id, "group": c.group, "setting": dict(c.params),
                            "fresh_clone": c.fresh_clone, "trials": len(total), "c0_p50_us": median(total),
                            "c0_p99_us": pct(total, 99),
                            "status": dict(Counter(x.get("status") for x in res.cold(system, build, c.params,
                                                                                     c.split)))})

            for w in system.warm:
                if w.build is not build:
                    continue

                for s in w.settings:
                    line = res.warm(system, build, s, w.split, w.threads)

                    if line is not None:
                        out.append({"kind": "warm", "label": label(system.name, system.variant, build.config_id),
                                    "config_id": build.config_id, "setting": dict(s), "threads": w.threads,
                                    "p50_us": line.get("p50_us"), "qps": line.get("qps"),
                                    "recall10": line.get("recall10"), "status": line.get("status"),
                                    "swapins_delta": line.get("swapins_delta"),
                                    "compressions_delta": line.get("compressions_delta")})

    return out


def fmt(x: float | None, digits: int = 2, scale: float = 1.0, comma: bool = False) -> str:
    if x is None:
        return "-"

    return f"{x / scale:,.{digits}f}" if comma else f"{x / scale:.{digits}f}"


def fmt_ms(us: float | None) -> str:
    """Microseconds as milliseconds with ~3 significant digits, eg. "0.332", "10.4" or "4,425"."""
    if us is None:
        return "-"

    ms = us / 1000
    return f"{ms:,.0f}" if ms >= 100 else f"{ms:.1f}" if ms >= 10 else f"{ms:.2f}" if ms >= 1 else f"{ms:.3f}"


def fmt_size(b: float | None) -> str:
    """Bytes in decimal units with ~3 significant digits, eg. "48 MB" or "9.45 GB"."""
    if b is None:
        return "-"

    for unit, size in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if b >= size:
            v = b / size
            return f"{v:.3g} {unit}" if v < 100 else f"{v:,.0f} {unit}"

    return f"{b:.0f} B"


def fmt_ratio(against: dict | None, name: str) -> str:
    value = (against or {}).get(name)
    return f"{value:.2f}x" if value is not None else "-"


SPEED = ("| System | Setting | Test Recall@10 | Warm p50 ms | QPS on 6 Threads | QPS on 12 Threads | QPS Method "
         "| C0 p50 ms | C0 p99 ms | C0 Trials (Queries) | C0 p50 at Exactly the Target ms "
         "| C0 Against ColdIVF, Paired by Query |")
MEMORY = ("| System | Setting | Open and First Query: Process + Index Pages | Warm Mode | Warm: Process + Cached Pages "
          "| Build Minutes | Build Peak | Build Malloc Cache | Index Size | Bytes per Vector |")


def rule(header: str) -> str:
    """The line under a table's header, one cell per column."""
    return "|" + "---|" * (header.count("|") - 1)


def method_text(r: dict) -> str:
    """How a row's lines on 6 and 12 threads ran their queries side by side, once when both ran the same way."""
    m = r.get("qps_method") or {}
    six, twelve = m.get(6), m.get(12)

    if six == twelve or twelve is None:
        return six or "-"

    return f"{twelve} on 12" if six is None else f"{six} on 6, {twelve} on 12"


def speed_line(r: dict) -> str:
    flagged = f" (warm {', '.join(f'{t} thread' + ('s' if t > 1 else '') for t in r['warm_invalid'])} invalid)" \
        if r["warm_invalid"] else ""
    p = (r.get("against") or {}).get("c0_paired")
    pair = f"{p['median']:.2f}x ({p['low']:.2f} to {p['high']:.2f}, {p['queries']} queries)" if p else "-"
    exact = fmt_ms(r["c0_exact"]["p50_us"]) if r.get("c0_exact") else "-"
    return (f"| {r['label']}{flagged} | {r['setting_text']} | {fmt(r['recall10'], 4)} | {fmt_ms(r['warm_p50_us'])} "
            f"| {fmt(r['qps'][6], 0, comma=True)} | {fmt(r['qps'][12], 0, comma=True)} | {method_text(r)} "
            f"| {fmt_ms(r['c0_p50_us'])} | {fmt_ms(r['c0_p99_us'])} | {r['trials']} ({r['queries']}) | {exact} | {pair} |")


def memory_line(r: dict) -> str:
    shared = " (another setup's file)" if r["shared_file"] else ""
    cold = f"{fmt_size(r['cold_process'])} + {fmt_size(r['cold_pages'])} = {fmt_size(r['cold_total'])}"
    warm = f"{fmt_size(r['warm_footprint'])} + {fmt_size(r['warm_cached'])} = {fmt_size(r['warm_total'])}"
    return (f"| {r['label']} | {r['setting_text']} | {cold} | {r['warm_mode'] or '-'} | {warm} "
            f"| {fmt(r['build_s'], 1, 60)}{shared} | {fmt_size(r['build_peak'])} | {r.get('build_malloc_cache') or '-'} "
            f"| {fmt_size(r['index_bytes'])} | {fmt(r['bytes_per_vector'], 0, comma=True)} |")


def markdown(data: dict) -> str:
    out = []

    for t, entry in data["targets"].items():
        out += [f"\n### Recall@10 {t:.2f}\n", SPEED, rule(SPEED)]
        out += [speed_line(r) for r in entry["rows"]]
        out += [f"| {m['label']} | {m['note']} |" + " |" * (SPEED.count("|") - 3) for m in entry["not_reached"]]
        out += ["", MEMORY, rule(MEMORY)]
        out += [memory_line(r) for r in entry["rows"]]

    if data["hand"]:
        out += ["\n### Systems That Reach No Target, at Their Cells\n", SPEED, rule(SPEED)]
        out += [speed_line(r) for r in data["hand"]]
        out += ["", MEMORY, rule(MEMORY)]
        out += [memory_line(r) for r in data["hand"]]

    cold = [c for c in data["controls"] if c["kind"] == "cold"]

    if cold:
        out += ["\n### Control Cells\n", "| Build | Group | Setting | Fresh Clone | Trials | C0 p50 ms | C0 p99 ms |",
                "|---|---|---|---|---|---|---|"]
        out += [f"| {c['label']} {c['config_id']} | {c['group']} | {setting_text(c['setting'])} "
                f"| {'yes' if c['fresh_clone'] else 'no'} | {c['trials']} | {fmt_ms(c['c0_p50_us'])} "
                f"| {fmt_ms(c['c0_p99_us'])} |" for c in cold]

    warm = [c for c in data["controls"] if c["kind"] == "warm"]

    if warm:
        out += ["\n### Warm Lines Measured Again\n",
                "| Build | Setting | Threads | p50 ms | QPS | Recall@10 | Status | Swap-Ins | Compressions |",
                "|---|---|---|---|---|---|---|---|---|"]
        out += [f"| {c['label']} {c['config_id']} | {setting_text(c['setting'])} | {c['threads']} "
                f"| {fmt_ms(c['p50_us'])} | {fmt(c['qps'], 0, comma=True)} | {fmt(c['recall10'], 4)} | {c['status']} "
                f"| {c['swapins_delta']} | {c['compressions_delta']} |" for c in warm]

    return "\n".join(out) + "\n"


def public(data: dict) -> dict:
    """`data` without the trials each row keeps for its comparisons, for the JSON file."""
    def clean(r: dict) -> dict:
        return {k: v for k, v in r.items() if not k.startswith("_")}

    return {"targets": {f"{t:.2f}": {**e, "rows": [clean(r) for r in e["rows"]]} for t, e in data["targets"].items()},
            "hand": [clean(r) for r in data["hand"]], "controls": data["controls"]}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--suite", type=Path, default=SUITE)
    p.add_argument("--reference", default=REFERENCE, help="the ColdIVF system the others are set against")
    p.add_argument("--json", type=Path)
    p.add_argument("--md", type=Path)
    args = p.parse_args(argv)
    suite = driver.load_suite(args.suite)
    data = collect(suite, Results(suite.results_root / suite.phase), args.reference)
    text = markdown(data)
    print(text)

    if args.md:
        args.md.write_text(text)

    if args.json:
        args.json.write_text(json.dumps(public(data), indent=1) + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
