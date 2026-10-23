"""The write benchmark's tables, from a write suite's result lines: what single writes and durable writes cost, cold and
warm searches at each state, maintenance and the searches after it, and the checks that deleted vectors stay gone and
inserted ones are found.

A build's states are its lines' states, eg. s0 to s10 on the joint ladder, s1i to s10i with inserts only, s10p and the
states after each maintenance action, eg. s10-fold, and its fresh builds are the builds named after it, eg. tl320-fresh
for tl320. Its pick for a recall@10 target is its cheapest S0 setting that reaches the target, i.e. the one of least S0
recall at or above it, as the head-to-head picks each setting.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import driver
import headtohead as h2h

SUITE = Path(__file__).resolve().parent.parent / "suites" / "bench-1m-writes.toml"
TARGETS = (0.95, 0.90)
REFERENCE = ("coldivf", "python")
# The states of the search tables, in their order, and those the maintenance table sets beside each action.
STATES = ("s0", "s1", "s5", "s10", "s10i", "fresh")
FRESH = ("fresh", "fresh-i")
# Names that the head-to-head's tables don't have, by system, variant and build.
NAMES = {("turbovec", "idmap", "b4-cal"): "turbovec 4-bit, calibrated"}


def name_of(system: str, variant: str, config_id: str) -> str:
    return NAMES.get((system, variant, config_id)) or h2h.label(system, variant, config_id)


def ms(us: float | None) -> str:
    return h2h.fmt_ms(us)


def p50(values: list[float]) -> float | None:
    return h2h.median([v for v in values if v is not None])


class Lines:
    """A write suite's ok and invalid lines, by build. A build is a system, its variant and a config id that isn't a
    fresh build's, and a fresh build's lines join the build it was named after."""

    def __init__(self, lines: list[dict]) -> None:
        self.by_build: dict[tuple, list[dict]] = defaultdict(list)

        for x in lines:
            if x.get("status") not in ("ok", "invalid"):
                continue

            config = x.get("config_id") or ""

            for suffix in ("-fresh-i", "-fresh"):
                if config.endswith(suffix):
                    config = config[: -len(suffix)]
                    break

            self.by_build[(x.get("system"), x.get("variant") or "", config)].append(x)

    def builds(self) -> list[tuple]:
        """Every build with a warm line at S0, ColdIVF's first."""
        found = [b for b, xs in self.by_build.items()
                 if any(x["kind"] == "warm" and x.get("state") == "s0" for x in xs)]
        return sorted(found, key=lambda b: (b[:2] != REFERENCE, name_of(*b)))

    def of(self, build: tuple, kind: str, **match) -> list[dict]:
        return [x for x in self.by_build[build] if x.get("kind") == kind
                and all(x.get(k) == v for k, v in match.items())]

    def warm(self, build: tuple, state: str, params: dict) -> dict | None:
        """The latest ok warm line of one thread on the test split, else the latest invalid one, marked by "invalid"."""
        found = [x for x in self.of(build, "warm", state=state, split="test", threads=1)
                 if h2h.key(x.get("search_params")) == h2h.key(params)]
        ok = [x for x in found if x["status"] == "ok"]

        if ok:
            return ok[-1]

        return dict(found[-1], invalid=True) if found else None

    def cold(self, build: tuple, state: str, params: dict) -> list[dict]:
        """The ok C0 trials on the test split."""
        return [x for x in self.of(build, "cold", state=state, split="test", regime="c0", status="ok")
                if x.get("total_us") is not None and h2h.key(x.get("search_params")) == h2h.key(params)]

    def write(self, build: tuple, mode: str, op: str, **match) -> dict | None:
        found = [x for x in self.of(build, "write", mode=mode, op=op, status="ok", **match)]
        return found[-1] if found else None

    def maintains(self, build: tuple) -> list[dict]:
        return [x for x in self.of(build, "maintain", status="ok")]

    def picks(self, build: tuple) -> dict[float, dict | None]:
        """The build's setting for each target: its S0 setting of least recall at or above the target."""
        settings = {}

        for x in self.of(build, "warm", state="s0", split="test", threads=1):
            if x.get("recall10") is not None:
                settings[h2h.key(x.get("search_params"))] = (x["recall10"], x.get("search_params") or {})

        out = {}

        for t in TARGETS:
            reach = [v for v in settings.values() if v[0] >= t]
            out[t] = min(reach, key=lambda v: v[0])[1] if reach else None

        return out


def cold_cell(trials: list[dict]) -> dict:
    total = [x["total_us"] for x in trials]
    return {"trials": len(trials), "p50_us": p50(total), "p99_us": h2h.pct(total, 99),
            "open_p50_us": p50([x.get("open_us") for x in trials]),
            "search_p50_us": p50([x.get("search_us") for x in trials]),
            "recall10": statistics.mean(x["recall10"] for x in trials if x.get("recall10") is not None)
            if any(x.get("recall10") is not None for x in trials) else None,
            "disk_p50": p50([x.get("disk_bytes") for x in trials]),
            "footprint_p50": p50([x.get("footprint_delta") for x in trials])}


def warm_cell(line: dict | None) -> dict:
    if line is None:
        return {}

    return {"p50_us": line.get("p50_us"), "p99_us": line.get("p99_us"), "recall10": line.get("recall10"),
            "qps": line.get("qps"), "dead_returned": line.get("dead_returned"),
            "recall10_inserted": line.get("recall10_inserted"), "invalid": bool(line.get("invalid")),
            "findability": (line.get("extra") or {}).get("findability")}


def search_rows(lines: Lines, target: float) -> list[dict]:
    """For each build with a pick for `target`, its cold and warm cells at every state it has."""
    rows = []

    for b in lines.builds():
        params = lines.picks(b)[target]

        if params is None:
            continue

        states = sorted({x.get("state") for x in lines.by_build[b] if x["kind"] in ("warm", "cold")} - {None},
                        key=state_order)
        rows.append({"build": b, "label": name_of(*b), "setting": params, "setting_text": h2h.setting_text(params),
                     "cold": {s: cold_cell(lines.cold(b, s, params)) for s in states if lines.cold(b, s, params)},
                     "warm": {s: warm_cell(lines.warm(b, s, params)) for s in states if lines.warm(b, s, params)}})

    return rows


def state_order(state: str) -> tuple:
    """S0, the joint ladder, the inserts-only ladder, S10p, the states after maintenance, then the fresh builds."""
    ladder = ["s0", "s1", "s5", "s10", "s1i", "s5i", "s10i", "s10p"]

    if state in ladder:
        return (0, ladder.index(state), "")

    return (2, 0, state) if state in FRESH else (1, 0, state)


def write_rows(lines: Lines) -> list[dict]:
    """For each build that writes, its single writes at S0 and S10, its durable ones, and its batches of the stream."""
    rows = []

    for b in lines.builds():
        if not lines.of(b, "write"):
            continue

        at = {}

        for state in ("s0", "s10"):
            ops = {op: lines.write(b, "latency", op, state=state)
                   for op in ("insert", "delete", "insert_sync", "sync", "durable_point")}
            at[state] = {op: {k: x.get(k) for k in ("p50_us", "p99_us", "calls", "rows", "durability", "rows_per_s",
                                                      "logical_bytes_written", "disk_bytes_written")}
                         for op, x in ops.items() if x is not None}

        stream = {op: lines.write(b, "stream", op, state_from="s0", state_to="s1") for op in ("insert", "delete")}
        rows.append({"build": b, "label": name_of(*b), "latency": at,
                     "stream": {op: {k: x.get(k) for k in ("p50_us", "p99_us", "rows_per_s", "durability", "batch")}
                                for op, x in stream.items() if x is not None}})

    return rows


def maintenance_rows(lines: Lines) -> list[dict]:
    """For each maintenance action, its cost and the searches on the state it left, beside S10 and the fresh build, at
    the build's pick for the first target it has a pick for."""
    rows = []

    for b in lines.builds():
        picks = lines.picks(b)
        target = next((t for t in TARGETS if picks[t] is not None), None)

        for x in lines.maintains(b):
            if x.get("state_from") != "s10":
                continue

            after = x.get("state_to")
            params = picks[target] if target is not None else None
            cells = {s: {"cold": cold_cell(lines.cold(b, s, params)) if params is not None and lines.cold(b, s, params)
                         else None, "warm": warm_cell(lines.warm(b, s, params)) if params is not None else {}}
                     for s in ("s10", after, "fresh")}
            rows.append({"build": b, "label": name_of(*b), "action": x.get("action"), "target": target,
                         "setting_text": h2h.setting_text(params) if params is not None else None,
                         "seconds": x.get("seconds"), "cpu_seconds": x.get("cpu_seconds"),
                         "threads": x.get("threads"), "peak_footprint": x.get("peak_footprint"),
                         "bytes_read": x.get("bytes_read"), "bytes_written": x.get("bytes_written"),
                         "report": (x.get("extra") or {}).get("report"), "state": after, "cells": cells})

    return rows


def policy_rows(lines: Lines) -> list[dict]:
    """The S10p pass of each build that has one: its runs, and its rows per second with them counted in."""
    rows = []

    for b in lines.builds():
        runs = [x for x in lines.maintains(b) if x.get("state_from") == "s10p"]
        stream = [x for x in lines.of(b, "write", mode="stream", status="ok", state_to="s10p")
                  if x.get("op") in ("insert", "delete") and x.get("rows_per_s")]

        if not runs or not stream:
            continue

        rows_done = sum(x["rows"] for x in stream)
        write_s = sum(x["rows"] / x["rows_per_s"] for x in stream)
        run_s = sum(x.get("seconds") or 0 for x in runs)
        rows.append({"build": b, "label": name_of(*b), "runs": len(runs),
                     "kinds": [((x.get("extra") or {}).get("report") or {}).get("kind") for x in runs],
                     "run_seconds": run_s, "write_seconds": write_s, "rows": rows_done,
                     "rows_per_s": rows_done / write_s, "rows_per_s_with_runs": rows_done / (write_s + run_s)})

    return rows


def check_rows(lines: Lines) -> list[dict]:
    """For each build that writes: the most deleted ids that any warm search returned per query, and at each state
    with a findability check, the share of inserted and base probes found, with the recall of the inserts."""
    rows = []

    for b in lines.builds():
        warm = [x for x in lines.of(b, "warm", split="test", threads=1) if x.get("state") != "s0"]

        if not lines.of(b, "write") or not warm:
            continue

        dead = [x.get("dead_returned") for x in warm if x.get("dead_returned") is not None]
        found = {}

        for x in warm:
            f = (x.get("extra") or {}).get("findability")

            if isinstance(f, str):
                try:
                    f = json.loads(f.replace("'", '"'))
                except ValueError:
                    f = None

            if isinstance(f, dict) and x.get("state") not in found:
                found[x["state"]] = {"inserted": f.get("inserted"), "base": f.get("base"),
                                     "recall10_inserted": x.get("recall10_inserted")}

        rows.append({"build": b, "label": name_of(*b), "dead_returned_max": max(dead) if dead else None,
                     "findability": dict(sorted(found.items(), key=lambda kv: state_order(kv[0])))})

    return rows


def tables(lines: Lines) -> dict:
    return {"writes": write_rows(lines), "search": {f"{t:.2f}": search_rows(lines, t) for t in TARGETS},
            "maintenance": maintenance_rows(lines), "policy": policy_rows(lines), "checks": check_rows(lines)}


def ratio(a: float | None, b: float | None) -> str:
    return f"{a / b:.2f}x" if a is not None and b else "-"


def write_table(rows: list[dict]) -> list[str]:
    out = ["| System | Durability | Insert p50 | Insert p99 | Delete p50 | Delete p99 | Insert and Sync p50 "
           "| Durable Point p50 | Batch of 1,000: Inserts per Second | Deletes per Second |",
           "|---|---|---|---|---|---|---|---|---|---|"]

    for r in rows:
        s0 = r["latency"].get("s0", {})
        ins, dele = s0.get("insert", {}), s0.get("delete", {})
        synced, point = s0.get("insert_sync", {}), s0.get("durable_point", {})
        stream = r["stream"]
        out.append(f"| {r['label']} | {ins.get('durability') or '-'} | {us(ins.get('p50_us'))} "
                   f"| {us(ins.get('p99_us'))} | {us(dele.get('p50_us'))} | {us(dele.get('p99_us'))} "
                   f"| {us(synced.get('p50_us'))} | {us(point.get('p50_us'))} "
                   f"| {h2h.fmt((stream.get('insert') or {}).get('rows_per_s'), 0, comma=True)} "
                   f"| {h2h.fmt((stream.get('delete') or {}).get('rows_per_s'), 0, comma=True)} |")

    return out


def us(value: float | None) -> str:
    """Microseconds below 1 ms, else milliseconds, with ~3 significant digits."""
    if value is None:
        return "-"

    return f"{value:.1f} µs" if value < 10 else f"{value:.0f} µs" if value < 1000 else f"{ms(value)} ms"


def search_tables(rows: list[dict], target: str) -> list[str]:
    head = " | ".join(s.upper() if s != "fresh" else "Fresh Build" for s in STATES)
    out = [f"**Recall@10 {target}, cold p50 in ms, then recall@10**", "",
           f"| System | Setting | {head} | S10 Against S0 |", "|---|---|" + "---|" * len(STATES) + "---|"]

    for r in rows:
        cells = [r["cold"].get(s) for s in STATES]
        text = [f"{ms(c['p50_us'])} ({h2h.fmt(warm_recall(r, s), 3)})" if c else "-" for c, s in zip(cells, STATES)]
        s0, s10 = r["cold"].get("s0"), r["cold"].get("s10")
        out.append(f"| {r['label']} | {r['setting_text']} | {' | '.join(text)} "
                   f"| {ratio((s10 or {}).get('p50_us'), (s0 or {}).get('p50_us'))} |")

    out += ["", f"**Recall@10 {target}, warm p50 in ms, then recall@10**", "",
            f"| System | Setting | {head} |", "|---|---|" + "---|" * len(STATES)]

    for r in rows:
        cells = [r["warm"].get(s) for s in STATES]
        text = [f"{ms(c['p50_us'])} ({h2h.fmt(c.get('recall10'), 3)}){' invalid' if c.get('invalid') else ''}"
                if c else "-" for c in cells]
        out.append(f"| {r['label']} | {r['setting_text']} | {' | '.join(text)} |")

    return out


def warm_recall(row: dict, state: str) -> float | None:
    """A state's recall@10 from its warm line, over every test query, else from its cold trials."""
    w = row["warm"].get(state) or {}
    return w.get("recall10") if w.get("recall10") is not None else (row["cold"].get(state) or {}).get("recall10")


def maintenance_table(rows: list[dict]) -> list[str]:
    out = ["| System | Action | Threads | Time | Processor Time | Peak Memory | Bytes Written | Cold p50 at S10, After "
           "and Fresh | Warm p50 at S10, After and Fresh | Recall@10 at S10, After and Fresh |",
           "|---|---|---|---|---|---|---|---|---|---|"]

    for r in rows:
        trio = [r["cells"][s] for s in ("s10", r["state"], "fresh")]
        cold = " / ".join(ms((c["cold"] or {}).get("p50_us")) for c in trio)
        warm = " / ".join(ms((c["warm"] or {}).get("p50_us")) for c in trio)
        recall = " / ".join(h2h.fmt((c["warm"] or {}).get("recall10"), 3) for c in trio)
        out.append(f"| {r['label']} | {r['action']} | {r['threads'] or '-'} | {h2h.fmt(r['seconds'], 1)} s "
                   f"| {h2h.fmt(r['cpu_seconds'], 1)} s | {h2h.fmt_size(r['peak_footprint'])} "
                   f"| {h2h.fmt_size(r['bytes_written'])} | {cold} | {warm} | {recall} |")

    return out


def markdown(data: dict) -> str:
    out = ["## Writes", "", *write_table(data["writes"]), ""]

    for t, rows in data["search"].items():
        out += ["## Search at Each State", "", *search_tables(rows, t), ""]

    out += ["## Maintenance", "", *maintenance_table(data["maintenance"]), ""]

    if data["policy"]:
        out += ["| System | Runs | Kinds | Run Time | Rows per Second | With the Runs |", "|---|---|---|---|---|---|"]
        out += [f"| {r['label']} | {r['runs']} | {', '.join(k or '-' for k in r['kinds'])} "
                f"| {h2h.fmt(r['run_seconds'], 1)} s | {h2h.fmt(r['rows_per_s'], 0, comma=True)} "
                f"| {h2h.fmt(r['rows_per_s_with_runs'], 0, comma=True)} |" for r in data["policy"]]
        out.append("")

    out += ["## Checks", "",
            "| System | Most Deleted Ids per Query | Findability: Inserted, Base and Inserts' Recall@10 |",
            "|---|---|---|"]

    for r in data["checks"]:
        found = "; ".join(f"{s.upper()} {h2h.fmt(f['inserted'], 3)}, {h2h.fmt(f['base'], 3)}, "
                          f"{h2h.fmt(f['recall10_inserted'], 3)}" for s, f in r["findability"].items())
        out.append(f"| {r['label']} | {h2h.fmt(r['dead_returned_max'], 2)} | {found or '-'} |")

    return "\n".join(out) + "\n"


def public(data: dict) -> dict:
    """`data` with each build as a list, for the JSON file."""
    def clean(x):
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}

        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]

        return x

    return clean(data)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--suite", type=Path, default=SUITE)
    p.add_argument("--results", type=Path, help="the suite's results folder, by default its results root and phase")
    p.add_argument("--json", type=Path)
    p.add_argument("--md", type=Path)
    args = p.parse_args(argv)
    suite = driver.load_suite(args.suite)
    data = tables(Lines(h2h.Results(args.results or suite.results_root / suite.phase).lines))
    text = markdown(data)
    print(text)

    if args.md:
        args.md.write_text(text)

    if args.json:
        args.json.write_text(json.dumps(public(data), indent=1) + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
