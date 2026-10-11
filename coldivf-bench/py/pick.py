"""Picks a system's cold settings from its results, the same way for every system.

For each recall target, the cheapest settings are those with the lowest warm p50 at 1 thread whose recall@10 on the
tune split reaches the target, and a tie goes to fewer bytes scanned. `screen` writes true-cold (C0) cells that time
the `count` cheapest of them on the tune split. `verdict` then takes, for each target, the one of them with the lowest
C0 p50 over its tune trials, and writes cells that time it on the test split. A cheaper one whose 95% interval for the
C0 median overlaps that one's is as fast cold within the trials' noise, so the cheapest such setting is taken instead,
eg. for a system that loads its whole index at open, whose C0 time hardly depends on the setting. It also writes a cell for the cheaper
setting just below the target, the one with the most recall below it that is no slower warm, so that a report can
interpolate latency to exactly the target. A target no setting reaches becomes a comment that names the best recall
the grid found. A setting whose every tune line is invalid ranks by its latest invalid line, since the driver marks a
warm line invalid only when the machine's swap-ins or compressions moved during its timed passes, which a long setting
of a slow system can rarely avoid on this machine, and its recall and its p50 still rank it. Its comment says so.

    python pick.py screen SUITE --system NAME [--variant V] --config ID [--targets 0.90,0.95,0.99] [--count 3]
                          [--trials 30] [--dummy-n N] [--threads T] [--group G] [--write]
    python pick.py verdict SUITE --system NAME [--variant V] --config ID [--targets 0.90,0.95,0.99] [--count 3]
                           [--trials 100] [--min-trials 20] [--no-below] [--warm] [--qps-threads 6,12]
                           [--dummy-n N | --dummy PATH] [--threads T] [--group G] [--write]

The cells take the dummy, threads and group of the system's first cold entry for that build, when it has one, unless
--dummy-n, --dummy, --threads or --group says otherwise. With --warm, verdict also writes warm entries on the test
split at the settings it picks, 3 passes at 1 thread and queries per second at each of --qps-threads, in the mode of
the build's first warm entry. Without --write the entries are printed. With it they go at the end of the system's
section of the suite, after the entries it holds, and the suite must still load. Entries the suite already holds are
left out either way.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import driver
import schema

TARGETS = (0.90, 0.95, 0.99)


@dataclass
class Point:
    """One search setting of a build, with its warm numbers and the C0 times of its tune trials."""

    params: dict
    recall10: float
    p50_us: float
    bytes_scanned: float | None = None
    cold_us: list[float] = field(default_factory=list)
    # Whether it ranks by an invalid line, one made while the machine swapped or compressed.
    invalid: bool = False

    @property
    def key(self) -> str:
        return canonical(self.params)

    @property
    def cold_p50_us(self) -> float | None:
        return statistics.median(self.cold_us) if self.cold_us else None


@dataclass
class Pick:
    """What one target gets, the cells to write and the lines that say why."""

    target: float
    cells: list[tuple[Point, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def canonical(params: dict) -> str:
    return json.dumps(params, sort_keys=True)


def find_system(suite: driver.Suite, name: str, variant: str) -> driver.System:
    for s in suite.systems:
        if s.name == name and s.variant == variant:
            return s

    raise SystemExit(f"the suite has no system {name} with variant {variant!r}")


def find_build(system: driver.System, config_id: str) -> driver.Build:
    for b in system.builds:
        if b.config_id == config_id:
            return b

    raise SystemExit(f"{system.name} has no build {config_id}")


def mine(line: dict, system: driver.System, build: driver.Build) -> bool:
    return (line.get("status") == "ok" and line.get("system") == system.name
            and (line.get("variant") or "") == system.variant and line.get("config_id") == build.config_id
            and line.get("n") == build.n)


def flagged(line: dict, system: driver.System, build: driver.Build) -> bool:
    """Whether the line is the build's and invalid, ie. measured while the machine swapped or compressed."""
    return line.get("status") == "invalid" and mine(line | {"status": "ok"}, system, build)


def points(lines: list[dict], system: driver.System, build: driver.Build) -> list[Point]:
    """The build's warm settings at 1 thread on the tune split, the latest ok line of each, or else its latest invalid
    line, with the C0 times of every tune trial at that setting."""
    out: dict[str, Point] = {}
    invalid: dict[str, Point] = {}

    for x in lines:
        if (x.get("kind") == "warm" and x.get("split") == "tune" and x.get("threads") == 1
                and x.get("recall10") is not None and x.get("p50_us") is not None):
            p = Point(dict(x.get("search_params") or {}), float(x["recall10"]), float(x["p50_us"]),
                      x.get("bytes_scanned"))

            if mine(x, system, build):
                out[p.key] = p
            elif flagged(x, system, build):
                p.invalid = True
                invalid[p.key] = p

    for k, p in invalid.items():
        out.setdefault(k, p)

    for x in lines:
        if (x.get("kind") == "cold" and mine(x, system, build) and x.get("split") == "tune"
                and x.get("regime") == "c0" and x.get("total_us") is not None):
            p = out.get(canonical(x.get("search_params") or {}))

            if p is not None:
                p.cold_us.append(float(x["total_us"]))

    return list(out.values())


def cost(p: Point) -> tuple[float, float]:
    return p.p50_us, p.bytes_scanned if p.bytes_scanned is not None else 0.0


def cheapest(grid: list[Point], target: float, count: int) -> list[Point]:
    return sorted((p for p in grid if p.recall10 >= target), key=cost)[:count]


def describe(p: Point) -> str:
    text = f"recall@10 {p.recall10:.4f}, warm p50 {p.p50_us:.0f} us"

    if p.cold_us:
        text += f", C0 p50 {p.cold_p50_us / 1000:.2f} ms in {len(p.cold_us)} tune trials"

    if p.invalid:
        text += ", warm from a line the machine's swapping made invalid"

    return text


def unreached(grid: list[Point], target: float) -> Pick:
    best = max(grid, key=lambda p: (p.recall10, -p.p50_us))
    return Pick(target, notes=[f"{target:.2f} not reached. The best is {canonical(best.params)}, {describe(best)}"])


def screen(grid: list[Point], targets: list[float], count: int) -> list[Pick]:
    picks = []

    for t in targets:
        found = cheapest(grid, t, count)

        if not found:
            picks.append(unreached(grid, t))
            continue

        picks.append(Pick(t, [(p, f"{t:.2f} screen {i + 1} of {len(found)}: {describe(p)}")
                              for i, p in enumerate(found)]))

    return picks


def median_interval(values: list[float]) -> tuple[float, float]:
    """The 95% interval of the median of `values` by order statistics, the k-th to the (n - k + 1)-th smallest of n,
    where k is n/2 - 0.98·sqrt(n) rounded, eg. the 10th to 21st of 30 and the 40th to 61st of 100."""
    v = sorted(values)
    k = min(max(1, round(len(v) / 2 - 0.98 * len(v) ** 0.5)), (len(v) + 1) // 2)
    return v[k - 1], v[len(v) - k]


def overlaps(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def verdict(grid: list[Point], targets: list[float], count: int, min_trials: int, below: bool) -> list[Pick]:
    picks = []

    for t in targets:
        found = cheapest(grid, t, count)

        if not found:
            picks.append(unreached(grid, t))
            continue

        timed = [p for p in found if len(p.cold_us) >= min_trials]

        if len(timed) < len(found):
            missing = [canonical(p.params) for p in found if len(p.cold_us) < min_trials]
            raise SystemExit(f"{t:.2f}: {missing} have fewer than {min_trials} C0 tune trials. Run the screen first")

        fastest = min(timed, key=lambda p: (p.cold_p50_us, cost(p)))
        chosen = min((p for p in timed if overlaps(median_interval(p.cold_us), median_interval(fastest.cold_us))),
                     key=cost)
        why = f"the fastest cold of {len(found)}" if chosen is fastest else (
            f"the cheapest of {len(found)} whose C0 median interval overlaps the fastest's, "
            f"{canonical(fastest.params)} at {fastest.cold_p50_us / 1000:.2f} ms")
        pick = Pick(t, [(chosen, f"{t:.2f} verdict, {why}: {describe(chosen)}")])
        lower = [p for p in grid if p.recall10 < t and p.p50_us <= chosen.p50_us]

        if below and lower:
            near = max(lower, key=lambda p: (p.recall10, -p.p50_us))
            pick.cells.append((near, f"{t:.2f} the cheaper point below it: {describe(near)}"))

        picks.append(pick)

    return picks


def toml_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"

    if isinstance(value, (int, float)):
        return repr(value)

    if isinstance(value, str):
        return json.dumps(value)

    raise ValueError(f"can't write {value!r} into a suite")


def inline(params: dict) -> str:
    return "{ " + ", ".join(f"{k} = {toml_value(v)}" for k, v in params.items()) + " }" if params else "{}"


@dataclass
class CellSpec:
    """The settings every cell of a pick shares."""

    config_id: str
    split: str
    trials: int
    group: str
    threads: int
    dummy: str | None = None
    dummy_n: int = 0


def cell_text(p: Point, note: str, spec: CellSpec) -> str:
    lines = [f"  # {note}", "  [[system.cold]]", f'  config_id = "{spec.config_id}"', f"  params = {inline(p.params)}",
             f'  split = "{spec.split}"', f"  trials = {spec.trials}"]

    if spec.group != "main":
        lines.append(f'  group = "{spec.group}"')

    if spec.threads != 1:
        lines.append(f"  threads = {spec.threads}")

    if spec.dummy is not None:
        lines.append(f'  dummy = "{spec.dummy}"')
    elif spec.dummy_n:
        lines.append(f"  dummy_n = {spec.dummy_n}")

    return "\n".join(lines) + "\n"


def held(system: driver.System, config_id: str, p: Point, split: str) -> bool:
    return any(c.build.config_id == config_id and c.regime == "c0" and c.split == split
               and canonical(c.params) == p.key for c in system.cold)


def block(picks: list[Pick], system: driver.System, spec: CellSpec) -> str:
    out, seen = [], set()

    for pick in picks:
        out += [f"  # {n}\n" for n in pick.notes]

        for p, note in pick.cells:
            if p.key in seen or held(system, spec.config_id, p, spec.split):
                continue

            seen.add(p.key)
            out.append(cell_text(p, note, spec))

    return "\n".join(out)


def spec_for(system: driver.System, build: driver.Build, split: str, trials: int, args) -> CellSpec:
    first = next((c for c in system.cold if c.build is build), None)
    spec = CellSpec(build.config_id, split, trials, first.group if first else "main", first.threads if first else 1)

    if first is not None and isinstance(first.dummy, driver.Build):
        spec.dummy_n = first.dummy.n
    elif first is not None and first.dummy is not None:
        spec.dummy = str(first.dummy)

    if args.group is not None:
        spec.group = args.group

    if args.threads is not None:
        spec.threads = args.threads

    if args.dummy_n is not None:
        spec.dummy, spec.dummy_n = None, args.dummy_n

    if args.dummy is not None:
        spec.dummy, spec.dummy_n = args.dummy, 0

    return spec


def warm_text(picks: list[Pick], system: driver.System, build: driver.Build, qps_threads: list[int]) -> str:
    """Warm entries on the test split at every setting the picks time cold, 3 passes at 1 thread and queries per
    second at each of `qps_threads`, leaving out settings an entry of the suite already holds."""
    first = next((w for w in system.warm if w.build is build), None)
    mode = first.mode if first is not None else "whole"
    chosen, seen = [], set()

    for pick in picks:
        for p, _ in pick.cells:
            if p.key not in seen:
                seen.add(p.key)
                chosen.append(p)

    out = []

    for threads in [1, *qps_threads]:
        held_keys = {canonical(s) for w in system.warm if w.build is build and w.split == "test"
                     and w.threads == threads and w.mode == mode for s in w.settings}
        todo = [p for p in chosen if p.key not in held_keys]

        if not todo:
            continue

        lines = ["  [[system.warm]]", f'  config_id = "{build.config_id}"', '  split = "test"']

        if mode != "whole":
            lines.append(f'  mode = "{mode}"')

        if threads == 1:
            lines += ["  passes = 3", "  warmup = 1000"]
        else:
            lines += [f"  threads = {threads}", "  qps_seconds = 10", "  qps_repeats = 3", "  warmup = 1000"]

        lines.append("  settings = [")
        lines += [f"    {inline(p.params)}," for p in todo]
        lines.append("  ]")
        out.append("\n".join(lines) + "\n")

    return "\n".join(out)


def insert(path: Path, system: driver.System, text: str) -> None:
    """Puts `text` at the end of the system's section of the suite file at `path`, then checks that the suite still
    loads before it replaces the file."""
    lines = Path(path).read_text().splitlines(keepends=True)
    starts = [i for i, line in enumerate(lines) if line.rstrip() == "[[system]]"] + [len(lines)]
    at = None

    for a, b in zip(starts, starts[1:]):
        entry = tomllib.loads("".join(lines[a:b]))["system"][0]

        if entry.get("name") == system.name and str(entry.get("variant", "")) == system.variant:
            at = b
            break

    if at is None:
        raise SystemExit(f"can't find the section of {system.name} in {path}")

    # Comments at the start of a line right above the next [[system]] describe that system, so the text goes above them.
    while at > a + 1 and (lines[at - 1].startswith("#") or not lines[at - 1].strip()):
        at -= 1

    head = "".join(lines[:at])
    head += "" if head.endswith("\n") else "\n"
    new = head + "\n" + text + ("\n" if at < len(lines) else "") + "".join(lines[at:])
    fd, tmp = tempfile.mkstemp(dir=Path(path).parent, prefix=f".{Path(path).name}.", suffix=".toml")

    try:
        with os.fdopen(fd, "w") as f:
            f.write(new)

        driver.load_suite(tmp)
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("stage", choices=("screen", "verdict"))
    p.add_argument("suite")
    p.add_argument("--system", required=True)
    p.add_argument("--variant", default="")
    p.add_argument("--config", required=True)
    p.add_argument("--targets", default=",".join(f"{t:.2f}" for t in TARGETS))
    p.add_argument("--count", type=int, default=3)
    p.add_argument("--trials", type=int)
    p.add_argument("--min-trials", type=int, default=20)
    p.add_argument("--no-below", action="store_true")
    p.add_argument("--dummy-n", type=int)
    p.add_argument("--dummy")
    p.add_argument("--warm", action="store_true")
    p.add_argument("--qps-threads", default="6,12")
    p.add_argument("--threads", type=int)
    p.add_argument("--group")
    p.add_argument("--write", action="store_true")
    args = p.parse_args(argv)
    suite = driver.load_suite(args.suite)
    system = find_system(suite, args.system, args.variant)
    build = find_build(system, args.config)
    grid = points(schema.read(suite.results_root / suite.phase / f"{system.name}.jsonl"), system, build)

    if not grid:
        raise SystemExit(f"no warm tune lines of {system.name} {args.config} in {suite.results_root / suite.phase}")

    targets = [float(t) for t in args.targets.split(",") if t.strip()]

    if args.stage == "screen":
        picks = screen(grid, targets, args.count)
        spec = spec_for(system, build, "tune", args.trials or 30, args)
    else:
        picks = verdict(grid, targets, args.count, args.min_trials, not args.no_below)
        spec = spec_for(system, build, "test", args.trials or 100, args)

    text = block(picks, system, spec)

    if args.warm and args.stage == "verdict":
        qps = [int(t) for t in args.qps_threads.split(",") if t.strip()]
        more = warm_text(picks, system, build, qps)
        text = text + "\n" + more if more else text

    if not args.write:
        print(text)
    elif "[[system.cold]]" in text or "[[system.warm]]" in text:
        insert(Path(args.suite), system, text)
        print(f"wrote to {args.suite}:\n{text}", file=sys.stderr)
    else:
        print(f"nothing new to write to {args.suite}:\n{text}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
