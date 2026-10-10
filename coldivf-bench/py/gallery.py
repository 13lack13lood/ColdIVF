"""Other ways to draw the numbers behind the README's charts, for picking the ones it keeps.

    python gallery.py [--data FILE] [--quiet DIR] [--repo DIR] [--out DIR]

Each drawing reads chart-data.json, which charts.py writes. The ones that show every cold trial or the stages of
ColdIVF's first query also read the quiet run's lines. Each is drawn for a light and a dark page, as <name>.svg and
<name>-dark.svg in docs/charts/gallery.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from charts import (DATASETS, FAMILIES, LONE_MARKERS, MARGIN, NAMES, OUT, PALETTE, PANELS, QUIET, REPO, THEMES,
                    Page, _grid, _log_ticks, _mpl, _rc, busy_note, fmt_bytes, fmt_ms, fmt_range, interpolate,
                    millions, percentile, read_lines, setting_key, text_width, wrap)

GALLERY = OUT / "gallery"

# The recall@10 that the warm comparisons across systems read each grid at.
WARM_TARGET = 0.90

# The recall@10 targets of the warm leaderboard.
LEADER_TARGETS = (0.85, 0.90, 0.95)


def fmt_time(us: float) -> str:
    """Microseconds in the unit that suits them, eg. "61 µs", "6.26 ms" or "1.84 s"."""
    if us < 1000:
        return f"{us:.0f} µs"

    return fmt_ms(us) if us < 1e6 else f"{us / 1e6:.2f} s"


def fmt_tick(us: float) -> str:
    """A time axis's tick, eg. "100 µs", "10 ms" or "1 s"."""
    if us < 1000:
        return f"{us:g} µs"

    return f"{us / 1000:g} ms" if us < 1e6 else f"{us / 1e6:g} s"


def fmt_ratio(r: float) -> str:
    """A ratio with 2 or 3 significant digits, eg. "0.27×", "3.5×", "104×" or "27,000×"."""
    if r >= 100:
        return f"{round(r, 2 - math.floor(math.log10(r))):,.0f}×"

    if r >= 10:
        return f"{r:,.0f}×"

    return f"{r:.1f}×" if r >= 1 else f"{r:.2f}×"


def fmt_seconds(s: float) -> str:
    return f"{s:.3g} s"


def fmt_per_vector(b: float) -> str:
    return f"{b:,.0f} B"


# The numbers the drawings across metrics compare, each lower-is-better: its key, name, what it is and its format.
METRICS = (
    ("cold", "Cold first query", "at its smoke setting", fmt_time),
    ("warm", "Warm query", f"at recall@10 {WARM_TARGET:.2f}", fmt_time),
    ("cold_mem", "Cold memory", "open and first query", fmt_bytes),
    ("warm_mem", "Warm memory", "process + cached pages", fmt_bytes),
    ("build_mem", "Build memory", "peak footprint", fmt_bytes),
    ("build_s", "Build time", "on 4 threads", fmt_seconds),
    ("disk", "Disk", "bytes per vector", fmt_per_vector),
)

# The stages of ColdIVF's cold first query in its cold binary's trace, in the order they run: the name, the fields it
# sums and whether it reads the index or computes.
STAGES = (
    ("Open the index file", ("open_us",), "read"),
    ("Make the buffers, start the table read", ("prepare_us",), "compute"),
    ("Load the cluster tables", ("tables_us",), "read"),
    ("Make the cluster buffer", ("buffers_us",), "compute"),
    ("Route to the clusters", ("route_us",), "compute"),
    ("Submit the cluster reads", ("submit_us",), "read"),
    ("Wait for the cluster reads", ("wave_us",), "read"),
    ("Scan the 1-bit codes", ("scan_us",), "compute"),
    ("Read the rerank rows", ("rows_us",), "read"),
    ("Rerank", ("rerank_us",), "compute"),
)


def by_id(data: dict) -> dict[str, dict]:
    return {c["id"]: c for c in data["charts"]}


def at_recall(points: list[dict], target: float) -> tuple[float, bool] | None:
    """A warm grid's p50 at recall@10 `target`, and whether it lies between two of its settings. Inside the grid it is
    interpolated as charts.py does, linear in the log of the latency. Where every setting is above `target`, it is
    the fastest of them. It is None when no setting reaches `target`."""
    between = interpolate(points, target)

    if between is not None:
        return between, True

    reach = [p["p50_us"] for p in points if (p.get("recall10") or 0) >= target]
    return (min(reach), False) if reach else None


def smoke_rows(data: dict) -> list[dict]:
    """Each system of the smoke test with the numbers that the drawings across metrics compare, None where a system
    has none. Every number is from the smoke test's dataset."""
    c = by_id(data)
    curves = {cu["system"]: cu for cu in c["warm-recall"]["curves"]}
    cold = {p["system"]: p for p in c["cold-smoke"]["points"]}
    memory = {r["system"]: r for r in c["memory"]["rows"]}
    rows = []

    for system in [s for s in PANELS if s in cold] + [s for s in cold if s not in PANELS]:
        p, mem = cold[system], memory.get(system) or {}
        points = (curves.get(system) or {}).get("points") or []
        same = next((q for q in points if q["setting"] == p["setting"]), None)
        warm = at_recall(points, WARM_TARGET)
        calls = [q["overhead_us"] for q in points if q.get("overhead_us")]
        w, build, disk = mem.get("warm") or {}, mem.get("build") or {}, mem.get("disk") or {}
        rows.append({
            "system": system, "name": NAMES.get(system, system), "label": p["label"], "threads": p["threads"],
            "recall": p["recall10_warm"], "cold": p["p50_us"], "cold_trials": p["trials"],
            "warm_same": same["p50_us"] if same else None,
            "warm": warm[0] if warm else None, "warm_between": warm[1] if warm else None,
            "best_recall": max((q["recall10"] for q in points), default=None),
            "calls": calls,
            # The binding's time for one call, when it is shorter than every search of the grid, as charts.py takes it.
            "call_us": max(calls) if calls and max(calls) < min(q["p50_us"] for q in points) else None,
            "cold_mem": ((mem.get("cold") or {}).get("footprint_delta") or {}).get("median"),
            "warm_mem": (w.get("process_bytes") or 0) + (w.get("resident_file_bytes") or 0) if w else None,
            "build_mem": build.get("peak_footprint"), "build_s": build.get("seconds"),
            "disk": disk.get("bytes_per_vector"), "index_bytes": disk.get("index_bytes"),
            "points": points,
        })

    return rows


def head_trials(head: dict, repo: Path) -> dict[str, list[float]]:
    """The first-query time of every trial of each head-to-head row, from the driver's lines, fastest first."""
    out = {}

    for r in head["rows"]:
        times = sorted(line["total_us"] for line in read_lines(repo / r["source"])
                       if line.get("kind") == "cold" and line.get("status") == "ok" and line.get("regime") == "c0"
                       and line.get("config_id") == r["config_id"] and line.get("cpu_state") == r["cpu_state"]
                       and line.get("threads") == r["threads"] and line.get("total_us") is not None
                       and setting_key(line.get("search_params") or {}) == setting_key(r["setting"]))

        if len(times) != r["trials"]:
            print(f"{r['label']}: {len(times)} trials in {r['source']}, while chart-data.json counts {r['trials']}",
                  file=sys.stderr)

        out[r["system"]] = times

    return out


def stages(quiet: Path) -> dict | None:
    """The mean of each stage of ColdIVF's cold first query over the middle half of its own harness's trials on a
    prewarmed core with read waves, by total time, with the bytes those trials read."""
    trials = sorted((line for line in read_lines(quiet / "a3-m1a.jsonl")
                     if line.get("seq") == 0 and not line.get("pages_resident_at_start")
                     and line.get("mode") == "wave" and line.get("variant") == "prewarm"),
                    key=lambda line: line["total_us"])

    if not trials:
        return None

    cut = len(trials) // 4
    middle = trials[cut:len(trials) - cut]

    def avg(key: str) -> float:
        return sum(line.get(key) or 0.0 for line in middle) / len(middle)

    parts = [(name, sum(avg(k) for k in keys), kind) for name, keys, kind in STAGES]
    total = avg("total_us")
    return {
        "trials": len(trials), "middle": len(middle), "total_us": total,
        "p50_us": percentile([line["total_us"] for line in trials], 50),
        "parts": parts + [("Everything else", max(0.0, total - sum(p[1] for p in parts)), "compute")],
        "syscall_open_us": avg("syscall_open_us"), "table_bytes": avg("table_bytes"),
        "cluster_bytes": avg("cluster_bytes"), "cluster_ranges": avg("cluster_ranges"),
        "row_bytes": avg("row_bytes"), "row_ranges": avg("row_ranges"), "header_bytes": avg("header_bytes"),
        "nprobe": middle[0].get("nprobe"), "rerank": middle[0].get("rerank"), "dataset": middle[0].get("dataset"),
        "n": middle[0].get("n"),
    }


def context(data: dict, quiet: Path, repo: Path) -> dict:
    c = by_id(data)
    curves = c["warm-recall"]["curves"]
    return {
        "data": data, "head": c["cold-head-to-head"], "warm": c["warm-recall"], "smoke": smoke_rows(data),
        "trials": head_trials(c["cold-head-to-head"], repo), "stages": stages(quiet),
        "chip": data["machine"].get("chip", "one machine"),
        "busy": busy_note([cu["machine_state"] for cu in curves], [cu["system"] == "coldivf" for cu in curves]),
    }


def smoke_intro(ctx: dict) -> str:
    w = ctx["warm"]
    return (f"First {millions(w['n'])} rows of {w['dataset']} ({DATASETS.get(w['dataset'], '')}), from the smoke "
            f"test on one {ctx['chip']}.")


def mix(a: str, b: str, f: float) -> str:
    """The color `f` of the way from `a` to `b`."""
    from matplotlib import colors

    ca, cb = colors.to_rgb(a), colors.to_rgb(b)
    return colors.to_hex([x + (y - x) * f for x, y in zip(ca, cb)])


def luminance(c: str) -> float:
    from matplotlib import colors

    r, g, b = colors.to_rgb(c)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def spread(wants: list[float], gap: float, lo: float = -math.inf, hi: float = math.inf) -> list[float]:
    """Positions as near `wants` as they can be with neighbors at least `gap` apart, between lo and hi. Labels that
    would overlap move as a block centered on where they want to be."""
    order = sorted(range(len(wants)), key=lambda i: wants[i])
    blocks = []

    def start(block: list[int]) -> float:
        s = sum(wants[i] - k * gap for k, i in enumerate(block)) / len(block)
        return min(max(s, lo), hi - gap * (len(block) - 1))

    for i in order:
        blocks.append([i])

        while len(blocks) > 1 and start(blocks[-1]) < start(blocks[-2]) + gap * len(blocks[-2]):
            last = blocks.pop()
            blocks[-1] += last

    out = [0.0] * len(wants)

    for block in blocks:
        s = start(block)

        for k, i in enumerate(block):
            out[i] = s + k * gap

    return out


def swarm(xs: list[float], size: float) -> list[float]:
    """Offsets across the row, in points, that set dots `size` points wide at `xs`, in points along it, side by side
    without overlapping, each as near the middle of the row as it can go."""
    placed, out = [], [0.0] * len(xs)

    for i in sorted(range(len(xs)), key=lambda i: xs[i]):
        x = xs[i]
        near = [(px, py) for px, py in placed if abs(px - x) < size]
        tries = {0.0} | {py + s * math.sqrt(size ** 2 - (px - x) ** 2) for px, py in near for s in (1, -1)}
        y = next(y for y in sorted(tries, key=abs)
                 if all((px - x) ** 2 + (py - y) ** 2 >= size ** 2 - 1e-6 for px, py in near))
        placed.append((x, y))
        out[i] = y

    return out


def place_labels(ax, items: list[tuple], obstacles: list[tuple], size: float = 8.5) -> None:
    """Puts each item's label beside its dot where it overlaps no other label or dot, trying 8 spots around the dot in
    turn, and takes the spot with the least overlap when none is clear. Each item is (x, y, label, color, weight,
    radius), with the dot's radius in points, and `obstacles` are boxes in display space."""
    renderer = ax.figure.canvas.get_renderer()
    frame = ax.get_window_extent(renderer)
    taken = list(obstacles)

    def overlap(a, b):
        w = min(a[2], b[2]) - max(a[0], b[0])
        h = min(a[3], b[3]) - max(a[1], b[1])
        return w * h if w > 0 and h > 0 else 0.0

    for x, y, label, color, weight, radius in items:
        d = radius + 3
        e = d * 0.72
        spots = [(d, 0, "left", "center"), (-d, 0, "right", "center"), (0, d, "center", "bottom"),
                 (0, -d, "center", "top"), (e, e, "left", "bottom"), (e, -e, "left", "top"), (-e, e, "right", "bottom"),
                 (-e, -e, "right", "top")]
        best = None

        for ox, oy, ha, va in spots:
            text = ax.annotate(label, (x, y), xytext=(ox, oy), textcoords="offset points", ha=ha, va=va, size=size,
                               color=color, weight=weight, annotation_clip=False)
            ext = text.get_window_extent(renderer)
            area = (ext.x0 - 1.5, ext.y0 - 1.5, ext.x1 + 1.5, ext.y1 + 1.5)
            cost = sum(overlap(area, b) for b in taken)
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


def ratio_ticks(ax, lo: float, hi: float) -> None:
    m = _mpl()
    ticks = [v for e in range(-3, 6) for v in (10.0 ** e, 2 * 10.0 ** e, 5 * 10.0 ** e) if lo <= v <= hi]
    ax.xaxis.set_major_locator(m.FixedLocator(ticks))
    ax.xaxis.set_minor_locator(m.NullLocator())
    ax.set_xticklabels([f"{v:g}×" for v in ticks])


def row_labels(ax, i: float, name: str, detail: str, t: dict, ours: bool = False, x: float = -0.02) -> None:
    """A row's name and, under it, its detail, right-aligned left of the axes."""
    pos = _mpl().blended(ax.transAxes, ax.transData)

    if detail:
        ax.text(x, i - 0.08, name, transform=pos, ha="right", va="bottom", size=9.5, color=t["ink"],
                weight="bold" if ours else "normal")
        ax.text(x, i + 0.08, detail, transform=pos, ha="right", va="top", size=8, color=t["ink2"])
    else:
        ax.text(x, i, name, transform=pos, ha="right", va="center", size=9.5, color=t["ink"],
                weight="bold" if ours else "normal")


def draw_speedup_cold(ctx: dict, t: dict, path: Path) -> None:
    head, trials = ctx["head"], ctx["trials"]
    ours = next(r for r in head["rows"] if r["system"] == "coldivf")
    others = [r for r in head["rows"] if r is not ours]
    marks = (("p50_us", "p50", 13.0), ("p95_us", "p95", 7.0), ("p99_us", "p99", 7.0))
    ratios = [r[k] / ours[k] for r in others for k, _, _ in marks]
    slow = trials.get("coldivf", [])[-2:]
    row_h, left, plot_w = 64.0, 236.0, 400.0
    foot = []

    if min(ratios) < 1 and len(slow) == 2:
        foot.append(f"ColdIVF's p99 is its second slowest of {len(trials['coldivf'])} trials. Its 2 slowest took "
                    f"{fmt_ms(slow[0])} and {fmt_ms(slow[1])}.")

    page = Page(t, "How many times longer the others' cold first query takes than ColdIVF's (preliminary)", [
        f"{millions(head['n'])} vectors of {head['dataset']} ({DATASETS.get(head['dataset'], '')}), at recall@10 "
        f"near {head['target_recall10']:.2f}, on one {ctx['chip']} in a quiet window. Every system ran interleaved "
        f"through the same driver, {fmt_range([r['trials'] for r in head['rows']])} trials each, each a fresh process "
        "with its index evicted, on a prewarmed core. Each bar is a system's time at one quantile of its trials over "
        f"ColdIVF's at the same quantile: {fmt_ms(ours['p50_us'])} at the p50, {fmt_ms(ours['p95_us'])} at the p95 "
        f"and {fmt_ms(ours['p99_us'])} at the p99. Log scale."
    ], 30 + row_h * len(others) + 36, foot)
    ax = page.axes(left, page.top + 30, plot_w, row_h * len(others))
    lo = 0.25 if min(ratios) < 1 else 0.5
    hi = max(ratios) * 2.6
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(len(others) - 0.5, -0.5)
    ax.yaxis.set_major_locator(_mpl().NullLocator())
    ax.spines["left"].set_visible(False)
    ratio_ticks(ax, lo, hi)
    _grid(ax, t)
    ax.set_xlabel("Time over ColdIVF's at the same quantile, log scale", labelpad=6)

    if min(ratios) < 1:
        ax.axvspan(lo, 1, color=mix(t["surface"], t["focus"], 0.08), zorder=0, linewidth=0)

    ax.axvline(1, color=t["coldivf"], linewidth=1.6, zorder=3)
    top = _mpl().blended(ax.transData, ax.transAxes)
    ax.annotate("ColdIVF = 1×", (1, 1), xycoords=top, xytext=(0, 18), textcoords="offset points", ha="center",
                va="bottom", size=9, weight="bold", color=t["coldivf"])

    if min(ratios) < 1:
        ax.annotate("faster than ColdIVF", (1, 1), xycoords=top, xytext=(-8, 4), textcoords="offset points",
                    ha="right", va="bottom", size=8.5, color=t["focus"])

    ax.annotate("slower than ColdIVF", (1, 1), xycoords=top, xytext=(8, 4), textcoords="offset points", ha="left",
                va="bottom", size=8.5, color=t["ink2"])

    for i, r in enumerate(others):
        row_labels(ax, i, r["label"], f"{', '.join(f'{k} {v}' for k, v in r['setting'].items())} · recall "
                   f"{r['recall10_warm']:.3f}", t)

        for j, (key, name, h) in enumerate(marks):
            ratio = r[key] / ours[key]
            y = i + (j - 1) * 0.27 + (0.04 if j else 0)
            faster = ratio < 1
            color = t["focus"] if faster else (t["rest"] if j == 0 else t["rest_tint"])
            ax.barh(y, ratio - 1, left=1, height=h / row_h, color=color, edgecolor="none", zorder=2)
            ax.annotate(f"{fmt_ratio(ratio)} at the {name}", (ratio, y), xytext=(-5 if faster else 5, 0),
                        textcoords="offset points", ha="right" if faster else "left", va="center",
                        size=9.5 if j == 0 else 8, weight="bold" if j == 0 else "normal",
                        color=t["ink"] if j == 0 else t["ink2"])

    page.save(path)


def draw_speedup_warm(ctx: dict, t: dict, path: Path) -> None:
    rows = ctx["smoke"]
    ours = next(r for r in rows if r["system"] == "coldivf")
    have = sorted((r for r in rows if r is not ours and r["warm"] is not None), key=lambda r: r["warm"])
    never = [r for r in rows if r is not ours and r["warm"] is None]
    above = [r for r in have if not r["warm_between"]]
    no_ring = [r for r in have if r["calls"] and r["call_us"] is None]
    calls = [c for r in have for c in r["calls"]]
    row_h, left, plot_w = 23.0, 150.0, 410.0
    foot = []

    if above:
        fastest = [(r["name"], min(r["points"], key=lambda q: q["p50_us"])["recall10"]) for r in above]
        foot.append(f"* Every setting of its grid is above recall {WARM_TARGET:.2f}, so the bar is its fastest "
                    "setting, at recall " + ", ".join(f"{recall:.3f} for {name}" for name, recall in fastest) + ".")

    if never:
        foot.append("No setting reaches recall 0.90 for " + ", ".join(f"{r['name']} (best {r['best_recall']:.3f})"
                                                                     for r in never) + ".")

    if calls:
        foot.append(f"The ring takes the binding's time for one call on a 1,000-vector index, {fmt_time(min(calls))} "
                    f"to {fmt_time(max(calls))}, out of the system's time first."
                    + (f" {', '.join(r['name'] for r in no_ring)} has no ring, since that call takes longer than its "
                       "fastest search." if no_ring else ""))

    if ctx["busy"]:
        foot.append(ctx["busy"])

    page = Page(t, f"How many times longer the others' warm query takes, at recall@10 {WARM_TARGET:.2f} "
                   "(preliminary)", [
        f"{smoke_intro(ctx)} Each bar is a system's warm p50 at recall@10 {WARM_TARGET:.2f} over ColdIVF's, "
        f"{fmt_time(ours['warm'])}, both read off their grids between the two settings around that recall, linear "
        "in the log of the latency. One thread, every index read into memory first. Orange bars are faster than "
        "ColdIVF. Log scale."
    ], 30 + row_h * len(have) + 36, [" ".join(foot)])
    ax = page.axes(left, page.top + 30, plot_w, row_h * len(have))
    ratios = [r["warm"] / ours["warm"] for r in have]
    lo, hi = min(ratios) / 2.8, max(ratios) * 3.2
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(len(have) - 0.5, -0.5)
    ax.yaxis.set_major_locator(_mpl().NullLocator())
    ax.spines["left"].set_visible(False)
    ratio_ticks(ax, lo, hi)
    _grid(ax, t)
    ax.set_xlabel(f"Warm p50 over ColdIVF's at recall@10 {WARM_TARGET:.2f}, log scale", labelpad=6)
    ax.axvspan(lo, 1, color=mix(t["surface"], t["focus"], 0.08), zorder=0, linewidth=0)
    ax.axvline(1, color=t["coldivf"], linewidth=1.6, zorder=3)
    top = _mpl().blended(ax.transData, ax.transAxes)
    ax.text(1, 1 + 10 / (row_h * len(have)), f"ColdIVF = 1×, {fmt_time(ours['warm'])}", transform=top, ha="center",
            va="bottom", size=9, weight="bold", color=t["coldivf"])

    for i, r in enumerate(have):
        ratio = r["warm"] / ours["warm"]
        faster = ratio < 1
        row_labels(ax, i, r["name"], "", t)
        ax.barh(i, ratio - 1, left=1, height=12 / row_h, color=t["focus"] if faster else t["rest"],
                edgecolor="none", zorder=2)
        if r["call_us"]:
            less = (r["warm"] - r["call_us"]) / ours["warm"]
            ax.plot([less], [i], marker="o", markersize=6, markerfacecolor=t["surface"], markeredgecolor=t["ink"],
                    markeredgewidth=1.2, linestyle="none", zorder=4)

        star = "" if r["warm_between"] else "*"
        pos = _mpl().blended(ax.transAxes, ax.transData)
        ax.text(1 + 12 / plot_w, i, f"{fmt_ratio(ratio)}{star}", transform=pos, ha="left", va="center", size=9,
                weight="bold", color=t["focus"] if faster else t["ink"])
        ax.text(1 + 62 / plot_w, i, fmt_time(r["warm"]), transform=pos, ha="left", va="center", size=8.5,
                color=t["ink2"])

    page.save(path)


def tone(t: dict, ratio: float) -> str:
    """A cell's fill for a system's value over ColdIVF's: blue where ColdIVF's is lower, orange where the system's is,
    stronger with the gap up to 100 times, and plain within 10%."""
    gap = math.log10(ratio)

    if abs(gap) < math.log10(1.1):
        return mix(t["surface"], t["grid"], 0.7)

    return mix(t["surface"], t["coldivf"] if gap > 0 else t["focus"], 0.16 + 0.68 * min(1.0, abs(gap) / 2))


def on(t: dict, fill: str) -> str:
    return "#ffffff" if luminance(fill) < 0.4 else t["ink"]


def draw_heatmap(ctx: dict, t: dict, path: Path) -> None:
    m = _mpl()
    rows = ctx["smoke"]
    ours = next(r for r in rows if r["system"] == "coldivf")
    order = [ours] + sorted((r for r in rows if r is not ours), key=lambda r: r["cold"])
    label_w, recall_w, cell_w, cell_h, head_h = 118.0, 44.0, 78.0, 28.0, 30.0
    width = label_w + recall_w + cell_w * len(METRICS)
    height = head_h + cell_h * len(order)
    threads = [r for r in order if (r["threads"] or 1) > 1]
    above = [r for r in order if r["warm"] is not None and not r["warm_between"]]
    foot = [f"Each cell shows a system's number and, under it, that number over ColdIVF's. Recall is the warm "
            "recall@10 of the setting its cold trials ran, which isn't matched across systems."]

    if threads:
        foot.append(f"{', '.join(r['name'] for r in threads)} ran its cold trials on {threads[0]['threads']} "
                    "threads, its default pool, and the rest on 1.")

    if above:
        foot.append(f"* Every setting is above recall {WARM_TARGET:.2f}, so it is the fastest one.")

    if ctx["busy"]:
        foot.append(ctx["busy"])

    page = Page(t, "Every system against ColdIVF on seven numbers (preliminary)", [
        f"{smoke_intro(ctx)} Lower is better in every column. Blue cells are where ColdIVF's number is lower, orange "
        "where the system's is, and the darker the cell the bigger the gap, up to 100 times. Gray cells are within "
        f"10%. The warm query is read off each grid at recall@10 {WARM_TARGET:.2f}, between the two settings around "
        "it."
    ], 30 + height + 6, [" ".join(foot)])
    ax = page.axes(MARGIN, page.top + 30, width, height)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")
    legend = [("ColdIVF lower", t["coldivf"], 1), ("the system lower", t["focus"], -1)]
    x = 0.0

    for name, color, sign in legend:
        ax.text(x, -14, name, size=8, color=t["ink2"], va="center", ha="left")
        x += text_width(name, 8) + 6

        for k, r in enumerate((1.5, 5, 20, 100)):
            fill = tone(t, r ** sign)
            ax.add_artist(rect(x, -20, 26, 12, fill))
            ax.text(x + 13, -14, f"{r:g}×", size=7, color=on(t, fill), ha="center", va="center")
            x += 28

        x += 16

    ax.add_artist(rect(x, -20, 26, 12, tone(t, 1.0)))
    ax.text(x + 32, -14, "within 10%", size=8, color=t["ink2"], va="center", ha="left")
    ax.text(label_w + recall_w / 2, head_h - 17, "Recall", size=8.5, weight="bold", color=t["ink"], ha="center",
            va="baseline")
    ax.text(label_w + recall_w / 2, head_h - 6, "cold setting", size=7, color=t["ink2"], ha="center", va="baseline")

    for k, (key, name, detail, _) in enumerate(METRICS):
        cx = label_w + recall_w + cell_w * (k + 0.5)
        ax.text(cx, head_h - 17, name, size=8.5, weight="bold", color=t["ink"], ha="center", va="baseline")
        ax.text(cx, head_h - 6, detail, size=7, color=t["ink2"], ha="center", va="baseline")

    for j, r in enumerate(order):
        y = head_h + cell_h * j
        me = r is ours
        ax.text(label_w - 8, y + cell_h / 2, r["name"], size=9, color=t["coldivf"] if me else t["ink"],
                weight="bold" if me else "normal", ha="right", va="center")
        ax.text(label_w + recall_w / 2, y + cell_h / 2, f"{r['recall']:.2f}", size=8.5, color=t["ink2"], ha="center",
                va="center")

        for k, (key, _, _, fmt) in enumerate(METRICS):
            x = label_w + recall_w + cell_w * k
            value, base = r[key], ours[key]
            star = "*" if key == "warm" and value is not None and not r["warm_between"] else ""

            if value is None:
                ax.add_artist(rect(x + 1, y + 1, cell_w - 2, cell_h - 2, mix(t["surface"], t["grid"], 0.35)))
                ax.text(x + cell_w / 2, y + cell_h / 2 - 4, "–", size=8.5, color=t["ink2"], ha="center",
                        va="center")
                ax.text(x + cell_w / 2, y + cell_h / 2 + 6, f"never {WARM_TARGET:.2f}" if key == "warm" else
                        "not measured", size=6.5, color=t["ink2"], ha="center", va="center")
                continue

            fill = mix(t["surface"], t["coldivf"], 0.1) if me else tone(t, value / base)
            ink = t["ink"] if me else on(t, fill)
            ax.add_artist(rect(x + 1, y + 1, cell_w - 2, cell_h - 2, fill))
            ax.text(x + cell_w / 2, y + cell_h / 2 - 4.5, fmt(value) + star, size=8.5, color=ink, ha="center",
                    va="center", weight="bold" if me else "normal")
            ax.text(x + cell_w / 2, y + cell_h / 2 + 6.5, "ColdIVF" if me else fmt_ratio(value / base), size=7,
                    color=ink, ha="center", va="center", alpha=0.85)

    page.save(path)


def rect(x: float, y: float, w: float, h: float, color: str):
    from matplotlib.patches import Rectangle

    return Rectangle((x, y), w, h, facecolor=color, edgecolor="none", linewidth=0, clip_on=False)


def draw_memory_latency(ctx: dict, t: dict, path: Path) -> None:
    rows = [r for r in ctx["smoke"] if r["cold_mem"] and r["cold"]]
    biggest = max(r["index_bytes"] or 0 for r in rows)
    threads = [r for r in rows if (r["threads"] or 1) > 1]
    foot = ["Recall isn't matched: each system runs one setting near its defaults, and its label gives that "
            f"setting's warm recall@10. Each dot is the median of {fmt_range([r['cold_trials'] for r in rows])} "
            "trials, each a fresh process with the index evicted, on a prewarmed core."]

    if threads:
        foot.append(f"{', '.join(r['name'] for r in threads)} ran on {threads[0]['threads']} threads, the rest on 1.")

    if ctx["busy"]:
        foot.append(ctx["busy"])

    page = Page(t, "Cold first query against the memory it takes (preliminary)", [
        f"{smoke_intro(ctx)} Each dot is a system's cold first query and the memory that the open and that query add "
        "to its process. The dot's area is the index's size on disk. Down and to the left is better, and no system "
        "is in the shaded corner, faster and smaller than the line of best trade-offs. Log scales."
    ], 410, [" ".join(foot)])
    left, plot_w, plot_h = 70.0, 630.0, 370.0
    ax = page.axes(left, page.top + 14, plot_w, plot_h)
    xs, ys = [r["cold"] for r in rows], [r["cold_mem"] for r in rows]
    xlo, xhi = min(xs) / 2.2, max(xs) * 2.2
    ylo, yhi = min(ys) / 2.2, max(ys) * 2.6
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    _log_ticks(ax, xlo, xhi, fmt=fmt_tick)
    _log_ticks(ax, ylo, yhi, axis="y", fmt=lambda v: fmt_bytes(v).replace(".00", ""))
    _grid(ax, t, "both")
    ax.set_xlabel("Cold first query, p50, log scale", labelpad=6)
    ax.set_ylabel("Memory added by the open and the first query, log scale", labelpad=6)
    best, low = [], math.inf

    for r in sorted(rows, key=lambda r: r["cold"]):
        if r["cold_mem"] < low:
            best.append(r)
            low = r["cold_mem"]

    # The frontier: the least memory of any system at least as fast as each time. Nothing lies below it, and nothing
    # is faster than the fastest system.
    fx, fy = [xlo, best[0]["cold"]], [yhi, yhi]

    for k, b in enumerate(best):
        fx += [b["cold"], best[k + 1]["cold"] if k + 1 < len(best) else xhi]
        fy += [b["cold_mem"], b["cold_mem"]]

    ax.fill_between(fx, [ylo] * len(fx), fy, color=mix(t["surface"], t["coldivf"], 0.07), linewidth=0, zorder=0)
    ax.plot(fx[1:], fy[1:], color=t["coldivf_tint"], linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
    edges = [ax.transData.transform(q) for q in zip(fx[1:], fy[1:])]
    obstacles = [(min(a[0], b[0]) - 2, min(a[1], b[1]) - 2, max(a[0], b[0]) + 2, max(a[1], b[1]) + 2)
                 for a, b in zip(edges, edges[1:])]
    items = []

    for r in sorted(rows, key=lambda r: -(r["index_bytes"] or 0)):
        me = r["system"] == "coldivf"
        size = 6 + 17 * math.sqrt((r["index_bytes"] or 0) / biggest)
        color = t["coldivf"] if me else t["rest"]
        ax.plot([r["cold"]], [r["cold_mem"]], marker="o", markersize=size, markerfacecolor=color, alpha=1.0,
                markeredgecolor=t["surface"], markeredgewidth=1.2, linestyle="none", zorder=4 if me else 3)
        dx, dy = ax.transData.transform((r["cold"], r["cold_mem"]))
        obstacles.append((dx - size / 2, dy - size / 2, dx + size / 2, dy + size / 2))
        items.append((r["cold"], r["cold_mem"], f"{r['name']} ({r['recall']:.2f})", t["ink"] if me else t["ink2"],
                      "bold" if me else "normal", size / 2))

    place_labels(ax, sorted(items, key=lambda q: q[4] != "bold"), obstacles)
    page.save(path)


def draw_slope(ctx: dict, t: dict, path: Path) -> None:
    rows = [r for r in ctx["smoke"] if r["warm_same"] and r["cold"]]
    threads = [r for r in rows if (r["threads"] or 1) > 1]
    foot = ["Recall isn't matched: each system runs one setting near its defaults, the same setting at both ends, and "
            "the label gives its warm recall@10."]

    if threads:
        foot.append(f"{', '.join(r['name'] for r in threads)} ran its cold trials on {threads[0]['threads']} threads "
                    "and its warm ones on 1, the rest both on 1.")

    if ctx["busy"]:
        foot.append(ctx["busy"])

    plot_h = 430.0
    page = Page(t, "What each system's first query costs over a warm one (preliminary)", [
        f"{smoke_intro(ctx)} Each line joins a system's warm p50, at 1 thread with its index in memory, to its cold "
        "first query at the same setting, a fresh process with the index evicted. The steeper the line, the more "
        "the cold start costs. Log scale."
    ], 26 + plot_h + 14, [" ".join(foot)])
    left, plot_w = 120.0, 230.0
    ax = page.axes(left, page.top + 26, plot_w, plot_h)
    values = [v for r in rows for v in (r["warm_same"], r["cold"])]
    lo, hi = min(values) / 1.5, max(values) * 1.5
    ax.set_yscale("log")
    ax.set_ylim(lo, hi)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_locator(_mpl().NullLocator())
    _log_ticks(ax, lo, hi, axis="y", fmt=fmt_tick)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, labelleft=False)
    _grid(ax, t, "y")

    for x, name in ((0, "Warm query"), (1, "Cold first query")):
        ax.axvline(x, color=t["axis"], linewidth=0.8, zorder=1)
        ax.text(x, 1 + 12 / plot_h, name, transform=_mpl().blended(ax.transData, ax.transAxes), ha="center",
                va="bottom", size=9.5, weight="bold", color=t["ink"])

    order = sorted(rows, key=lambda r: r["system"] == "coldivf")

    for r in order:
        me = r["system"] == "coldivf"
        color = t["coldivf"] if me else t["rest"]
        ax.plot([0, 1], [r["warm_same"], r["cold"]], color=color, linewidth=2.8 if me else 1.2, zorder=5 if me else 3,
                marker="o", markersize=6 if me else 4, markeredgecolor=t["surface"], markeredgewidth=1.0)

    for side, key in ((1, "cold"), (0, "warm_same")):
        wants = [ax.transData.transform((side, r[key]))[1] for r in rows]
        got = spread(wants, 12.5)

        for r, want, y in zip(rows, wants, got):
            me = r["system"] == "coldivf"

            if side:
                text = (f"{r['name']}  {fmt_time(r['cold'])}, {fmt_ratio(r['cold'] / r['warm_same'])} its warm query  "
                        f"({r['recall']:.2f})")
            else:
                text = fmt_time(r["warm_same"])

            ax.annotate(text, (side, r[key]), xytext=(10 if side else -10, y - want), textcoords="offset points",
                        ha="left" if side else "right", va="center", size=8.5 if side else 8,
                        color=t["coldivf"] if me else (t["ink"] if side else t["ink2"]),
                        weight="bold" if me else "normal", annotation_clip=False,
                        arrowprops={"arrowstyle": "-", "color": t["axis"], "linewidth": 0.6, "shrinkA": 0,
                                    "shrinkB": 3})

    page.save(path)


def draw_ecdf(ctx: dict, t: dict, path: Path) -> None:
    head, trials = ctx["head"], ctx["trials"]
    rows = sorted(head["rows"], key=lambda r: r["p50_us"])
    plot_h, left, plot_w = 330.0, 64.0, 640.0
    ours = trials.get("coldivf", [])
    page = Page(t, "Every cold trial of the head-to-head, as the share finished by each time (preliminary)", [
        f"{millions(head['n'])} vectors of {head['dataset']} ({DATASETS.get(head['dataset'], '')}), at recall@10 "
        f"near {head['target_recall10']:.2f}, through the same driver in one quiet window on one {ctx['chip']}. "
        "Each step is one trial, a fresh process with its index evicted, on a prewarmed core. A curve further left "
        "finishes sooner, and the top of each curve is its slowest trials. Log scale."
    ], 14 + plot_h + 36, [f"ColdIVF's slowest {min(2, len(ours))} of {len(ours)} trials took "
                          f"{' and '.join(fmt_ms(v) for v in ours[-2:])}, which is why LanceDB IVF_RQ's p99 is lower."
                          if len(ours) > 2 else ""])
    ax = page.axes(left, page.top + 14, plot_w, plot_h)
    every = [v for r in rows for v in trials[r["system"]]]
    lo, hi = min(every) / 1.4, max(every) * 1.6
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, 1.02)
    _log_ticks(ax, lo, hi, fmt=fmt_tick)
    ax.yaxis.set_major_locator(_mpl().FixedLocator([0, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]))
    ax.set_yticklabels(["0", "25%", "50%", "75%", "95%", "", "100%"])
    _grid(ax, t, "both")
    ax.set_xlabel("Cold first query, log scale", labelpad=6)
    ax.set_ylabel("Share of trials finished", labelpad=6)

    for q, name in ((0.5, "p50"), (0.99, "p99")):
        ax.axhline(q, color=t["axis"], linewidth=0.8, linestyle=(0, (2, 2)), zorder=1)
        ax.text(1.005, q, name, transform=_mpl().blended(ax.transAxes, ax.transData), ha="left", va="center",
                size=8, color=t["ink2"])

    for r in rows:
        times = trials[r["system"]]
        me = r["system"] == "coldivf"
        color = t["coldivf"] if me else t["rest"]
        n = len(times)
        ax.step([times[0]] + times, [0] + [(i + 1) / n for i in range(n)], where="post", color=color,
                linewidth=2.4 if me else 1.4, zorder=4 if me else 3)

        for q in (50, 99):
            v = percentile(times, q)
            ax.plot([v], [q / 100], marker="o", markersize=6, color=color, markeredgecolor=t["surface"],
                    markeredgewidth=1.2, linestyle="none", zorder=5)

        p50 = percentile(times, 50)
        ax.annotate(f"{r['label']}\np50 {fmt_ms(p50)}", (p50, 0.5), xytext=(-7, -10), textcoords="offset points",
                    ha="right", va="top", size=8.5, color=t["ink"], weight="bold" if me else "normal")

    page.save(path)


def draw_beeswarm(ctx: dict, t: dict, path: Path) -> None:
    head, trials = ctx["head"], ctx["trials"]
    rows = sorted(head["rows"], key=lambda r: r["p50_us"])
    row_h, left, plot_w, dot = 80.0, 190.0, 420.0, 3.6
    page = Page(t, "Every cold trial of the head-to-head (preliminary)", [
        f"{millions(head['n'])} vectors of {head['dataset']} ({DATASETS.get(head['dataset'], '')}), at recall@10 "
        f"near {head['target_recall10']:.2f}, through the same driver in one quiet window on one {ctx['chip']}. "
        "Each dot is one trial, a fresh process with its index evicted, on a prewarmed core, and dots stack where "
        "trials took about the same time. The tall line is the p50 and the short one the p99. Log scale."
    ], 8 + row_h * len(rows) + 36)
    ax = page.axes(left, page.top + 8, plot_w, row_h * len(rows))
    every = [v for r in rows for v in trials[r["system"]]]
    lo, hi = min(every) / 1.4, max(every) * 1.4
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.yaxis.set_major_locator(_mpl().NullLocator())
    ax.spines["left"].set_visible(False)
    _log_ticks(ax, lo, hi, fmt=fmt_tick)
    _grid(ax, t)
    ax.set_xlabel("Cold first query, log scale", labelpad=6)
    pos = _mpl().blended(ax.transAxes, ax.transData)

    for i, r in enumerate(rows):
        times = trials[r["system"]]
        me = r["system"] == "coldivf"
        color = t["coldivf"] if me else t["rest"]
        xs = [ax.transData.transform((v, i))[0] for v in times]
        offsets = swarm(xs, dot + 0.6)
        reach = row_h / 2 - 6
        scale = min(1.0, reach / max(max(abs(o) for o in offsets), 1e-9))
        ax.scatter(times, [i + o * scale / row_h for o in offsets], s=dot ** 2, color=color, edgecolors="none",
                   zorder=3, alpha=0.9)
        p50, p99 = percentile(times, 50), percentile(times, 99)
        ax.plot([p50, p50], [i - 0.42, i + 0.42], color=t["ink"], linewidth=1.6, zorder=4)
        ax.plot([p99, p99], [i - 0.2, i + 0.2], color=t["ink"], linewidth=1.2, zorder=4)
        row_labels(ax, i, r["label"], f"{len(times)} trials · recall {r['recall10_warm']:.3f}", t, ours=me)
        ax.text(1.02, i - 0.08, f"p50 {fmt_ms(p50)}", transform=pos, ha="left", va="bottom", size=9,
                color=t["ink"], weight="bold" if me else "normal")
        ax.text(1.02, i + 0.08, f"p99 {fmt_ms(p99)}", transform=pos, ha="left", va="top", size=8, color=t["ink2"])

    page.save(path)


def ranks(rows: list[dict], key: str) -> dict[str, int]:
    """Each system's place on `key`, 1 for the lowest, ties sharing the better place. Systems with no number have
    none."""
    have = [r[key] for r in rows if r[key] is not None]
    return {r["system"]: 1 + sum(v < r[key] for v in have) for r in rows if r[key] is not None}


def draw_bump(ctx: dict, t: dict, path: Path) -> None:
    rows = ctx["smoke"]
    places = [ranks(rows, key) for key, _, _, _ in METRICS]
    n = len(rows)
    row_h, left, plot_w = 22.0, 150.0, 430.0
    page = Page(t, "Where each system places on seven numbers (preliminary)", [
        f"{smoke_intro(ctx)} Each column ranks the {n} systems on one number, 1 for the lowest, which is best in "
        "every column, and each line follows one system across them. A system with no number for a column, eg. one "
        f"that never reaches recall@10 {WARM_TARGET:.2f}, skips it."
    ], 44 + row_h * (n - 1) + 24, ["Recall isn't matched for the cold query: each system runs one setting near "
                                   f"its defaults. The warm query is read off each grid at recall {WARM_TARGET:.2f}. "
                                   + (ctx["busy"] or "")])
    ax = page.axes(left, page.top + 44, plot_w, row_h * (n - 1))
    ax.set_xlim(-0.08, len(METRICS) - 0.92)
    ax.set_ylim(n + 0.1, 0.9)
    ax.axis("off")
    order = sorted(rows, key=lambda r: r["system"] == "coldivf")

    for k, (key, name, detail, _) in enumerate(METRICS):
        ax.axvline(k, color=t["grid"], linewidth=0.8, zorder=0)
        top = _mpl().blended(ax.transData, ax.transAxes)
        ax.text(k, 1 + 30 / (row_h * (n - 1)), name, transform=top, ha="center", va="bottom", size=8.5,
                weight="bold", color=t["ink"])
        ax.text(k, 1 + 18 / (row_h * (n - 1)), detail, transform=top, ha="center", va="bottom", size=7,
                color=t["ink2"])

    for r in order:
        me = r["system"] == "coldivf"
        color = t["coldivf"] if me else t["rest"]
        pts = [(k, p.get(r["system"])) for k, p in enumerate(places)]
        have = [(k, y) for k, y in pts if y is not None]

        for (k0, y0), (k1, y1) in zip(have, have[1:]):
            ax.plot([k0, k1], [y0, y1], color=color, linewidth=3 if me else 1.1, alpha=1 if me else 0.75,
                    linestyle="-" if k1 == k0 + 1 else (0, (2, 2)), zorder=4 if me else 2)

        for k, y in have:
            if me:
                ax.plot([k], [y], marker="o", markersize=15, color=t["coldivf"], markeredgecolor=t["surface"],
                        markeredgewidth=1.5, zorder=5)
                ax.text(k, y, str(y), ha="center", va="center", size=8, weight="bold", color="#ffffff", zorder=6)
            else:
                ax.plot([k], [y], marker="o", markersize=5, color=t["rest"], markeredgecolor=t["surface"],
                        markeredgewidth=0.8, zorder=3)

        first, last = have[0], have[-1]
        ax.annotate(f"{r['name']}  {first[1]}", (first[0], first[1]), xytext=(-12, 0), textcoords="offset points",
                    ha="right", va="center", size=8.5, color=t["coldivf"] if me else t["ink2"],
                    weight="bold" if me else "normal", annotation_clip=False)
        ax.annotate(f"{last[1]}  {r['name']}", (last[0], last[1]), xytext=(12, 0), textcoords="offset points",
                    ha="left", va="center", size=8.5, color=t["coldivf"] if me else t["ink2"],
                    weight="bold" if me else "normal", annotation_clip=False)

    page.save(path)


def scores(rows: list[dict]) -> dict[str, list[float]]:
    """Each system's place between the field's worst and best on each metric, on a log scale, from 0.12 for the worst
    to 1 for the best, or 0 where it has no number."""
    out = {r["system"]: [] for r in rows}

    for key, _, _, _ in METRICS:
        have = [r[key] for r in rows if r[key]]
        lo, hi = math.log(min(have)), math.log(max(have))

        for r in rows:
            out[r["system"]].append(0.0 if not r[key] else 0.12 + 0.88 * (hi - math.log(r[key])) / ((hi - lo) or 1))

    return out


def draw_radar(ctx: dict, t: dict, path: Path) -> None:
    rows = ctx["smoke"]
    ours = next(r for r in rows if r["system"] == "coldivf")
    order = [ours] + [r for r in rows if r is not ours]
    s = scores(rows)
    cols, cell_w, cell_h = 5, 141.6, 150.0
    lines = math.ceil(len(order) / cols)
    short = ("Cold", "Warm", "Cold\nmemory", "Warm\nmemory", "Build\nmemory", "Build\ntime", "Disk")
    page = Page(t, "Each system's shape across seven numbers (preliminary)", [
        f"{smoke_intro(ctx)} Each spoke is one number, and the further out, the better the system is on it: the "
        "field's best sits at the rim and its worst near the middle, on a log scale. ColdIVF's shape is the blue "
        "outline on every panel. Spokes: the cold first query at the smoke setting, the warm query at recall@10 "
        f"{WARM_TARGET:.2f}, the memory added by a cold query, warm memory with cached pages, the build's peak "
        "memory and time, and the bytes per vector on disk."
    ], cell_h * lines + 6, ["Recall isn't matched for the cold query, and a system with no number for a spoke "
                            f"sits at the middle of it, eg. one that never reaches recall {WARM_TARGET:.2f}. "
                            + (ctx["busy"] or "")])
    angles = [math.pi / 2 - 2 * math.pi * k / len(METRICS) for k in range(len(METRICS))]
    ring = [a for a in angles] + [angles[0]]

    for idx, r in enumerate(order):
        row, col = divmod(idx, cols)
        cx = MARGIN + cell_w * (col + 0.5)
        cy = page.top + cell_h * row + 84
        radius = 44.0
        ax = page.axes(cx - radius - 28, cy - radius - 28, 2 * radius + 56, 2 * radius + 56)
        ax.set_xlim(-1.64, 1.64)
        ax.set_ylim(-1.64, 1.64)
        ax.axis("off")
        me = r is ours

        for level in (0.5, 1.0):
            ax.plot([level * math.cos(a) for a in ring], [level * math.sin(a) for a in ring], color=t["grid"],
                    linewidth=0.8, zorder=1)

        for a, label in zip(angles, short):
            ax.plot([0, math.cos(a)], [0, math.sin(a)], color=t["grid"], linewidth=0.8, zorder=1)
            ax.text(1.3 * math.cos(a), 1.3 * math.sin(a), label, ha="center", va="center", size=6, color=t["ink2"],
                    linespacing=0.95)

        def shape(values):
            return ([v * math.cos(a) for v, a in zip(values, angles)] + [values[0] * math.cos(angles[0])],
                    [v * math.sin(a) for v, a in zip(values, angles)] + [values[0] * math.sin(angles[0])])

        x, y = shape(s[r["system"]])

        if me:
            ax.fill(x, y, color=t["coldivf"], alpha=0.35, linewidth=0, zorder=2)
            ax.plot(x, y, color=t["coldivf"], linewidth=1.6, zorder=3)
        else:
            ax.fill(x, y, color=t["rest"], alpha=0.35, linewidth=0, zorder=2)
            ax.plot(x, y, color=t["rest"], linewidth=1.1, zorder=3)
            bx, by = shape(s[ours["system"]])
            ax.plot(bx, by, color=t["coldivf"], linewidth=1.2, zorder=4)

        page.text(cx, page.top + cell_h * row + 22, r["name"], ha="center", size=9,
                  weight="bold" if me else "normal", color=t["coldivf"] if me else t["ink"])

    page.save(path)


def draw_stages(ctx: dict, t: dict, path: Path) -> None:
    st, head = ctx["stages"], ctx["head"]
    parts = [p for p in st["parts"] if p[1] >= 0.5]
    row_h, left, plot_w = 24.0, 236.0, 380.0
    others = [r for r in head["rows"] if r["system"] != "coldivf"]
    detail = {
        "Open the index file": f"reads its first {st['header_bytes'] / 1024:.0f} KiB; a plain open() alone takes "
                               f"{fmt_ms(st['syscall_open_us'])}",
        "Load the cluster tables": f"{fmt_bytes(st['table_bytes'])} of 4-bit codes and extents",
        "Wait for the cluster reads": f"{fmt_bytes(st['cluster_bytes'])} in {st['cluster_ranges']:.0f} ranges",
        "Read the rerank rows": f"{fmt_bytes(st['row_bytes'])} in {st['row_ranges']:.0f} ranges",
    }
    page = Page(t, "Where ColdIVF's cold first query spends its time (preliminary)", [
        f"{millions(st['n'])} vectors of {st['dataset']} ({DATASETS.get(st['dataset'], '')}), at nprobe "
        f"{st['nprobe']}, rerank {st['rerank']}, recall@10 near 0.95, on one {ctx['chip']} in a quiet window. Each "
        f"bar is a stage's mean over the middle half of the {st['trials']} trials of ColdIVF's own cold binary on a "
        f"prewarmed core, the {st['middle']} trials between its p25 and p75, and each starts where the one before "
        f"ends. The stages add up to {fmt_ms(st['total_us'])}, beside a p50 of {fmt_ms(st['p50_us'])} over all "
        "the trials."
    ], 26 + row_h * (len(parts) + 1) + 40, [
        "At the same recall through the driver, the whole first query takes " + ", ".join(
            f"{fmt_ms(r['p50_us'])} for {r['label']}" for r in others) + " at the p50. A read stage includes the "
        "CPU time to issue and copy its reads."])
    ax = page.axes(left, page.top + 26, plot_w, row_h * (len(parts) + 1))
    hi = st["total_us"] * 1.04
    ax.set_xlim(0, hi)
    ax.set_ylim(len(parts) + 0.5, -0.5)
    ax.yaxis.set_major_locator(_mpl().NullLocator())
    ax.spines["left"].set_visible(False)
    ax.xaxis.set_major_locator(_mpl().FixedLocator([v * 1000 for v in range(0, int(hi // 1000) + 1)]))
    ax.set_xticklabels([f"{v} ms" for v in range(0, int(hi // 1000) + 1)])
    _grid(ax, t)
    ax.set_xlabel("Time since the query arrived (ms)", labelpad=6)
    pos = _mpl().blended(ax.transAxes, ax.transData)
    at = 0.0

    for i, (name, us, kind) in enumerate(parts):
        color = t["coldivf"] if kind == "read" else t["coldivf_tint"]
        ax.barh(i, us, left=at, height=14 / row_h, color=color, edgecolor="none", zorder=2)

        if i:
            ax.plot([at, at], [i - 1 + 7 / row_h, i - 7 / row_h], color=t["axis"], linewidth=0.7, zorder=1)

        sub = detail.get(name)
        ax.text(-0.02, i - (0.06 if sub else 0), name, transform=pos, ha="right",
                va="bottom" if sub else "center", size=9, color=t["ink"])

        if sub:
            ax.text(-0.02, i + 0.06, sub, transform=pos, ha="right", va="top", size=7.5, color=t["ink2"])

        share = us / st["total_us"]
        ax.annotate(f"{fmt_time(us)} · {share:.0%}" if share >= 0.005 else f"{fmt_time(us)} · under 1%", (at + us, i),
                    xytext=(5, 0), textcoords="offset points",
                    ha="left", va="center", size=8.5, color=t["ink"])
        at += us

    total = len(parts)
    ax.axhline(total - 0.5, color=t["axis"], linewidth=0.8)
    ax.barh(total, st["total_us"], left=0, height=14 / row_h, color=t["ink"], edgecolor="none", zorder=2)
    ax.text(-0.02, total, "The whole first query", transform=pos, ha="right", va="center", size=9.5, weight="bold",
            color=t["ink"])
    ax.annotate(fmt_ms(st["total_us"]), (st["total_us"], total), xytext=(5, 0), textcoords="offset points",
                ha="left", va="center", size=9.5, weight="bold", color=t["ink"])
    handles = [_mpl().Patch(facecolor=t["coldivf"], label="Reading the index"),
               _mpl().Patch(facecolor=t["coldivf_tint"], label="Computing")]
    page.fig.legend(handles=handles, loc="upper left", ncol=2, handlelength=1.2, columnspacing=1.6, borderaxespad=0,
                    borderpad=0, bbox_to_anchor=page.anchor(MARGIN, page.top + 4))
    page.save(path)


def draw_leaderboard(ctx: dict, t: dict, path: Path) -> None:
    rows = ctx["smoke"]
    boards = []

    for target in LEADER_TARGETS:
        got = [(r, at_recall(r["points"], target)) for r in rows]
        boards.append((target, sorted(((r, a) for r, a in got if a), key=lambda q: q[1][0]),
                       [r for r, a in got if not a]))

    longest = max(len(b[1]) for b in boards)
    every = [a[0] for _, have, _ in boards for _, a in have]
    lo, hi = min(every) / 1.5, max(every) * 1.5
    row_h, label_w, plot_w, value_w, gap = 16.0, 104.0, 90.0, 42.0, 6.0
    panel_w = label_w + plot_w + value_w
    page = Page(t, "The fastest warm query at three recall targets (preliminary)", [
        f"{smoke_intro(ctx)} Each panel ranks the systems by warm p50 at one recall@10, read off each grid between "
        "the two settings around it, linear in the log of the latency, at 1 thread with every index read into "
        "memory first. A system whose every setting is above a target shows its fastest, marked *. All three "
        "panels share one log scale."
    ], 30 + row_h * longest + 66, [ctx["busy"] or ""])

    for b, (target, have, never) in enumerate(boards):
        x0 = MARGIN + b * (panel_w + gap)
        ax = page.axes(x0 + label_w, page.top + 30, plot_w, row_h * longest)
        ax.set_xscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(longest - 0.5, -0.5)
        ax.yaxis.set_major_locator(_mpl().NullLocator())
        ax.spines["left"].set_visible(False)
        _log_ticks(ax, lo, hi, fmt=lambda v: f"{v / 1000:g}")
        ax.tick_params(axis="x", labelsize=7)
        ax.set_xlabel("ms, log scale", size=7.5, labelpad=3)
        _grid(ax, t)
        page.text(x0, page.top + 18, f"Recall@10 {target:.2f}", size=10, weight="bold", color=t["ink"])
        pos = _mpl().blended(ax.transAxes, ax.transData)

        for i, (r, (us, between)) in enumerate(have):
            me = r["system"] == "coldivf"
            color = t["coldivf"] if me else t["rest"]
            ax.plot([lo, us], [i, i], color=color, linewidth=1.2 if not me else 2, zorder=2, solid_capstyle="butt")
            ax.plot([us], [i], marker="o", markersize=6 if me else 4.5, color=color, markeredgecolor=t["surface"],
                    markeredgewidth=0.8, zorder=3)
            ax.text(-label_w / plot_w, i, f"{i + 1}", transform=pos, ha="left", va="center", size=7.5,
                    color=t["ink2"])
            ax.text(-0.04, i, r["name"], transform=pos, ha="right", va="center", size=8,
                    color=t["coldivf"] if me else t["ink"], weight="bold" if me else "normal")
            ax.text(1 + value_w / plot_w, i, fmt_time(us) + ("" if between else "*"), transform=pos, ha="right",
                    va="center", size=8, color=t["coldivf"] if me else t["ink"], weight="bold" if me else "normal")

        if never:
            names = ", ".join(r["name"] for r in never)

            for k, line in enumerate(wrap(f"Never reaches {target:.2f}: {names}.", 7.5, panel_w - 4)):
                page.text(x0, page.top + 30 + row_h * longest + 40 + 10 * k, line, size=7.5, color=t["ink2"])

    page.save(path)


def draw_qps(ctx: dict, t: dict, path: Path) -> None:
    curves = ctx["warm"]["curves"]
    plot_w, plot_h, left = 470.0, 360.0, 64.0
    page = Page(t, f"Queries per second against recall@10 at {millions(ctx['warm']['n'])} vectors (preliminary)", [
        f"{smoke_intro(ctx)} The layout of ann-benchmarks: each dot is one setting of a system, at its warm "
        "recall@10 and the queries per second (QPS) one thread serves at it, with every index read into memory "
        "first. Up and to the right is better. Solid lines are inverted-file indexes, dashed lines graphs, and lone "
        "markers systems with one setting. Log scale for QPS."
    ], 20 + plot_h + 34, [ctx["busy"] or ""])
    ax = page.axes(left, page.top + 20, plot_w, plot_h)
    qps = [p["qps"] for c in curves for p in c["points"] if p.get("qps")]
    lo, hi = min(qps) / 1.6, max(qps) * 1.6
    ax.set_yscale("log")
    ax.set_ylim(lo, hi)
    ax.set_xlim(0.55, 1.02)
    _log_ticks(ax, lo, hi, axis="y", fmt=lambda v: f"{v:,g}")
    ax.xaxis.set_major_locator(_mpl().FixedLocator([0.6, 0.7, 0.8, 0.9, 1.0]))
    _grid(ax, t, "both")
    ax.set_xlabel("Recall@10", labelpad=6)
    ax.set_ylabel("Queries per second, 1 thread, log scale", labelpad=4)
    handles, used, lone = [], 0, 0

    for c in curves:
        x = [p["recall10"] for p in c["points"]]
        y = [p["qps"] for p in c["points"]]

        if c["system"] == "coldivf":
            line = ax.plot(x, y, color=t["coldivf"], linewidth=2.8, marker="o", markersize=7, zorder=5,
                           markeredgecolor=t["surface"], markeredgewidth=1.2, label="ColdIVF")[0]
        elif len(c["points"]) == 1:
            line = ax.plot(x, y, linestyle="none", marker=LONE_MARKERS[lone % len(LONE_MARKERS)], markersize=7,
                           color=t["rest"], markeredgecolor=t["surface"], markeredgewidth=0.8, zorder=4,
                           label=c["label"])[0]
            lone += 1
        else:
            family = next((f for f, names in FAMILIES.items() if c["system"] in names), "graph")
            color = PALETTE[used % len(PALETTE)]
            color = t["ink"] if color == "ink" else color
            used += 1
            line = ax.plot(x, y, color=color, linewidth=1.5, linestyle="-" if family == "ivf" else (0, (4, 2)),
                           marker="o", markersize=4, markeredgecolor=t["surface"], markeredgewidth=0.6, zorder=3,
                           label=c["label"])[0]

        handles.append(line)

    page.fig.legend(handles=handles, loc="upper left", ncol=1, handlelength=2.6, labelspacing=0.55, borderaxespad=0,
                    borderpad=0, fontsize=8.5, bbox_to_anchor=page.anchor(left + plot_w + 18, page.top + 20))
    page.save(path)


def draw_relative(ctx: dict, t: dict, path: Path) -> None:
    curves = [c for c in ctx["warm"]["curves"] if (c.get("against_coldivf") or {}).get("points")]
    ours = next(c for c in ctx["warm"]["curves"] if c["system"] == "coldivf")
    recalls = [p["recall10"] for p in ours["points"]]
    plot_w, plot_h, left = 450.0, 360.0, 64.0
    page = Page(t, "How each system's warm time compares with ColdIVF's along the recall (preliminary)", [
        f"{smoke_intro(ctx)} Each line is one system's warm p50 over ColdIVF's at the same recall@10, with ColdIVF's "
        "read off its grid between the two settings around it, inside the recall its grid covers, "
        f"{min(recalls):.2f} to {max(recalls):.2f}. Below the blue line a system is faster than ColdIVF, above it "
        "slower. One thread, every index read into memory first. Log scale."
    ], 20 + plot_h + 34, [ctx["busy"] or ""])
    ax = page.axes(left, page.top + 20, plot_w, plot_h)
    ratios = [p["ratio"] for c in curves for p in c["against_coldivf"]["points"]]
    lo, hi = min(ratios) / 1.6, max(ratios) * 1.6
    ax.set_yscale("log")
    ax.set_ylim(lo, hi)
    ax.set_xlim(min(recalls) - 0.005, max(recalls) + 0.005)
    ticks = [v for e in range(-2, 4) for v in (10.0 ** e, 3 * 10.0 ** e) if lo <= v <= hi]
    ax.yaxis.set_major_locator(_mpl().FixedLocator(ticks))
    ax.yaxis.set_minor_locator(_mpl().NullLocator())
    ax.set_yticklabels([f"{v:g}×" for v in ticks])
    _grid(ax, t, "both")
    ax.set_xlabel("Recall@10", labelpad=6)
    ax.set_ylabel("Warm p50 over ColdIVF's at the same recall, log scale", labelpad=4)
    ax.axhspan(lo, 1, color=mix(t["surface"], t["focus"], 0.08), zorder=0, linewidth=0)
    ax.axhline(1, color=t["coldivf"], linewidth=2.4, zorder=4)
    ax.text(0.01, 1, "ColdIVF", transform=_mpl().blended(ax.transAxes, ax.transData), ha="left", va="bottom",
            size=9, weight="bold", color=t["coldivf"])
    ax.text(0.01, 0.02, "faster than ColdIVF", transform=ax.transAxes, ha="left", va="bottom", size=8.5,
            color=t["focus"])
    used, ends = 0, []

    for c in curves:
        pts = c["against_coldivf"]["points"]
        family = next((f for f, names in FAMILIES.items() if c["system"] in names), "graph")
        color = PALETTE[used % len(PALETTE)]
        color = t["ink"] if color == "ink" else color
        used += 1
        x = [p["recall10"] for p in pts]
        y = [p["ratio"] for p in pts]
        ax.plot(x, y, color=color, linewidth=1.5, linestyle="-" if family == "ivf" else (0, (4, 2)), marker="o",
                markersize=4, markeredgecolor=t["surface"], markeredgewidth=0.6, zorder=3)
        ends.append((c, x[-1], y[-1], color))

    wants = [ax.transData.transform((x, y))[1] for _, x, y, _ in ends]
    got = spread(wants, 11.5)
    right = ax.get_xlim()[1]

    for (c, x, y, color), want, at in zip(ends, wants, got):
        ax.annotate(c["label"], (x, y), xytext=(ax.transData.transform((right, y))[0]
                                                - ax.transData.transform((x, y))[0] + 10, at - want),
                    textcoords="offset points", ha="left", va="center", size=8.5, color=color,
                    annotation_clip=False,
                    arrowprops={"arrowstyle": "-", "color": color, "linewidth": 0.6, "alpha": 0.6, "shrinkA": 0,
                                "shrinkB": 2})

    page.save(path)


def draw_number_line(ctx: dict, t: dict, path: Path) -> None:
    rows = sorted(ctx["smoke"], key=lambda r: r["cold"])
    threads = [r for r in rows if (r["threads"] or 1) > 1]
    left, plot_w, level_h, levels = 30.0, 690.0, 27.0, 7
    plot_h = level_h * levels + 30
    foot = ["Recall isn't matched: each system runs one setting near its defaults, and its label gives that setting's "
            "warm recall@10."]

    if threads:
        foot.append(f"{', '.join(r['name'] for r in threads)} ran on {threads[0]['threads']} threads, the rest on 1.")

    if ctx["busy"]:
        foot.append(ctx["busy"])

    page = Page(t, "Each system's cold first query on one time line (preliminary)", [
        f"{smoke_intro(ctx)} Each stem marks a system's median cold first query, a fresh process with its index "
        "evicted, on a prewarmed core. The lines mark one frame of a 60 Hz screen, 100 ms, about where a response "
        "stops feeling instant, and one second. Log scale."
    ], plot_h + 40, [" ".join(foot)])
    ax = page.axes(left, page.top, plot_w, plot_h)
    lo, hi = 10 ** math.floor(math.log10(rows[0]["cold"])) * 2, 10 ** math.ceil(math.log10(rows[-1]["cold"]))
    lo = min(lo, rows[0]["cold"] / 1.5)
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, plot_h)
    ax.spines["left"].set_visible(False)
    ax.yaxis.set_major_locator(_mpl().NullLocator())
    _log_ticks(ax, lo, hi, fmt=fmt_tick)
    ax.set_xlabel("Cold first query, p50, log scale", labelpad=6)

    for v, name in ((16_700, "one 60 Hz frame"), (100_000, "100 ms"), (1_000_000, "1 second")):
        if lo < v < hi:
            ax.axvline(v, color=t["axis"], linewidth=0.9, linestyle=(0, (3, 3)), zorder=1)
            ax.text(v, plot_h - 4, f" {name}", ha="left", va="top", size=8, color=t["ink2"], style="italic")

    ends = []

    for r in rows:
        me = r["system"] == "coldivf"
        x = ax.transData.transform((r["cold"], 0))[0] - ax.transData.transform((lo, 0))[0]
        name, sub = r["name"], f"{fmt_time(r['cold'])} · recall {r['recall']:.2f}"
        w = max(text_width(name, 8.5, "bold" if me else "normal"), text_width(sub, 7.5)) + 6
        flip = x + w > plot_w
        span = (x - w, x) if flip else (x, x + w)
        level = next(k for k in range(levels + 1) if k == levels or k >= len(ends)
                     or all(e[1] + 4 < span[0] or e[0] - 4 > span[1] for e in ends[k]))

        while len(ends) <= level:
            ends.append([])

        ends[level].append(span)
        y = 12 + level * level_h
        color = t["coldivf"] if me else t["rest"]
        ax.plot([r["cold"], r["cold"]], [0, y + 18], color=color, linewidth=2 if me else 1, zorder=2)
        ax.plot([r["cold"]], [0], marker="o", markersize=7 if me else 5, color=color, markeredgecolor=t["surface"],
                markeredgewidth=1, zorder=3, clip_on=False)
        ha, dx = ("right", -4) if flip else ("left", 4)
        back = {"boxstyle": "square,pad=0.1", "facecolor": t["surface"], "edgecolor": "none"}
        ax.annotate(name, (r["cold"], y + 9), xytext=(dx, 0), textcoords="offset points", ha=ha, va="bottom",
                    size=8.5, color=t["coldivf"] if me else t["ink"], weight="bold" if me else "normal", bbox=back,
                    zorder=4)
        ax.annotate(sub, (r["cold"], y + 8), xytext=(dx, 0), textcoords="offset points", ha=ha, va="top", size=7.5,
                    color=t["ink2"], bbox=back, zorder=4)

    page.save(path)


def draw_quadrant(ctx: dict, t: dict, path: Path) -> None:
    rows = [r for r in ctx["smoke"] if r["warm_same"] and r["cold"]]
    ours = next(r for r in rows if r["system"] == "coldivf")
    threads = [r for r in rows if (r["threads"] or 1) > 1]
    foot = ["Recall isn't matched: each system runs one setting near its defaults, the same setting on both axes, and "
            "its label gives that setting's warm recall@10."]

    if threads:
        foot.append(f"{', '.join(r['name'] for r in threads)} ran its cold trials on {threads[0]['threads']} threads "
                    "and its warm ones on 1.")

    if ctx["busy"]:
        foot.append(ctx["busy"])

    page = Page(t, "Warm against cold: which systems are fast both ways (preliminary)", [
        f"{smoke_intro(ctx)} Each dot is a system at its smoke setting: its warm p50, at 1 thread with every index "
        "read into memory first, against its cold first query, a fresh process with its index evicted. The blue "
        "lines cross at ColdIVF, splitting the systems into those faster or slower than it each way. Log scales."
    ], 400, [" ".join(foot)])
    left, plot_w, plot_h = 70.0, 630.0, 360.0
    ax = page.axes(left, page.top + 14, plot_w, plot_h)
    xs, ys = [r["warm_same"] for r in rows], [r["cold"] for r in rows]
    xlo, xhi, ylo, yhi = min(xs) / 2.2, max(xs) * 2.2, min(ys) / 2.6, max(ys) * 2.2
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    _log_ticks(ax, xlo, xhi, fmt=fmt_tick)
    _log_ticks(ax, ylo, yhi, axis="y", fmt=fmt_tick)
    _grid(ax, t, "both")
    ax.set_xlabel("Warm query, p50, log scale", labelpad=6)
    ax.set_ylabel("Cold first query, p50, log scale", labelpad=6)
    ax.axvline(ours["warm_same"], color=t["coldivf"], linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    ax.axhline(ours["cold"], color=t["coldivf"], linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    ax.fill_between([xlo, ours["warm_same"]], [ylo] * 2, [ours["cold"]] * 2,
                    color=mix(t["surface"], t["coldivf"], 0.08), linewidth=0, zorder=0)

    def count(fx, fy):
        return sum(1 for r in rows if r is not ours and fx(r["warm_same"]) and fy(r["cold"]))

    wx, cy = ours["warm_same"], ours["cold"]
    corners = (
        (0.01, 0.985, "left", "top", "Faster warm, slower cold", count(lambda v: v < wx, lambda v: v > cy)),
        (0.99, 0.985, "right", "top", "Slower both ways", count(lambda v: v > wx, lambda v: v > cy)),
        (0.01, 0.015, "left", "bottom", "Faster both ways", count(lambda v: v < wx, lambda v: v < cy)),
        (0.99, 0.015, "right", "bottom", "Faster cold, slower warm", count(lambda v: v > wx, lambda v: v < cy)),
    )

    for x, y, ha, va, name, k in corners:
        ax.text(x, y, f"{name}: {k if k else 'none'}", transform=ax.transAxes, ha=ha, va=va, size=8.5,
                color=t["ink2"], style="italic")

    items, obstacles = [], []

    for r in rows:
        me = r is ours
        size = 10 if me else 7.5
        ax.plot([r["warm_same"]], [r["cold"]], marker="o", markersize=size, color=t["coldivf"] if me else t["rest"],
                markeredgecolor=t["surface"], markeredgewidth=1.5, linestyle="none", zorder=4 if me else 3)
        dx, dy = ax.transData.transform((r["warm_same"], r["cold"]))
        obstacles.append((dx - size / 2, dy - size / 2, dx + size / 2, dy + size / 2))
        items.append((r["warm_same"], r["cold"], f"{r['name']} ({r['recall']:.2f})", t["ink"] if me else t["ink2"],
                      "bold" if me else "normal", size / 2))

    renderer = page.fig.canvas.get_renderer()
    cx, cy = ax.transData.transform((ours["warm_same"], ours["cold"]))
    frame = ax.get_window_extent(renderer)
    obstacles += [(cx - 1.5, frame.y0, cx + 1.5, frame.y1), (frame.x0, cy - 1.5, frame.x1, cy + 1.5)]

    for text in ax.texts:
        e = text.get_window_extent(renderer)
        obstacles.append((e.x0, e.y0, e.x1, e.y1))

    place_labels(ax, sorted(items, key=lambda q: q[4] != "bold"), obstacles)
    page.save(path)


def draw_report_card(ctx: dict, t: dict, path: Path) -> None:
    rows = ctx["smoke"]
    ours = next(r for r in rows if r["system"] == "coldivf")
    order = [ours] + sorted((r for r in rows if r is not ours), key=lambda r: r["cold"])
    label_w, col_w, bar_w, row_h, head_h = 112.0, 84.0, 40.0, 19.0, 30.0
    width = label_w + col_w * len(METRICS)
    height = head_h + row_h * len(order)
    page = Page(t, "A report card: each system's seven numbers beside ColdIVF's (preliminary)", [
        f"{smoke_intro(ctx)} Each bar is a system's number on a log scale from the field's lowest, a short bar, to its "
        "highest, a full one. Lower is better in every column. The blue line in each column is ColdIVF's number, so "
        "a bar that stops short of it beats ColdIVF there, and the field's best number in each column is bold."
    ], height + 8, ["Recall isn't matched for the cold query: each system runs one setting near its defaults. The "
                    f"warm query is read off each grid at recall@10 {WARM_TARGET:.2f}. " + (ctx["busy"] or "")])
    ax = page.axes(MARGIN, page.top, width, height)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")

    for j, r in enumerate(order):
        y = head_h + row_h * j

        if r is ours:
            ax.add_artist(rect(0, y, width, row_h, mix(t["surface"], t["coldivf"], 0.1)))
        elif j % 2 == 0:
            ax.add_artist(rect(0, y, width, row_h, mix(t["surface"], t["grid"], 0.45)))

        ax.text(label_w - 8, y + row_h / 2, r["name"], size=8.5, ha="right", va="center",
                color=t["coldivf"] if r is ours else t["ink"], weight="bold" if r is ours else "normal")

    for k, (key, name, detail, fmt) in enumerate(METRICS):
        x0 = label_w + col_w * k
        have = [r[key] for r in order if r[key]]
        lo, hi = math.log(min(have)), math.log(max(have))
        ax.text(x0 + 2, head_h - 17, name, size=8.5, weight="bold", color=t["ink"], ha="left", va="baseline")
        ax.text(x0 + 2, head_h - 6, detail, size=7, color=t["ink2"], ha="left", va="baseline")

        def length(v):
            return 2 + (bar_w - 2) * (math.log(v) - lo) / ((hi - lo) or 1)

        for j, r in enumerate(order):
            y = head_h + row_h * j
            value = r[key]

            if not value:
                ax.text(x0 + 2, y + row_h / 2, "–", size=8, color=t["ink2"], ha="left", va="center")
                continue

            me = r is ours
            ax.add_artist(rect(x0 + 2, y + 5, length(value), row_h - 10, t["coldivf"] if me else t["rest"]))
            star = "*" if key == "warm" and not r["warm_between"] else ""
            ax.text(x0 + bar_w + 6, y + row_h / 2, fmt(value) + star, size=7.5, ha="left", va="center",
                    color=t["ink"], weight="bold" if value == min(have) else "normal")

        if ours[key]:
            bx = x0 + 2 + length(ours[key])
            ax.plot([bx, bx], [head_h, height], color=t["coldivf"], linewidth=0.9, zorder=5)

    page.save(path)


# The gallery in the README's order: each drawing's file name and function.
GALLERY_ORDER = (
    ("speedup-cold", draw_speedup_cold),
    ("cold-trials-ecdf", draw_ecdf),
    ("cold-trials-swarm", draw_beeswarm),
    ("cold-stages", draw_stages),
    ("speedup-warm", draw_speedup_warm),
    ("warm-leaderboard", draw_leaderboard),
    ("warm-qps", draw_qps),
    ("warm-relative", draw_relative),
    ("heatmap", draw_heatmap),
    ("report-card", draw_report_card),
    ("rank-bump", draw_bump),
    ("radar", draw_radar),
    ("memory-latency", draw_memory_latency),
    ("warm-cold-quadrant", draw_quadrant),
    ("cold-penalty-slope", draw_slope),
    ("cold-time-line", draw_number_line),
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", type=Path, default=OUT / "chart-data.json", help="the chart-data.json charts.py wrote")
    p.add_argument("--quiet", type=Path, default=QUIET, help="the quiet-window run's results folder")
    p.add_argument("--repo", type=Path, default=REPO, help="the root that paths in chart-data.json are relative to")
    p.add_argument("--out", type=Path, default=GALLERY, help="where the drawings go")
    p.add_argument("--only", nargs="*", help="draw only these, by file name")
    args = p.parse_args(argv)
    ctx = context(json.loads(args.data.read_text()), args.quiet, args.repo)
    args.out.mkdir(parents=True, exist_ok=True)
    m = _mpl()

    for name, theme in THEMES.items():
        with m.matplotlib.rc_context(_rc(theme)):
            for slug, draw in GALLERY_ORDER:
                if args.only and slug not in args.only:
                    continue

                draw(ctx, theme, args.out / (f"{slug}.svg" if name == "light" else f"{slug}-{name}.svg"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
