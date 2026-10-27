"""The write benchmark's charts for the README, from the tables of writes_report.py, drawn as charts.py draws the
head-to-head's: dark scalable vector graphics (SVG) with ColdIVF in blue.

    python writes_charts.py [--suite SUITE.toml] [--results DIR] [--out DIR]

writes-cold-95 and writes-cold-90 give each system's cold first query at S0 and at each state of the joint ladder,
one line per system that writes, at its pick for the target. writes-durable gives each system's p50 for one insert
that survives a power loss: an insert and a sync timed together, or for a system that saves its whole index, the save.
Drawing needs matplotlib, eg. the workspace's venv-plot.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import charts
import driver
import headtohead as h2h
import writes_report as wr

OUT = charts.OUT


def cold_lines(rows: list[dict]) -> tuple[list[str], list[dict]]:
    """The joint ladder's states, S0 first, and the rows that have a cold p50 at every one of them."""
    states = [s for s in wr.columns(rows) if wr.JOINT.fullmatch(s)]
    shown = [r for r in rows if all(r["cold"].get(s) for s in states)]
    return states, shown


def spread(values: list[float], gap: float) -> list[float]:
    """`values` moved apart, keeping their order, so that each is at least `gap` from the next, each moving as little
    as it can."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = list(values)

    for _ in range(50):
        moved = False

        for a, b in zip(order, order[1:]):
            if out[b] - out[a] < gap:
                push = (gap - (out[b] - out[a])) / 2
                out[a] -= push
                out[b] += push
                moved = True

        if not moved:
            break

    return out


def draw_cold(rows: list[dict], target: str, path: Path, t: dict) -> None:
    """One line per system across the joint ladder's states, on a log scale, labeled at its right end."""
    states, shown = cold_lines(rows)
    plot_h, left, plot_w = 300.0, 64.0, 330.0
    page = charts.Page(t, f"Cold first query after writes, recall@10 {target}", plot_h + 64)
    top = page.top + 18
    ax = page.axes(left, top, plot_w, plot_h)
    values = [r["cold"][s]["p50_us"] / 1000 for r in shown for s in states]
    lo, hi = min(values) / 1.4, max(values) * 1.4
    ax.set_yscale("log")
    ax.set_ylim(lo, hi)
    ax.set_xlim(-0.15, len(states) - 0.85)
    m = charts._mpl()
    ticks = [10.0 ** e for e in range(math.ceil(math.log10(lo)), math.floor(math.log10(hi)) + 1)]
    ax.yaxis.set_major_locator(m.FixedLocator(ticks))
    ax.yaxis.set_minor_locator(m.NullLocator())
    ax.set_yticklabels([f"{v:,g} ms" for v in ticks])
    ax.xaxis.set_major_locator(m.FixedLocator(range(len(states))))
    ax.set_xticklabels([s.upper() for s in states])
    ax.set_xlabel("State, from no writes to 10% of the vectors inserted and as many deleted", labelpad=6)
    charts._grid(ax, t, "y")
    # Labels at least 13 points apart, in powers of ten.
    per = plot_h / math.log10(hi / lo)
    ends = spread([math.log10(r["cold"][states[-1]]["p50_us"] / 1000) for r in shown], 13 / per)

    for r, end in sorted(zip(shown, ends), key=lambda x: x[0]["build"][:2] == wr.REFERENCE):
        ours = r["build"][:2] == wr.REFERENCE
        color = t["coldivf"] if ours else t["rest"]
        ys = [r["cold"][s]["p50_us"] / 1000 for s in states]
        ax.plot(range(len(states)), ys, color=color, linewidth=2.4 if ours else 1.2, zorder=4 if ours else 2)

        for x, y in enumerate(ys):
            charts._dot(ax, x, y, color, t, 6 if ours else 4.5, z=5 if ours else 3)

        last = ys[-1]
        label = f"{r['label']}  {wr.ms(last * 1000)} ms"
        y_pt = top + plot_h - (end - math.log10(lo)) * per
        page.text(left + plot_w + 10, y_pt + 3, label, size=8.5, color=t["ink"] if ours else t["ink2"],
                  weight="bold" if ours else "normal")

    page.save(path)


def durable_rows(rows: list[dict]) -> list[dict]:
    """Each writing system's p50 for one insert that survives a power loss, fastest first. A system with no such call,
    eg. AsterVec, whose close syncs none of its vectors, is left out."""
    out = []

    for r in rows:
        s0 = r["latency"].get("s0", {})
        synced, save = s0.get("insert_sync") or {}, s0.get("durable_point") or {}
        value = synced.get("p50_us") or save.get("p50_us")
        power = (synced.get("durability") if synced else save.get("durability")) == "power"

        if value and power:
            out.append({"label": r["label"], "ours": r["build"][:2] == wr.REFERENCE, "us": value,
                        "how": "insert and sync" if synced else "save of the whole index"})

    return sorted(out, key=lambda x: x["us"])


def draw_durable(rows: list[dict], path: Path, t: dict) -> None:
    """A dot per system at its durable insert's p50, on a log scale, with its name and how it becomes durable."""
    shown = durable_rows(rows)
    row_h, left, plot_w = 28.0, 250.0, 380.0
    page = charts.Page(t, "One insert that survives a power loss, p50", 30 + row_h * len(shown) + 40)
    top = page.top + 22
    ax = page.axes(left, top, plot_w, row_h * len(shown))
    lo = 10 ** math.floor(math.log10(min(x["us"] for x in shown) / 1000 / 1.2))
    hi = 10 ** math.ceil(math.log10(max(x["us"] for x in shown) / 1000 * 1.2))
    m = charts._mpl()
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(len(shown) - 0.5, -0.5)
    ax.yaxis.set_major_locator(m.NullLocator())
    ax.spines["left"].set_visible(False)
    charts._log_ticks(ax, lo, hi, fmt=lambda v: f"{v:,g} ms")
    charts._grid(ax, t)
    rowpos = m.blended(ax.transAxes, ax.transData)

    for i, x in enumerate(shown):
        color = t["coldivf"] if x["ours"] else t["rest"]
        ax.axhline(i, color=t["grid"], linewidth=0.6, zorder=0)
        charts._dot(ax, x["us"] / 1000, i, color, t, 7 if x["ours"] else 5.5)
        ax.text(-0.02, i, x["label"], transform=rowpos, ha="right", va="bottom", size=9,
                color=t["ink"] if x["ours"] else t["ink2"], weight="bold" if x["ours"] else "normal")
        ax.text(-0.02, i, x["how"], transform=rowpos, ha="right", va="top", size=7.5, color=t["ink2"])
        ax.text(1.02, i, f"{wr.ms(x['us'])} ms", transform=rowpos, ha="left", va="center", size=8.5,
                color=t["ink"] if x["ours"] else t["ink2"], weight="bold" if x["ours"] else "normal")

    page.save(path)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--suite", type=Path, default=wr.SUITE)
    p.add_argument("--results", type=Path, help="the suite's results folder, by default its results root and phase")
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args(argv)
    suite = driver.load_suite(args.suite)
    data = wr.tables(wr.Lines(h2h.Results(args.results or suite.results_root / suite.phase).lines))
    t = charts.THEME

    with charts._mpl().matplotlib.rc_context(charts._rc(t)):
        for target, rows in data["search"].items():
            path = args.out / f"writes-cold-{target[2:]}-dark.svg"
            draw_cold(rows, target, path, t)
            print(path)

        path = args.out / "writes-durable-dark.svg"
        draw_durable(data["writes"], path, t)
        print(path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
